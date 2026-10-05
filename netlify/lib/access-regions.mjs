// access_ingest jobs and region loads (WANT-17). Fetches PAD-US, OSM roads/trails
// and (with RIDB_API_KEY) Recreation.gov fee records for a bbox, upserts them
// tagged with a region, and records the load in access_regions.
// Network and DB are injected (fetchImpl, client) so this runs against fakes.

import {
  OVERPASS_URL, PADUS_FEATURE_URL, applyRidbFees, overpassQuery, padusQueryUrl, parseOsmWays,
  parsePadUs, parseRidbFacilities, ridbUrl, tileBbox,
} from './access-ingest.mjs'
import { ACCESS_OSM_TILE_DEG } from './quotas.mjs'

/** Defined regions. bbox is [west, south, east, north], approximate state extent. */
export const REGIONS = {
  colorado: { name: 'colorado', bbox: [-109.06, 36.99, -102.04, 41.0] },
}

export const ALL_SOURCES = ['padus', 'osm', 'ridb']

export class AccessSpecError extends Error {}

/** Validate/normalise an access_ingest spec -> { kind, region, bbox, sources, force, title }. */
export function normaliseAccessSpec(input = {}) {
  const named = input.region ? REGIONS[String(input.region).toLowerCase()] : null
  let bbox = input.bbox
  if (typeof bbox === 'string') bbox = bbox.split(',').map(Number)
  if (!bbox && named) bbox = named.bbox
  if (!Array.isArray(bbox) || bbox.length !== 4 || bbox.some((v) => !Number.isFinite(Number(v)))) {
    throw new AccessSpecError('Provide a known region name or bbox [west, south, east, north].')
  }
  bbox = bbox.map(Number)
  const [w, s, e, n] = bbox
  if (!(w < e && s < n) || w < -180 || e > 180 || s < -90 || n > 90) throw new AccessSpecError('That bbox is not valid.')
  const sources = Array.isArray(input.sources) && input.sources.length ? input.sources.map(String) : ALL_SOURCES
  const bad = sources.filter((x) => !ALL_SOURCES.includes(x))
  if (bad.length) throw new AccessSpecError(`Unknown source: ${bad.join(', ')}.`)
  const round = (v) => Math.round(v * 100) / 100
  const region = named && !input.bbox ? named.name
    : String(input.region || '').trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-').slice(0, 60)
      || `bbox-${bbox.map(round).join('_')}`
  return {
    kind: 'access_ingest', region, bbox, sources: ALL_SOURCES.filter((x) => sources.includes(x)),
    force: input.force === true, title: String(input.title || `Load access data: ${region}`).slice(0, 120),
  }
}

const contains = (outer, inner) => outer[0] <= inner[0] && outer[1] <= inner[1] && outer[2] >= inner[2] && outer[3] >= inner[3]

/** A loaded region (status loaded) that already covers `bbox`, or null. */
export function findCoveringRegion(regions, bbox) {
  return (regions || []).find((r) => r.status === 'loaded' && Array.isArray(r.bbox) && contains(r.bbox, bbox)) || null
}

// ---- DB helpers -------------------------------------------------------------

export async function listRegions(client) {
  const { data, error } = await client.from('access_regions').select('*').order('name')
  if (error) throw new Error(error.message)
  return data || []
}

async function saveRegion(client, row) {
  const { error } = await client.from('access_regions').upsert(row, { onConflict: 'name' })
  if (error) throw new Error(`access_regions: ${error.message}`)
}

async function rpcChunks(client, fn, rows, region, size = 100) {
  let n = 0
  for (let i = 0; i < rows.length; i += size) {
    const { data, error } = await client.rpc(fn, { p_rows: rows.slice(i, i + size), p_region: region })
    if (error) throw new Error(`${fn}: ${error.message}`)
    n += Number(data) || 0
  }
  return n
}

// ---- Fetching ---------------------------------------------------------------

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function getJson(fetchImpl, url, init, { retries = 2, backoffMs = 5000, wait = sleep } = {}) {
  for (let attempt = 0; ; attempt++) {
    const res = await fetchImpl(url, init)
    if (res.ok) return res.json()
    // Overpass signals load with 429/504; back off and retry before giving up.
    if ((res.status === 429 || res.status === 504 || res.status === 503) && attempt < retries) {
      await wait(backoffMs * (attempt + 1))
      continue
    }
    throw new Error(`Upstream ${res.status} for ${String(url).slice(0, 80)}`)
  }
}

async function fetchPadusTile(fetchImpl, base, tile) {
  const features = []
  for (let offset = 0; ; offset += 500) {
    const fc = await getJson(fetchImpl, padusQueryUrl(base, tile, offset))
    features.push(...(fc.features || []))
    // ArcGIS sets exceededTransferLimit (top-level or in properties) when more pages remain.
    if (!(fc.exceededTransferLimit || fc.properties?.exceededTransferLimit)) break
    if (offset > 50000) break
  }
  return { type: 'FeatureCollection', features }
}

async function fetchRidbTile(fetchImpl, apiKey, tile) {
  const out = []
  for (let offset = 0; offset < 2000; offset += 50) {
    const json = await getJson(fetchImpl, ridbUrl(tile, offset), { headers: { apikey: apiKey } })
    const got = parseRidbFacilities(json)
    out.push(...got)
    if ((json.RECDATA || []).length < 50) break
  }
  return out
}

/**
 * Run one region load. Returns { status, areas, lines, ridbMatched, ridb }.
 * deadlineMs: stop starting new tiles after this wall-clock time (status 'partial';
 * upserts are idempotent, so re-running continues the work).
 */
export async function runAccessIngest({
  client, spec, userId = null, fetchImpl = fetch, env = process.env, onProgress = async () => {},
  deadlineMs = Infinity, wait = sleep, politeMs = 1000,
}) {
  const { region, bbox, sources } = spec
  const base = { name: region, bbox, sources, requested_by: userId, error: null }
  await saveRegion(client, { ...base, status: 'loading' })
  const pTiles = sources.includes('padus') ? tileBbox(bbox, 1) : []
  const oTiles = sources.includes('osm') ? tileBbox(bbox, ACCESS_OSM_TILE_DEG) : []
  const total = pTiles.length + oTiles.length || 1
  let done = 0, areas = 0, lines = 0, ridbMatched = 0, partial = false
  let ridb = sources.includes('ridb') ? (env.RIDB_API_KEY ? 'used' : 'skipped_no_key') : 'not_requested'
  const padusBase = env.PADUS_FEATURE_URL || PADUS_FEATURE_URL
  const overpass = env.OVERPASS_URL || OVERPASS_URL

  try {
    for (const tile of pTiles) {
      if (Date.now() > deadlineMs) { partial = true; break }
      const rows = parsePadUs(await fetchPadusTile(fetchImpl, padusBase, tile))
      if (ridb === 'used') {
        try { ridbMatched += applyRidbFees(rows, await fetchRidbTile(fetchImpl, env.RIDB_API_KEY, tile)).matched }
        catch (err) { ridb = `failed: ${String(err.message).slice(0, 80)}` } // estimates still stored
      }
      areas += await rpcChunks(client, 'access_upsert_areas', rows, region)
      await onProgress({ fraction: ++done / total, stage: 'padus', message: `PAD-US tile ${done}/${pTiles.length}` })
    }
    for (const tile of oTiles) {
      if (partial || Date.now() > deadlineMs) { partial = true; break }
      const json = await getJson(fetchImpl, overpass, {
        method: 'POST', headers: { 'content-type': 'application/x-www-form-urlencoded' },
        body: `data=${encodeURIComponent(overpassQuery(tile))}`,
      }, { wait })
      lines += await rpcChunks(client, 'access_upsert_lines', parseOsmWays(json), region, 500)
      await onProgress({ fraction: ++done / total, stage: 'osm', message: `OSM tile ${done - pTiles.length}/${oTiles.length}` })
      if (politeMs) await wait(politeMs)
    }
  } catch (err) {
    await saveRegion(client, { ...base, status: 'failed', area_count: areas, line_count: lines, error: String(err.message).slice(0, 300) })
    throw err
  }
  const status = partial ? 'partial' : 'loaded'
  await saveRegion(client, {
    ...base, status, area_count: areas, line_count: lines, ridb_matched: ridbMatched,
    loaded_at: new Date().toISOString(),
  })
  return { status, areas, lines, ridbMatched, ridb }
}
