// Fetch occurrences from GBIF for a taxon and persist to Supabase Storage.
// Mirrors fetch-species.mjs for iNaturalist but calls the GBIF occurrence API.
// Use for large / historical pulls (> 5 000 records or date span > 60 days).
//
//   GET /.netlify/functions/gbif-fetch?species=Morchella&lat=40&lng=-105&radius=500
//   Optional: d1=2020-01-01&d2=2022-12-31

import { supabaseConfigured, publicUrl } from '../lib/supabase-storage.mjs'
import { uploadJson, readJson } from '../lib/datasets-store.mjs'
import { requireUser, loadProfile, adminClient } from '../lib/auth.mjs'
import { clusterFeatures } from '../lib/cluster.mjs'
import { submitJob } from '../lib/job-queue.mjs'
import { measureSource } from '../lib/job-source.mjs'
import { viewerFrom } from '../lib/dataset-access.mjs'
import { earthEngineConfigured } from '../lib/ee-runner.mjs'

export const config = { timeout: 120 }

const GBIF_SEARCH = 'https://api.gbif.org/v1/occurrence/search'
const MAX_RECORDS = 10000
const PER_PAGE = 300

function slugify(s) {
  return String(s).trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'species'
}
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
}
function dayOfYear(dateStr) {
  if (!dateStr) return null
  const d = new Date(`${String(dateStr).slice(0, 10)}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return null
  const start = Date.UTC(d.getUTCFullYear(), 0, 0)
  return Math.floor((d.getTime() - start) / 86400000)
}

function gbifToFeature(rec) {
  const lat = rec.decimalLatitude
  const lng = rec.decimalLongitude
  if (lat == null || lng == null) return null
  const dateStr = rec.eventDate ? String(rec.eventDate).slice(0, 10) : null
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [lng, lat] },
    properties: {
      uuid: rec.occurrenceID || String(rec.key ?? rec.gbifID ?? '') || null,
      gbif_id: rec.key || rec.gbifID || null,
      species: rec.species || rec.scientificName || null,
      date: dateStr,
      day_of_year: dayOfYear(dateStr),
      location: rec.locality || rec.verbatimLocality || null,
      num_identification_agreements: null,
      elevation: rec.elevation ?? null,
      land_cover_label: null,
      ndvi: null,
      soil_moisture: null,
      solar_exposure: null,
      wind_exposure: null,
      water_retention: null,
      slope: null,
      aspect: null,
      cluster: null,
    },
  }
}

export default async (request) => {
  try {
    const auth = await requireUser(request)
    if (!auth.ok) return auth.response

    const url = new URL(request.url)
    const species = url.searchParams.get('species')?.trim()
    if (!species) return json({ ok: false, error: 'Missing "species"' }, 400)

    const lat = Number(url.searchParams.get('lat') ?? process.env.INAT_LAT ?? 40.0)
    const lng = Number(url.searchParams.get('lng') ?? process.env.INAT_LNG ?? -105.0)
    const radius = Number(url.searchParams.get('radius') ?? process.env.INAT_RADIUS ?? 500)
    const d1 = url.searchParams.get('d1') || null
    const d2 = url.searchParams.get('d2') || null

    // Approximate bounding box from centre + radius in km (1 deg ≈ 111 km).
    const deg = radius / 111
    const params = new URLSearchParams({
      scientificName: species,
      hasCoordinate: 'true',
      hasGeospatialIssue: 'false',
      basisOfRecord: 'HUMAN_OBSERVATION',
      decimalLatitude: `${(lat - deg).toFixed(4)},${(lat + deg).toFixed(4)}`,
      decimalLongitude: `${(lng - deg).toFixed(4)},${(lng + deg).toFixed(4)}`,
      limit: String(PER_PAGE),
      offset: '0',
    })
    if (d1) params.set('eventDate', `${d1},${d2 || new Date().toISOString().slice(0, 10)}`)

    const features = []
    let offset = 0
    let total = Infinity

    while (features.length < MAX_RECORDS && offset < total) {
      params.set('offset', String(offset))
      const resp = await fetch(`${GBIF_SEARCH}?${params.toString()}`)
      if (!resp.ok) return json({ ok: false, error: `GBIF API ${resp.status}` }, 502)
      const data = await resp.json()

      if (total === Infinity) total = Math.min(data.count ?? 0, MAX_RECORDS)
      for (const rec of data.results ?? []) {
        const feat = gbifToFeature(rec)
        if (feat) features.push(feat)
      }
      if (data.endOfRecords) break
      offset += PER_PAGE
    }

    if (!features.length) {
      return json({ ok: true, count: 0, species, slug: `gbif-${slugify(species)}`, path: null })
    }

    const clustered = clusterFeatures(features)
    const geojson = { type: 'FeatureCollection', features: clustered }
    const slug = `gbif-${slugify(species)}`
    let path = null

    if (supabaseConfigured() && clustered.length) {
      try {
        await uploadJson(`species/${slug}.geojson`, geojson)
        path = publicUrl(`species/${slug}.geojson`)
        const manifest = (await readJson('datasets.json')) || []
        const entry = { id: slug, label: `${species} (GBIF, ${clustered.length})`, path, count: clustered.length }
        const idx = manifest.findIndex((e) => e.id === slug)
        if (idx >= 0) manifest[idx] = entry; else manifest.push(entry)
        await uploadJson('datasets.json', manifest)
      } catch {
        // best-effort; fall through and return the geojson inline
      }
    }

    if (auth.user && path && clustered.length && earthEngineConfigured()) {
      try {
        const profile = await loadProfile(auth.user.id)
        const { job } = await submitJob({
          user: auth.user,
          profile,
          spec: { kind: 'enrich', source: { type: 'dataset', slug }, title: `Enrich ${species}` },
          counter: (spec) => measureSource(spec, { client: adminClient(), viewer: viewerFrom(auth) }),
        })
        if (job?.id) {
          const secret = process.env.WORKER_POKE_SECRET
          if (secret) {
            const ctrl = new AbortController()
            const t = setTimeout(() => ctrl.abort(), 500)
            fetch(new URL('/.netlify/functions/ee-worker', request.url), {
              method: 'POST', headers: { 'x-worker-secret': secret }, signal: ctrl.signal,
            }).catch(() => {}).finally(() => clearTimeout(t))
          }
        }
      } catch (e) {
        console.warn('Auto-enrich skipped:', String(e))
      }
    }

    return json({
      ok: true, count: clustered.length, species, slug, path,
      geojson: path ? undefined : geojson,
    })
  } catch (e) {
    return json({ ok: false, error: String(e?.message ?? e) }, 500)
  }
}
