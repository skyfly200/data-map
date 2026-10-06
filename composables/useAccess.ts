// Land-access data for the foray planner (WANT-17). Talks to the access
// endpoint (netlify/functions/access.mjs) and degrades to an explicit
// "not loaded" state when it is missing, empty, or the region is not loaded.
// Callers must treat anything but status === 'loaded' as "no access info".

import { computed } from 'vue'
import { ACCESS_STATUS_MESSAGE, bboxInColorado, parseAccessResponse } from './forayPlanner'
import type { AccessArea, AccessStatus } from './forayPlanner'

export const ACCESS_ENDPOINT = '/.netlify/functions/access'

type BBox = [number, number, number, number]
export const MAX_TILE_DEG = 3
export const MAX_SPLIT_DEPTH = 3
export const TILE_CONCURRENCY = 4

interface TileResult { status: AccessStatus, areas: AccessArea[], region: string | null, truncated: boolean, lines: any[] }

export interface FetchAccessOptions {
  /** Signed-in session token: sends include_sets=1 + Authorization so user/club areas come back. */
  token?: string | null
  /** Ask for OSM road/trail lines (the endpoint only returns them for bbox <= 0.5 deg). */
  lines?: boolean
}
const tileCache = new Map<string, TileResult>()
export function clearAccessCache() { tileCache.clear() }

/** Split a bbox into a grid of tiles no wider/taller than maxDeg. */
export function tileBbox([w, s, e, n]: BBox, maxDeg = MAX_TILE_DEG): BBox[] {
  const cols = Math.max(1, Math.ceil((e - w) / maxDeg - 1e-9))
  const rows = Math.max(1, Math.ceil((n - s) / maxDeg - 1e-9))
  const dx = (e - w) / cols, dy = (n - s) / rows
  const out: BBox[] = []
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      out.push([w + c * dx, s + r * dy, c === cols - 1 ? e : w + (c + 1) * dx, r === rows - 1 ? n : s + (r + 1) * dy])
    }
  }
  return out
}

const quadrants = ([w, s, e, n]: BBox): BBox[] => {
  const mx = (w + e) / 2, my = (s + n) / 2
  return [[w, s, mx, my], [mx, s, e, my], [w, my, mx, n], [mx, my, e, n]]
}

const key = (b: BBox) => b.map((n) => n.toFixed(4)).join(',')
const FAIL: TileResult = { status: 'unavailable', areas: [], region: null, truncated: false, lines: [] }

async function fetchTile(bbox: BBox, fetchFn: typeof fetch, opts: FetchAccessOptions = {}): Promise<TileResult> {
  const k = key(bbox)
  // Sets are per user, so a signed-in tile is cached apart from the public one.
  const ck = opts.token ? `${k}|u${opts.token.slice(-12)}${opts.lines ? '|l' : ''}` : `${k}${opts.lines ? '|l' : ''}`
  const hit = tileCache.get(ck)
  if (hit) return hit
  try {
    const url = `${ACCESS_ENDPOINT}?bbox=${k}${opts.token ? '&include_sets=1' : ''}${opts.lines ? '&lines=1' : ''}`
    // Plain URL-only call when there is no token, as before.
    const res = opts.token
      ? await fetchFn(url, { headers: { Authorization: `Bearer ${opts.token}` } })
      : await fetchFn(url)
    if (!res.ok) return FAIL
    const body = await res.json()
    const p = parseAccessResponse(body)
    const lines = Array.isArray(body?.lines?.features) ? body.lines.features : []
    const r: TileResult = { status: p.status, areas: p.areas, region: p.region, truncated: p.truncated === true, lines }
    if (r.status !== 'unavailable') tileCache.set(ck, r)
    return r
  } catch {
    return FAIL
  }
}

/** One tile; if the server truncated it, subdivide. A tile still truncated at max depth is "incomplete". */
async function resolveTile(bbox: BBox, fetchFn: typeof fetch, depth: number, opts: FetchAccessOptions = {}): Promise<{ results: TileResult[], incomplete: boolean }> {
  const r = await fetchTile(bbox, fetchFn, opts)
  if (!r.truncated) return { results: [r], incomplete: false }
  if (depth >= MAX_SPLIT_DEPTH) return { results: [r], incomplete: true }
  const subs = await Promise.all(quadrants(bbox).map((q) => resolveTile(q, fetchFn, depth + 1, opts)))
  return { results: subs.flatMap((x) => x.results), incomplete: subs.some((x) => x.incomplete) }
}

async function pool<T, R>(items: T[], limit: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length)
  let i = 0
  const worker = async () => { while (i < items.length) { const j = i++; out[j] = await fn(items[j]) } }
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker))
  return out
}

/**
 * Tiled fetch (endpoint caps bbox at 3 deg and 400 areas). Merges and de-dupes
 * areas by id. Any failed, unloaded or still-truncated tile makes the status
 * 'partial' (never 'loaded'); callers treat unmatched cells as unknown.
 * Never throws.
 */
export async function fetchAccess(
  bbox: BBox,
  fetchFn: typeof fetch = fetch,
  opts: FetchAccessOptions = {},
): Promise<{ status: AccessStatus, areas: AccessArea[], region: string | null, lines: any[] }> {
  if (!bboxInColorado(bbox)) return { status: 'not-loaded', areas: [], region: null, lines: [] }
  const tiles = await pool(tileBbox(bbox), TILE_CONCURRENCY, (t) => resolveTile(t, fetchFn, 0, opts))
  const results = tiles.flatMap((t) => t.results)
  const incomplete = tiles.some((t) => t.incomplete)
  const byId = new Map<string, AccessArea>()
  let n = 0
  for (const r of results) for (const a of r.areas) byId.set(a.id || `anon-${n++}`, a)
  const areas = [...byId.values()]
  const ok = results.filter((r) => r.status === 'loaded' || r.status === 'no-data')
  const lineById = new Map<string, any>()
  for (const r of results) for (const f of r.lines) lineById.set(String(f?.properties?.id ?? lineById.size), f)
  const lines = [...lineById.values()]
  const region = results.find((r) => r.region)?.region ?? null
  if (!ok.length) {
    const unavailable = results.some((r) => r.status === 'unavailable')
    return { status: unavailable ? 'unavailable' : 'not-loaded', areas: [], region, lines: [] }
  }
  if (ok.length < results.length || incomplete) return { status: 'partial', areas, region, lines }
  return { status: areas.length ? 'loaded' : 'no-data', areas, region, lines }
}

export function useAccess() {
  const status = useState<AccessStatus>('foray-access-status', () => 'idle')
  const areas = useState<AccessArea[]>('foray-access-areas', () => [])
  const region = useState<string | null>('foray-access-region', () => null)

  async function load(bbox: [number, number, number, number]) {
    status.value = 'loading'
    const r = await fetchAccess(bbox)
    areas.value = r.areas
    region.value = r.region
    status.value = r.status
  }

  const loaded = computed(() => status.value === 'loaded')
  const message = computed(() => ACCESS_STATUS_MESSAGE[status.value])

  return { status, areas, region, loaded, message, load }
}
