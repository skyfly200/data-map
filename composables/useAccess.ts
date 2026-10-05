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

interface TileResult { status: AccessStatus, areas: AccessArea[], region: string | null, truncated: boolean }
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
const FAIL: TileResult = { status: 'unavailable', areas: [], region: null, truncated: false }

async function fetchTile(bbox: BBox, fetchFn: typeof fetch): Promise<TileResult> {
  const k = key(bbox)
  const hit = tileCache.get(k)
  if (hit) return hit
  try {
    const res = await fetchFn(`${ACCESS_ENDPOINT}?bbox=${k}`)
    if (!res.ok) return FAIL
    const p = parseAccessResponse(await res.json())
    const r: TileResult = { status: p.status, areas: p.areas, region: p.region, truncated: p.truncated === true }
    if (r.status !== 'unavailable') tileCache.set(k, r)
    return r
  } catch {
    return FAIL
  }
}

/** One tile; if the server truncated it, subdivide. A tile still truncated at max depth is "incomplete". */
async function resolveTile(bbox: BBox, fetchFn: typeof fetch, depth: number): Promise<{ results: TileResult[], incomplete: boolean }> {
  const r = await fetchTile(bbox, fetchFn)
  if (!r.truncated) return { results: [r], incomplete: false }
  if (depth >= MAX_SPLIT_DEPTH) return { results: [r], incomplete: true }
  const subs = await Promise.all(quadrants(bbox).map((q) => resolveTile(q, fetchFn, depth + 1)))
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
): Promise<{ status: AccessStatus, areas: AccessArea[], region: string | null }> {
  if (!bboxInColorado(bbox)) return { status: 'not-loaded', areas: [], region: null }
  const tiles = await pool(tileBbox(bbox), TILE_CONCURRENCY, (t) => resolveTile(t, fetchFn, 0))
  const results = tiles.flatMap((t) => t.results)
  const incomplete = tiles.some((t) => t.incomplete)
  const byId = new Map<string, AccessArea>()
  let n = 0
  for (const r of results) for (const a of r.areas) byId.set(a.id || `anon-${n++}`, a)
  const areas = [...byId.values()]
  const ok = results.filter((r) => r.status === 'loaded' || r.status === 'no-data')
  const region = results.find((r) => r.region)?.region ?? null
  if (!ok.length) {
    const unavailable = results.some((r) => r.status === 'unavailable')
    return { status: unavailable ? 'unavailable' : 'not-loaded', areas: [], region }
  }
  if (ok.length < results.length || incomplete) return { status: 'partial', areas, region }
  return { status: areas.length ? 'loaded' : 'no-data', areas, region }
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
