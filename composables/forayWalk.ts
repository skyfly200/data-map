// Parking and walk-in estimate for the foray planner (WANT-17). Pure, no
// framework imports. Distances come from /.netlify/functions/access-nearest
// (nearest OSM parking, accessible parking, trailhead and road to a cell's
// centre); this turns them into a tier, a label and a filter.

export interface NearPoint { m: number, name: string | null, lon: number, lat: number, fee?: string | null }
export interface NearestInfo {
  parking: NearPoint | null
  accessible: NearPoint | null
  trailhead: NearPoint | null
  road_m: number | null
}

export type WalkTier = 'roadside' | 'short' | 'hike' | 'far' | 'unknown'
export type MaxWalk = 'any' | 'roadside' | 'short' | 'hike'

/** Tier boundaries in metres: a quarter mile, a mile, and the lookup radius. */
export const ROADSIDE_M = 400
export const SHORT_M = 1609
export const HIKE_M = 5000
/** Accessible parking within this of the centre counts as accessible. */
export const ACCESSIBLE_M = ROADSIDE_M

export const WALK_LABELS: Record<WalkTier, string> = {
  roadside: 'Roadside', short: 'Short walk', hike: 'Hike in', far: 'Remote', unknown: 'Walk unknown',
}
export const MAX_WALK_OPTIONS: { value: MaxWalk, label: string }[] = [
  { value: 'any', label: 'Any' },
  { value: 'roadside', label: 'Roadside (¼ mi)' },
  { value: 'short', label: 'Short walk (1 mi)' },
  { value: 'hike', label: 'Hike (3 mi)' },
]

/** Metres from the cell centre to the nearer of parking and a road; null when neither is within reach. */
export function walkMetres(info: NearestInfo | null | undefined): number | null {
  if (!info) return null
  const d = [info.parking?.m, info.road_m].filter((v): v is number => v != null && Number.isFinite(Number(v))).map(Number)
  return d.length ? Math.min(...d) : null
}

/**
 * Tier for a cell. `loaded` says the access data covers it: then "nothing within
 * reach" is a real answer (remote), otherwise it is unknown.
 */
export function walkTier(info: NearestInfo | null | undefined, loaded: boolean): WalkTier {
  if (!info) return 'unknown'
  const m = walkMetres(info)
  if (m == null) return loaded ? 'far' : 'unknown'
  return m <= ROADSIDE_M ? 'roadside' : m <= SHORT_M ? 'short' : m <= HIKE_M ? 'hike' : 'far'
}

export const isAccessible = (info: NearestInfo | null | undefined) => Boolean(info?.accessible && info.accessible.m <= ACCESSIBLE_M)

const RANK: Record<WalkTier, number> = { roadside: 0, short: 1, hike: 2, far: 3, unknown: 4 }
const LIMIT: Record<MaxWalk, number> = { any: 9, roadside: 0, short: 1, hike: 2 }

export interface WalkFilter { maxWalk: MaxWalk, accessible: boolean, keepUnknown: boolean }

/** Does a cell's walk pass the filter? Unknown passes only when kept (or nothing is filtered). */
export function passesWalk(tier: WalkTier, accessible: boolean, f: WalkFilter): boolean {
  const active = f.maxWalk !== 'any' || f.accessible
  if (!active) return true
  if (tier === 'unknown') return f.keepUnknown
  if (RANK[tier] > LIMIT[f.maxWalk]) return false
  return !f.accessible || accessible
}

export const miles = (m: number) => m / 1609.344

/** "0.3 mi walk from parking" style text for the card; null when unknown. */
export function walkText(info: NearestInfo | null | undefined, tier: WalkTier): string | null {
  if (tier === 'unknown') return null
  if (tier === 'far') return `No parking or road within ${miles(HIKE_M).toFixed(0)} mi`
  const m = walkMetres(info)!
  const from = info?.parking && info.parking.m === m ? 'parking' : 'a road'
  const d = miles(m)
  return `${d < 0.1 ? '<0.1' : d.toFixed(1)} mi from ${from}`
}

/** Where to drive to: the nearest parking when known, else the place itself. */
export function driveTarget(cell: { lat: number, lon: number }, info: NearestInfo | null | undefined): { lat: number, lon: number, parking: boolean } {
  const p = info?.parking
  return p ? { lat: p.lat, lon: p.lon, parking: true } : { lat: cell.lat, lon: cell.lon, parking: false }
}

/** Keys still to look up, at most `max`, in the order given (callers pass best-first). */
export function keysToFetch(cells: { key: string }[], have: Map<string, unknown>, max: number): string[] {
  const out: string[] = []
  for (const c of cells) {
    if (out.length >= max) break
    if (!have.has(c.key)) out.push(c.key)
  }
  return out
}
