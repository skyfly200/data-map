// Foray score (WANT-17 phase 1). Pure, no framework imports.
//
// For a grid cell: the share of the cell's own finds that are (a) of a species
// currently in season and (b) inside the selected window, each weighted by how
// close that species' typical fruiting is to the selected day. Dividing by the
// cell's total finds keeps it effort-neutral like the `season` heatmap: a cell
// is not better for having been visited more. Only cells with finds are scored.

export const FORAY_MIN_SAMPLE = 3
export const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/**
 * Weight 0..1 for a species from the days its median find sits from the target
 * day and its fruiting-window width (IQR). Same shape as the dashboard ranking
 * (dist + iqr/2, lower is better), mapped to a weight with a 14-day half-scale.
 */
export function phenologyWeight(dist: number, iqr: number): number {
  const d = Math.max(0, Number(dist) || 0)
  const w = Math.max(0, Number(iqr) || 0)
  return 1 / (1 + (d + w * 0.5) / 14)
}

export interface SeasonSpeciesLike { name: string, dist: number, iqr: number }

/** Map of species -> weight for the in-season list. */
export function seasonWeights(species: SeasonSpeciesLike[]): Map<string, number> {
  const m = new Map<string, number>()
  for (const s of species || []) m.set(s.name, phenologyWeight(s.dist, s.iqr))
  return m
}

export interface ScoreInput {
  /** All finds in the cell (after any land-cover filter). */
  n: number
  /** In-window finds per in-season species. */
  speciesInWindow: Map<string, number>
}

export interface ScoreComponent { species: string, weight: number, count: number, contribution: number }

export interface ForayScore {
  /** 0..1 */
  score: number
  /** Unweighted share of the cell's finds that are in-season species in the window. */
  share: number
  n: number
  /** In-season, in-window finds. */
  seasonalN: number
  components: ScoreComponent[]
  /** True when n is below the minimum sample; such cells should not be shown. */
  thin: boolean
}

export function scoreCell(input: ScoreInput, weights: Map<string, number>, minSample = FORAY_MIN_SAMPLE): ForayScore {
  const n = input.n || 0
  const components: ScoreComponent[] = []
  let seasonalN = 0
  let weighted = 0
  for (const [species, count] of input.speciesInWindow || []) {
    const w = weights.get(species)
    if (!w || !count) continue
    seasonalN += count
    weighted += w * count
    components.push({ species, weight: w, count, contribution: n ? (w * count) / n : 0 })
  }
  components.sort((a, b) => b.contribution - a.contribution)
  return {
    score: n ? Math.min(1, weighted / n) : 0,
    share: n ? Math.min(1, seasonalN / n) : 0,
    n, seasonalN, components,
    thin: n < minSample,
  }
}

/** Day of year for the middle of a month (1..12), non-leap. */
export function monthMidDay(month: number): number {
  const cum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
  const m = Math.min(12, Math.max(1, Math.round(month)))
  return cum[m - 1] + 15
}

export interface TimeSelection { day: number, window: number, label: string }

/** The "now" / month selector. `today` is a day of the year. */
export function resolveTimeSelection(sel: 'now' | number, today: number, nowWindow = 14): TimeSelection {
  if (sel === 'now') return { day: today, window: nowWindow, label: 'Now' }
  const m = Math.min(12, Math.max(1, Math.round(Number(sel))))
  return { day: monthMidDay(m), window: 15, label: MONTH_LABELS[m - 1] }
}

/** Min-max normalise scores to a 0..1 `t` for colouring. */
export function normaliseScores<T extends { score: number }>(cells: T[]): (T & { t: number })[] {
  if (!cells.length) return []
  let lo = Infinity, hi = -Infinity
  for (const c of cells) { lo = Math.min(lo, c.score); hi = Math.max(hi, c.score) }
  return cells.map((c) => ({ ...c, t: hi === lo ? 0.5 : (c.score - lo) / (hi - lo) }))
}

/** Highest score first; ties broken by larger sample then key for stability. */
export function rankCells<T extends { score: number, n: number, key: string }>(cells: T[], limit = Infinity): T[] {
  return [...cells]
    .sort((a, b) => b.score - a.score || b.n - a.n || (a.key < b.key ? -1 : 1))
    .slice(0, limit)
}

/** Plain-language band for the Forager view, from the normalised t. */
export function scoreBand(t: number): 'Great' | 'Good' | 'Fair' | 'Low' {
  return t >= 0.75 ? 'Great' : t >= 0.5 ? 'Good' : t >= 0.25 ? 'Fair' : 'Low'
}

/** Most common key of a count map, or null. */
export function modalKey(m: Map<string, number> | undefined): string | null {
  let best: string | null = null, bestN = 0
  for (const [k, n] of m || []) if (n > bestN) { best = k; bestN = n }
  return best
}
