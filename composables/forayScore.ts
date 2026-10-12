// Foray score (WANT-17 phase 1). Pure, no framework imports.
//
// For a grid cell: the share of the cell's own finds that are (a) of a species
// currently in season and (b) inside the selected window, each weighted by how
// close that species' typical fruiting is to the selected day. Dividing by the
// cell's total finds keeps it effort-neutral like the `season` heatmap: a cell
// is not better for having been visited more. Only cells with finds are scored.

export const FORAY_MIN_SAMPLE = 3
export const FORAY_HALF_SCALE = 14
export const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

/**
 * Weight 0..1 for a species from the days its median find sits from the target
 * day and its fruiting-window width (IQR). Same shape as the dashboard ranking
 * (dist + iqr/2, lower is better), mapped to a weight with a `halfScale`-day
 * half-scale (14 by default): a species that far off its peak counts half.
 */
export function phenologyWeight(dist: number, iqr: number, halfScale = FORAY_HALF_SCALE): number {
  const d = Math.max(0, Number(dist) || 0)
  const w = Math.max(0, Number(iqr) || 0)
  const h = Math.max(1, Number(halfScale) || FORAY_HALF_SCALE)
  return 1 / (1 + (d + w * 0.5) / h)
}

export interface SeasonSpeciesLike { name: string, dist: number, iqr: number }

/** Map of species -> weight for the in-season list. */
export function seasonWeights(species: SeasonSpeciesLike[], halfScale = FORAY_HALF_SCALE): Map<string, number> {
  const m = new Map<string, number>()
  for (const s of species || []) m.set(s.name, phenologyWeight(s.dist, s.iqr, halfScale))
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
export function resolveTimeSelection(sel: 'now' | number, today: number, nowWindow = 14, monthWindow = 15): TimeSelection {
  if (sel === 'now') return { day: today, window: nowWindow, label: 'Now' }
  const m = Math.min(12, Math.max(1, Math.round(Number(sel))))
  return { day: monthMidDay(m), window: monthWindow, label: MONTH_LABELS[m - 1] }
}

/**
 * Pull each cell's score toward the pooled mean by `priorFinds` pseudo-finds,
 * so a cell with 3 finds that happen to be one in-season species does not
 * outrank a 40-find cell with a slightly lower share. Adds `adj`; `score` is kept.
 */
export const FORAY_PRIOR_FINDS = 5
export function adjustForSample<T extends { score: number, n: number }>(cells: T[], priorFinds = FORAY_PRIOR_FINDS): (T & { adj: number })[] {
  let num = 0, den = 0
  for (const c of cells) { num += c.score * c.n; den += c.n }
  const mean = den ? num / den : 0
  return cells.map((c) => ({ ...c, adj: (c.score * c.n + mean * priorFinds) / (c.n + priorFinds) }))
}

/** Min-max normalise `key` (default `score`) to a 0..1 `t` for colouring. All-zero cells stay at 0. */
export function normaliseScores<T extends { score: number }>(cells: T[], key: keyof T & string = 'score'): (T & { t: number })[] {
  if (!cells.length) return []
  const v = (c: T) => Number(c[key]) || 0
  let lo = Infinity, hi = -Infinity
  for (const c of cells) { lo = Math.min(lo, v(c)); hi = Math.max(hi, v(c)) }
  return cells.map((c) => ({ ...c, t: hi === lo ? (hi > 0 ? 0.5 : 0) : (v(c) - lo) / (hi - lo) }))
}

/** Highest `key` (default `score`) first; ties broken by larger sample then key for stability. */
export function rankCells<T extends { score: number, n: number, key: string }>(cells: T[], limit = Infinity, by: keyof T & string = 'score'): T[] {
  const v = (c: T) => Number(c[by]) || 0
  return [...cells]
    .sort((a, b) => v(b) - v(a) || b.n - a.n || (a.key < b.key ? -1 : 1))
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

// ── Tunable scoring (user controls) ─────────────────────────────────────────

export type Grading = 'relative' | 'absolute'

export interface ForayParams {
  /** Days off peak at which a species counts half (phenology half-scale). */
  halfScale: number
  /** Pseudo-finds that pull small samples toward the average. */
  priorFinds: number
  /** Fewest finds a place needs to be scored. */
  minSample: number
  /** Days either side of the chosen day that a find counts; 0 = the default for Now / a month. */
  window: number
  /** relative: graded against the other places on the map; absolute: fixed cut-offs. */
  grading: Grading
  /** 0..1, how far model suitability can lift places with few finds (0 = finds only). */
  modelMix: number
}

export const DEFAULT_PARAMS: ForayParams = {
  halfScale: FORAY_HALF_SCALE, priorFinds: FORAY_PRIOR_FINDS, minSample: FORAY_MIN_SAMPLE,
  window: 0, grading: 'relative', modelMix: 0.5,
}

/** Allowed range and step per numeric param, shared by the sliders and the URL parser. */
export const PARAM_LIMITS: Record<'halfScale' | 'priorFinds' | 'minSample' | 'window' | 'modelMix', { min: number, max: number, step: number }> = {
  halfScale: { min: 3, max: 45, step: 1 },
  priorFinds: { min: 0, max: 30, step: 1 },
  minSample: { min: 1, max: 20, step: 1 },
  window: { min: 0, max: 45, step: 1 },
  modelMix: { min: 0, max: 1, step: 0.05 },
}

const URL_KEYS: Record<keyof ForayParams, string> = {
  halfScale: 'fh', priorFinds: 'fp', minSample: 'fm', window: 'fw', grading: 'fg', modelMix: 'fx',
}

const clampParam = (k: keyof typeof PARAM_LIMITS, v: unknown): number => {
  const { min, max, step } = PARAM_LIMITS[k]
  const n = Number(v)
  if (!Number.isFinite(n)) return DEFAULT_PARAMS[k]
  return Number(Math.min(max, Math.max(min, Math.round(n / step) * step)).toFixed(6))
}

/** Params from a query string (or URLSearchParams); unknown or out-of-range values fall back or clamp. */
export function parseParams(q: URLSearchParams | string | null | undefined): ForayParams {
  const sp = q instanceof URLSearchParams ? q : new URLSearchParams(q || '')
  const out = { ...DEFAULT_PARAMS }
  for (const k of Object.keys(PARAM_LIMITS) as (keyof typeof PARAM_LIMITS)[]) {
    const raw = sp.get(URL_KEYS[k])
    if (raw !== null && raw !== '') out[k] = clampParam(k, raw)
  }
  const g = sp.get(URL_KEYS.grading)
  if (g === 'relative' || g === 'absolute') out.grading = g
  return out
}

/** Only the params that differ from the defaults, as query entries, so a default plan keeps a clean URL. */
export function paramsToQuery(p: ForayParams): Record<string, string> {
  const out: Record<string, string> = {}
  for (const k of Object.keys(URL_KEYS) as (keyof ForayParams)[]) {
    if (p[k] !== DEFAULT_PARAMS[k]) out[URL_KEYS[k]] = String(p[k])
  }
  return out
}

export const paramsAreDefault = (p: ForayParams) => !Object.keys(paramsToQuery(p)).length

/**
 * Adjusted score (or blend) that reads as a full-strength place under absolute
 * grading. A sample-adjusted score is a weighted share of finds, and in practice
 * even excellent places rarely pass ~0.5, so 0.5 maps to the top of the scale:
 * Great from 0.375, Good from 0.25, Fair from 0.125.
 */
export const ABSOLUTE_FULL = 0.5

/** An adjusted share on the absolute scale (1 = top mark); not capped, so ranking keeps its order above it. */
export const scoreUnits = (adj: number) => Math.max(0, Number(adj) || 0) / ABSOLUTE_FULL

/** Absolute-scale value as a 0..1 `t`, capped at 1. */
export const absoluteT = (adj: number) => Math.min(1, scoreUnits(adj))

/**
 * Grade cells into `t` (0..1) by `key`, a value already on the absolute scale
 * (see scoreUnits): min-max across the cells (relative) or capped as-is (absolute).
 */
export function gradeScores<T extends { score: number }>(cells: T[], key: keyof T & string, grading: Grading): (T & { t: number })[] {
  if (grading === 'absolute') return cells.map((c) => ({ ...c, t: Math.min(1, Math.max(0, Number(c[key]) || 0)) }))
  return normaliseScores(cells, key)
}

// ── Model blend (places with few or no finds) ───────────────────────────────

/**
 * Weight of the finds score when blending with a model's suitability (both on
 * the absolute scale). Finds count at least `1 - mix`, and more as finds pile
 * up (`n / (n + k)`, k rising with mix): a place with many finds stays
 * finds-led, and an unvisited one gets at most `mix` of the model. mix 0 means
 * finds only; mix 1 lets an unvisited place be graded by the model alone.
 */
export function blendWeight(n: number, mix: number): number {
  const m = Math.min(1, Math.max(0, Number(mix) || 0))
  if (m === 0) return 1
  if (m === 1) return 0
  const k = (m / (1 - m)) * 10
  const finds = Math.max(0, Number(n) || 0)
  return Math.max(1 - m, finds / (finds + k))
}

export interface BlendParts { findsT: number, model: number | null, findsWeight: number, blend: number }

/** Blended score on the absolute scale; without a model value it is the finds score alone. */
export function blendScore(adj: number, n: number, model: number | null | undefined, mix: number): BlendParts {
  const findsT = scoreUnits(adj)
  if (model == null || !Number.isFinite(Number(model))) return { findsT, model: null, findsWeight: 1, blend: findsT }
  const m = Math.min(1, Math.max(0, Number(model)))
  const a = blendWeight(n, mix)
  return { findsT, model: m, findsWeight: a, blend: a * findsT + (1 - a) * m }
}

// ── Transparency ────────────────────────────────────────────────────────────

export interface ExplainSpecies { species: string, count: number, weight: number, dist: number | null, iqr: number | null, contribution: number }

export interface Explanation {
  band: ReturnType<typeof scoreBand>
  t: number
  grading: Grading
  n: number
  seasonalN: number
  raw: number
  adj: number
  mean: number
  priorFinds: number
  rank: number
  of: number
  blend: BlendParts | null
  /** The value graded, on the absolute scale (blend when a model is mixed in). */
  value: number
  species: ExplainSpecies[]
}

/**
 * Everything the "Why this grade?" card shows for one cell. `ctx.cells` are the
 * finds-scored cells (for the mean), `ctx.ranked` every graded cell best-first
 * (for the rank). Shows the raw score, the
 * sample adjustment toward the mean, the optional model blend, the grade's
 * position and the species behind it with their peak distance and weight.
 */
export function explainCell(
  cell: { key: string, n: number, seasonalN: number, score: number, adj: number, t: number, final: number, components: ScoreComponent[], blend?: BlendParts | null },
  ctx: { cells: { key: string, score: number, n: number }[], ranked: { key: string }[], species: SeasonSpeciesLike[], params: ForayParams },
): Explanation {
  let num = 0, den = 0
  for (const c of ctx.cells) { num += c.score * c.n; den += c.n }
  const info = new Map((ctx.species || []).map((s) => [s.name, s]))
  const idx = ctx.ranked.findIndex((c) => c.key === cell.key)
  return {
    band: scoreBand(cell.t), t: cell.t, grading: ctx.params.grading,
    n: cell.n, seasonalN: cell.seasonalN, raw: cell.score, adj: cell.adj,
    mean: den ? num / den : 0, priorFinds: ctx.params.priorFinds,
    rank: idx + 1, of: ctx.ranked.length,
    blend: cell.blend && cell.blend.model != null ? cell.blend : null,
    value: cell.final,
    species: cell.components.map((x) => ({
      species: x.species, count: x.count, weight: x.weight, contribution: x.contribution,
      dist: info.get(x.species)?.dist ?? null, iqr: info.get(x.species)?.iqr ?? null,
    })),
  }
}
