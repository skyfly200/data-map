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

// ── Planned date or range ───────────────────────────────────────────────────

/** 'YYYY-MM-DD' -> day of the year (1..366), or null when malformed. */
export function dateToDoy(iso: string | null | undefined): number | null {
  const m = String(iso || '').match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (!m) return null
  const y = Number(m[1]), mo = Number(m[2]), d = Number(m[3])
  const t = Date.UTC(y, mo - 1, d)
  const back = new Date(t)
  if (back.getUTCMonth() !== mo - 1 || back.getUTCDate() !== d) return null
  return Math.round((t - Date.UTC(y, 0, 0)) / 86400000)
}

/** Whole days from `a` to `b` ('YYYY-MM-DD'), or null. */
export function daysBetween(a: string, b: string): number | null {
  const pa = String(a || '').match(/^(\d{4})-(\d{2})-(\d{2})$/), pb = String(b || '').match(/^(\d{4})-(\d{2})-(\d{2})$/)
  if (!pa || !pb || dateToDoy(a) == null || dateToDoy(b) == null) return null
  return Math.round((Date.UTC(+pb[1], +pb[2] - 1, +pb[3]) - Date.UTC(+pa[1], +pa[2] - 1, +pa[3])) / 86400000)
}

const shortDate = (iso: string) => {
  const [, mo, d] = iso.split('-').map(Number)
  return `${MONTH_LABELS[mo - 1]} ${d}`
}

/** Longest range the planner accepts; a longer one is a season, not a trip. */
export const MAX_RANGE_DAYS = 60

export interface Plan {
  kind: 'now' | 'date' | 'range' | 'month'
  /** 'YYYY-MM-DD' for date and range. */
  date?: string
  /** 'YYYY-MM-DD' end of a range (inclusive). */
  end?: string
  month?: number
}

/**
 * The planned day and how many days either side of it a find counts. A date
 * works like Now on that day. A range is centred on its middle day, with the
 * window widened by half its length so finds from its first and last days
 * count fully. A malformed date falls back to Now.
 */
export function resolvePlan(plan: Plan, today: number, nowWindow = 14, monthWindow = 15): TimeSelection {
  if (plan.kind === 'month' && plan.month) return resolveTimeSelection(plan.month, today, nowWindow, monthWindow)
  const start = plan.kind === 'date' || plan.kind === 'range' ? dateToDoy(plan.date) : null
  if (start == null) return resolveTimeSelection('now', today, nowWindow, monthWindow)
  if (plan.kind === 'range' && plan.end) {
    const span = daysBetween(plan.date!, plan.end)
    if (span != null && span > 0 && span <= MAX_RANGE_DAYS) {
      const half = span / 2
      const mid = ((Math.round(start + half) - 1) % 365) + 1
      return { day: mid, window: nowWindow + Math.ceil(half), label: `${shortDate(plan.date!)} – ${shortDate(plan.end)}` }
    }
  }
  return { day: start, window: nowWindow, label: shortDate(plan.date!) }
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

/**
 * What a place is graded against:
 * - now: the other places in the region on the planned date. The strongest
 *   places that day read Great even late in the season.
 * - season: places across the whole fruiting season (all years pooled), so a
 *   late-season Great means as good as a typical place at the height of it.
 * - history: places across every past season taken year by year, so the scale
 *   includes the good years and the poor ones.
 */
export type Grading = 'now' | 'season' | 'history'
export const GRADINGS: Grading[] = ['now', 'season', 'history']
export const GRADING_LABELS: Record<Grading, string> = { now: 'This date', season: 'Whole season', history: 'All past seasons' }

export interface ForayParams {
  /** Days off peak at which a species counts half (phenology half-scale). */
  halfScale: number
  /** Pseudo-finds that pull small samples toward the average. */
  priorFinds: number
  /** Fewest finds a place needs to be scored. */
  minSample: number
  /** Days either side of the chosen day that a find counts; 0 = the default for Now / a month. */
  window: number
  /** What a place is graded against (see Grading). */
  grading: Grading
  /** 0..1, how far model suitability can lift places with few finds (0 = finds only). */
  modelMix: number
}

export const DEFAULT_PARAMS: ForayParams = {
  halfScale: FORAY_HALF_SCALE, priorFinds: FORAY_PRIOR_FINDS, minSample: FORAY_MIN_SAMPLE,
  window: 0, grading: 'now', modelMix: 0.5,
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
  if (g && (GRADINGS as string[]).includes(g)) out.grading = g as Grading
  // Links from before the three grading modes: relative was this date, absolute a fixed scale.
  else if (g === 'relative') out.grading = 'now'
  else if (g === 'absolute') out.grading = 'season'
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

/** Linear-interpolated scoreQuantile of finite values (q in 0..1); null when empty. */
export function scoreQuantile(values: number[], q: number): number | null {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b)
  if (!v.length) return null
  const i = (v.length - 1) * Math.min(1, Math.max(0, q)), lo = Math.floor(i)
  return v[lo] + (v[Math.ceil(i)] - v[lo]) * (i - lo)
}

/**
 * Share of the reference places that sit at or above the top mark. The top
 * mark is the 90th percentile rather than the single best place, so one
 * outlier does not push every other place down a grade.
 */
export const GRADE_TOP_QUANTILE = 0.9

/** The value that reads as the top of the scale for a set of reference values; null without any above 0. */
export function referenceTop(values: number[], q = GRADE_TOP_QUANTILE): number | null {
  const top = scoreQuantile(values.filter((x) => x > 0), q)
  return top && top > 0 ? top : null
}

/**
 * Grade cells into `t` (0..1) by `key`, a value on the absolute scale (see
 * scoreUnits), as a share of `top`, the reference top mark (capped at 1).
 * Without a top (no reference yet) the cells grade against themselves.
 */
export function gradeScores<T extends { score: number }>(cells: T[], key: keyof T & string, top?: number | null): (T & { t: number })[] {
  const ref = top && top > 0 ? top : referenceTop(cells.map((c) => Number(c[key]) || 0))
  return cells.map((c) => ({ ...c, t: ref ? Math.min(1, Math.max(0, Number(c[key]) || 0) / ref) : 0 }))
}

/**
 * Days to sample a fruiting season at: `count` days evenly spread between the
 * 5th and 95th percentile of the finds' days of the year.
 */
export function seasonSampleDays(doys: number[], count = 10): number[] {
  const lo = scoreQuantile(doys, 0.05), hi = scoreQuantile(doys, 0.95)
  if (lo == null || hi == null) return []
  if (count <= 1 || hi - lo < 1) return [Math.round((lo + hi) / 2)]
  return Array.from({ length: count }, (_, i) => Math.round(lo + ((hi - lo) * i) / (count - 1)))
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
  /** The reference top mark the grade is a share of (absolute scale). */
  top: number | null
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
  ctx: { cells: { key: string, score: number, n: number }[], ranked: { key: string }[], species: SeasonSpeciesLike[], params: ForayParams, top?: number | null },
): Explanation {
  let num = 0, den = 0
  for (const c of ctx.cells) { num += c.score * c.n; den += c.n }
  const info = new Map((ctx.species || []).map((s) => [s.name, s]))
  const idx = ctx.ranked.findIndex((c) => c.key === cell.key)
  return {
    band: scoreBand(cell.t), t: cell.t, grading: ctx.params.grading,
    n: cell.n, seasonalN: cell.seasonalN, raw: cell.score, adj: cell.adj,
    mean: den ? num / den : 0, priorFinds: ctx.params.priorFinds,
    rank: idx + 1, of: ctx.ranked.length, top: ctx.top ?? null,
    blend: cell.blend && cell.blend.model != null ? cell.blend : null,
    value: cell.final,
    species: cell.components.map((x) => ({
      species: x.species, count: x.count, weight: x.weight, contribution: x.contribution,
      dist: info.get(x.species)?.dist ?? null, iqr: info.get(x.species)?.iqr ?? null,
    })),
  }
}
