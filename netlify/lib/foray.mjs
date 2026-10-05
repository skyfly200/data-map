// Foray planner core (WANT-17 phases 3 and 5). Pure.

/** Normalise one model's contributions over only its own predictors -> fractions summing to 1. */
export function normaliseContributions(contribs) {
  const e = Object.entries(contribs || {}).filter(([, v]) => Number.isFinite(v) && v > 0)
  const total = e.reduce((s, [, v]) => s + v, 0)
  return total ? Object.fromEntries(e.map(([k, v]) => [k, v / total])) : {}
}

/** Average normalised contributions across models. A predictor absent from a model counts 0 for it. */
export function mergeContributions(models) {
  const norm = models.map((m) => normaliseContributions(m.contributions)).filter((n) => Object.keys(n).length)
  const out = {}
  for (const n of norm) for (const [k, v] of Object.entries(n)) out[k] = (out[k] || 0) + v / norm.length
  return out
}

/** Preferred range of a field across observation values: interquartile range plus extent. */
export function preferredRange(values) {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b)
  if (v.length < 4) return null
  const q = (p) => { const i = (v.length - 1) * p, lo = Math.floor(i); return v[lo] + (v[Math.ceil(i)] - v[lo]) * (i - lo) }
  return { p25: q(0.25), p75: q(0.75), min: v[0], max: v[v.length - 1] }
}

/** Weighted fraction of top predictors whose cell value lies in the preferred range. */
export function habitatScore(cell, weights, ranges, { top = 5 } = {}) {
  const ks = Object.entries(weights).sort((a, b) => b[1] - a[1]).slice(0, top)
    .filter(([k]) => ranges[k] && Number.isFinite(cell[k]))
  const w = ks.reduce((s, [, x]) => s + x, 0)
  if (!w) return null
  return ks.reduce((s, [k, x]) => s + (cell[k] >= ranges[k].p25 && cell[k] <= ranges[k].p75 ? x : 0), 0) / w
}

/** True when any scored predictor falls outside the training min..max envelope (extrapolation). */
export function outsideEnvelope(cell, ranges) {
  return Object.entries(ranges).some(([k, r]) => Number.isFinite(cell[k]) && (cell[k] < r.min || cell[k] > r.max))
}

/**
 * Under-sampled opportunity: cells passing the access mask, ranked by promise / (1 + effort).
 * cell: { id, promise, finds, access: { publicLand, trailM }, envelope? }
 */
export function rankOpportunities(cells, { maxTrailM = 5000, requirePublic = true, auc = null, limit = 20 } = {}) {
  return cells
    .filter((c) => c.promise != null && (!requirePublic || c.access?.publicLand) && (c.access?.trailM ?? Infinity) <= maxTrailM)
    .map((c) => {
      const reach = 1 / (1 + (c.access?.trailM ?? maxTrailM) / 1000) // observer-reach proxy
      const effort = (c.finds || 0) + reach
      return { ...c, effort, rank: c.promise / (1 + effort), extrapolated: !!c.envelope, auc }
    })
    .sort((a, b) => b.rank - a.rank).slice(0, limit)
}
