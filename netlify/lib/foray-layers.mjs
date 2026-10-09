// Foray planner model layers (WANT-17 phases 2, 3 and 5).
//
//   ensemble  Phase 2: a weighted mean of saved suitability surfaces for the
//             species in season, weights from phenology closeness.
//   habitat   Phase 3: species-agnostic. The predictors that matter most across
//             the in-season models, scored by whether a pixel sits inside the
//             range the finds were made in. No model refit, so it is cheap.
//
// Both carry an `outside` band: 1 where any of the habitat predictors falls
// outside the range seen at any presence (the training envelope), so a
// "promising" pixel there is an extrapolation, not a finding (phase 5 guard).
//
// Weighting, predictor choice and the envelope are pure and tested here; the
// Earth Engine builders take `ee` as an argument so they run against a stub.

import { MAXENT_PREDICTORS, SUITABILITY_PALETTE, predictorStack } from './maxent.mjs'

export const FORAY_LAYERS = ['ensemble', 'habitat']
/** At most this many models feed a layer: each ensemble model is a refit. */
export const MAX_LAYER_MODELS = 6
/** Habitat score uses this many top predictors. */
export const HABITAT_TOP_PREDICTORS = 6
/** At most this many points per sample request. */
export const MAX_SAMPLE_POINTS = 1500

export class LayerSpecError extends Error {
  constructor(message) { super(message); this.status = 400 }
}

/**
 * `models=<id>:<weight>,<id>:<weight>` -> [{id, weight}]. Weights default to 1,
 * must be positive, and repeats keep the first. Throws LayerSpecError.
 */
export function parseModelWeights(raw) {
  const out = []
  const seen = new Set()
  for (const part of String(raw || '').split(',').map((s) => s.trim()).filter(Boolean)) {
    const [id, w] = part.split(':')
    if (!/^[0-9a-f-]{8,64}$/i.test(id || '')) throw new LayerSpecError(`"${id}" is not a model id.`)
    const weight = w == null || w === '' ? 1 : Number(w)
    if (!Number.isFinite(weight) || weight <= 0) throw new LayerSpecError(`Weight for ${id} must be a positive number.`)
    if (seen.has(id)) continue
    seen.add(id)
    out.push({ id, weight })
  }
  if (!out.length) throw new LayerSpecError('Pick at least one model.')
  if (out.length > MAX_LAYER_MODELS) throw new LayerSpecError(`At most ${MAX_LAYER_MODELS} models at once.`)
  return out
}

/** Weights scaled to sum to 1. */
export function normaliseWeights(models) {
  const total = models.reduce((t, m) => t + (m.weight > 0 ? m.weight : 0), 0)
  return total > 0 ? models.map((m) => ({ ...m, weight: (m.weight > 0 ? m.weight : 0) / total })) : models
}

/**
 * Which predictors the habitat score uses, with weights and ranges.
 *
 * `models`: [{weight, contributions: {p: %}, ranges: {p: {p25,p75,min,max}}}].
 * Each model's contributions are normalised over only its own predictors that
 * have a range (so a model with five predictors does not lose to one with
 * twenty), then averaged across models by `weight`. The top `top` predictors
 * are kept and renormalised.
 *
 * Range per predictor: the union of the models' interquartile ranges (the
 * habitat any of them finds typical), and the envelope is the union of their
 * min..max. Importance has no direction, which is why the range is stored.
 *
 * Returns { predictors: [{key, weight, p25, p75, min, max}], skipped: [ids] }.
 */
export function habitatPredictors(models, { top = HABITAT_TOP_PREDICTORS } = {}) {
  const score = new Map()
  const range = new Map()
  const skipped = []
  let used = 0
  for (const m of models) {
    const ranges = m.ranges || {}
    const keys = Object.keys(m.contributions || {})
      .filter((k) => MAXENT_PREDICTORS[k] && ranges[k] && Number(m.contributions[k]) > 0)
    const total = keys.reduce((t, k) => t + Number(m.contributions[k]), 0)
    if (!keys.length || !(total > 0) || !(m.weight > 0)) { skipped.push(m.id ?? null); continue }
    used += m.weight
    for (const k of keys) {
      score.set(k, (score.get(k) || 0) + m.weight * (Number(m.contributions[k]) / total))
      const r = ranges[k]
      const prev = range.get(k)
      range.set(k, prev
        ? { p25: Math.min(prev.p25, r.p25), p75: Math.max(prev.p75, r.p75), min: Math.min(prev.min, r.min), max: Math.max(prev.max, r.max) }
        : { p25: r.p25, p75: r.p75, min: r.min, max: r.max })
    }
  }
  const ranked = [...score.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])).slice(0, top)
  const kept = ranked.reduce((t, [, s]) => t + s, 0)
  return {
    predictors: kept > 0 && used > 0 ? ranked.map(([key, s]) => ({ key, weight: s / kept, ...range.get(key) })) : [],
    skipped,
  }
}

/** Legend for both layers: 0..1 on the suitability ramp. */
export function forayLayerLegend(layer) {
  return {
    type: 'ramp',
    unit: layer === 'habitat' ? 'habitat match' : 'suitability',
    min: '0', max: '1', stops: SUITABILITY_PALETTE,
  }
}

export const FORAY_LAYER_VIS = { min: 0, max: 1, palette: SUITABILITY_PALETTE }

/**
 * 1 where any habitat predictor is outside its envelope (min..max over the
 * presences of every model used), else 0.
 */
export function buildEnvelopeImage(ee, predictors) {
  if (!predictors.length) return ee.Image.constant(0).rename('outside')
  const stack = predictorStack(ee, predictors.map((p) => p.key))
  const outs = predictors.map((p) => {
    const band = stack.select(p.key)
    return band.lt(p.min).or(band.gt(p.max))
  })
  return ee.ImageCollection(outs).max().rename('outside')
}

/**
 * Habitat score: sum over predictors of weight * (p25 <= value <= p75), so 1
 * means every top predictor is in its typical range. Bands: habitat, outside.
 */
export function buildHabitatImage(ee, predictors) {
  const stack = predictorStack(ee, predictors.map((p) => p.key))
  const parts = predictors.map((p) => {
    const band = stack.select(p.key)
    return band.gte(p.p25).and(band.lte(p.p75)).multiply(p.weight)
  })
  const habitat = ee.ImageCollection(parts).sum().rename('habitat')
  return habitat.addBands(buildEnvelopeImage(ee, predictors))
}

/**
 * Ensemble: weighted mean of suitability surfaces, over only the models whose
 * surface covers a pixel (each is clipped to its own region), so a pixel one
 * model never projected onto is not dragged toward zero. Bands: ensemble,
 * outside (from the habitat predictors, when there are any).
 *
 * `surfaces`: [{image, weight}], images single-band 0..1.
 */
export function buildEnsembleImage(ee, surfaces, envelopePredictors = []) {
  const num = ee.ImageCollection(surfaces.map((s) => s.image.unmask(0).multiply(s.weight))).sum()
  const den = ee.ImageCollection(surfaces.map((s) => s.image.mask().gt(0).multiply(s.weight))).sum()
  const ensemble = num.divide(den).updateMask(den.gt(0)).rename('ensemble')
  return ensemble.addBands(buildEnvelopeImage(ee, envelopePredictors))
}

/** [[lon, lat], ...] -> validated points, or throws LayerSpecError. */
export function parseSamplePoints(raw) {
  if (!Array.isArray(raw) || !raw.length) throw new LayerSpecError('Send at least one point.')
  if (raw.length > MAX_SAMPLE_POINTS) throw new LayerSpecError(`At most ${MAX_SAMPLE_POINTS} points at once.`)
  return raw.map((p, i) => {
    const lon = Number(p?.[0]), lat = Number(p?.[1])
    if (!(lon >= -180 && lon <= 180 && lat >= -90 && lat <= 90)) throw new LayerSpecError(`Point ${i} is not a lon/lat pair.`)
    return [lon, lat]
  })
}

/** Evaluated sampleRegions output -> [{i, value, outside}] in input order (missing -> null). */
export function shapeSamples(fc, n, band) {
  const out = Array.from({ length: n }, (_, i) => ({ i, value: null, outside: null }))
  for (const f of fc?.features || []) {
    const i = Number(f.properties?.i)
    if (!(i >= 0 && i < n)) continue
    const v = Number(f.properties?.[band])
    const o = f.properties?.outside
    out[i] = { i, value: Number.isFinite(v) ? v : null, outside: o == null ? null : Number(o) > 0 }
  }
  return out
}
