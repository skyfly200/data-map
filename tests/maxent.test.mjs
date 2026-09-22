/* The modelling core: predictor registry, spec validation, the background plan,
 * and the Earth Engine builder exercised against a stub.
 *
 * The point of splitting maxent.mjs so the only EE-touching function takes `ee`
 * as an argument is that all of this runs without a live Earth Engine session.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  CV_FOLDS, DEFAULT_BACKGROUND, DEFAULT_PREDICTORS, MAX_BACKGROUND, MIN_BACKGROUND, MIN_PREDICTORS,
  assignFold, backgroundPlan, buildSuitabilityImage, crossValidate, crossValidationSummary,
  estimateModelUnits, foldPoints, modelCacheKey, modelPlan, normaliseModelSpec, predictorStack,
  PREDICTOR_KEYS, randomBackground, rocAuc, suitabilityLegend,
} from '../netlify/lib/maxent.mjs'
import { normaliseSpec, SpecError } from '../netlify/lib/ee-pipeline.mjs'

const region = { north: 40.25, south: 39.55, east: -105.15, west: -105.8 }
const bboxSource = { type: 'bbox', bounds: region, taxon: 'Morchella' }

// ── Spec validation ──────────────────────────────────────────────────────────

test('a bare request gets the default predictors and background', () => {
  const spec = normaliseModelSpec({ region })
  assert.deepEqual(spec.predictors, DEFAULT_PREDICTORS)
  assert.equal(spec.background, 1000)
  assert.equal(spec.kind, 'model')
})

test('predictors are returned in registry order and deduplicated', () => {
  const spec = normaliseModelSpec({ region, predictors: ['ndvi', 'slope', 'ndvi', 'elevation'] })
  // Registry order is elevation, slope, aspect, ndvi, soil_moisture.
  assert.deepEqual(spec.predictors, ['elevation', 'slope', 'ndvi'])
})

test('an unknown predictor is refused by name', () => {
  assert.throws(() => normaliseModelSpec({ region, predictors: ['elevation', 'moon_phase'] }),
    (err) => err instanceof SpecError && /moon_phase/.test(err.message))
})

test('too few predictors is refused', () => {
  assert.throws(() => normaliseModelSpec({ region, predictors: ['elevation'] }),
    (err) => err instanceof SpecError && err.message.includes(String(MIN_PREDICTORS)))
})

test('the background count is clamped into range', () => {
  assert.equal(normaliseModelSpec({ region, background: 5 }).background, MIN_BACKGROUND)
  assert.equal(normaliseModelSpec({ region, background: 999999 }).background, MAX_BACKGROUND)
  assert.equal(normaliseModelSpec({ region, background: 2500.9 }).background, 2500)
})

test('a non-numeric background is refused rather than coerced to a default', () => {
  assert.throws(() => normaliseModelSpec({ region, background: 'lots' }), SpecError)
})

test('the source is validated through the injected normaliser', () => {
  const spec = normaliseModelSpec(
    { region, source: { type: 'dataset', slug: 'my-finds' } },
    { normaliseSource: (s) => ({ ...s, checked: true }) },
  )
  assert.equal(spec.source.checked, true)
})

// ── Dispatch through the shared entry point ──────────────────────────────────

test('normaliseSpec routes a model kind to the model normaliser', () => {
  const spec = normaliseSpec({ kind: 'model', source: bboxSource })
  assert.equal(spec.kind, 'model')
  assert.deepEqual(spec.predictors, DEFAULT_PREDICTORS)
  assert.equal(spec.source.type, 'bbox')
})

test('normaliseSpec still produces an enrichment spec by default', () => {
  const spec = normaliseSpec({ source: bboxSource })
  assert.equal(spec.kind, 'enrich')
  assert.ok(Array.isArray(spec.stages) && spec.stages.length)
})

test('an unknown job kind is still refused', () => {
  assert.throws(() => normaliseSpec({ kind: 'teleport', source: bboxSource }), SpecError)
})

// ── Cost and plan ────────────────────────────────────────────────────────────

test('the model estimate grows with points, predictors and background', () => {
  const base = estimateModelUnits({ points: 200, predictors: ['elevation', 'slope'], background: 500 })
  const morePts = estimateModelUnits({ points: 5000, predictors: ['elevation', 'slope'], background: 500 })
  const morePreds = estimateModelUnits({ points: 200, predictors: DEFAULT_PREDICTORS, background: 500 })
  assert.ok(morePts > base)
  assert.ok(morePreds > base)
  assert.ok(Number.isInteger(base) && base > 0)
})

test('the model plan is weighted phases from 0 to 1', () => {
  const plan = modelPlan()
  assert.deepEqual(plan.map((s) => s.key), ['presences', 'background', 'fit', 'validate', 'project'])
  assert.equal(plan[0].from, 0)
  assert.equal(plan[plan.length - 1].to, 1)
  // Monotonic, no gaps.
  for (let i = 1; i < plan.length; i += 1) assert.equal(plan[i].from, plan[i - 1].to)
})

// ── The background plan ──────────────────────────────────────────────────────

test('the background is never smaller than the presences', () => {
  const plan = backgroundPlan({ presenceCount: 3000, background: 1000 })
  assert.equal(plan.n, 3000)
})

test('effort weighting is on by default and passes through', () => {
  assert.equal(backgroundPlan({ presenceCount: 50 }).weighted, true)
  assert.equal(backgroundPlan({ presenceCount: 50, effortWeighted: false }).weighted, false)
})

test('too few presences is flagged, not silently modelled', () => {
  assert.equal(backgroundPlan({ presenceCount: 5 }).enough, false)
  assert.equal(backgroundPlan({ presenceCount: 500 }).enough, true)
})

test('the background plan never exceeds the maximum, even against many presences', () => {
  assert.equal(backgroundPlan({ presenceCount: 999999, background: 1000 }).n, MAX_BACKGROUND)
})

// ── The legend ───────────────────────────────────────────────────────────────

test('the suitability legend is a 0..1 ramp', () => {
  const legend = suitabilityLegend()
  assert.equal(legend.type, 'ramp')
  assert.equal(legend.min, '0')
  assert.equal(legend.max, '1')
  assert.ok(legend.stops.length >= 2)
})

// ── Result caching (V12-PERF-2) ──────────────────────────────────────────────

test('the model cache key is stable across equivalent requests', () => {
  const a = { predictors: ['ndvi', 'elevation'], background: 1000, source: { type: 'dataset', slug: 'x' } }
  const b = { predictors: ['elevation', 'ndvi'], background: 1000, source: { type: 'dataset', slug: 'x' } }
  // Predictor order does not change the model, so it must not change the key.
  assert.equal(modelCacheKey(a, region), modelCacheKey(b, region))
})

test('the model cache key separates models that differ', () => {
  const base = { predictors: DEFAULT_PREDICTORS, background: 1000, source: { type: 'dataset', slug: 'x' } }
  const key = modelCacheKey(base, region)
  assert.notEqual(key, modelCacheKey({ ...base, predictors: ['elevation', 'slope'] }, region))
  assert.notEqual(key, modelCacheKey({ ...base, source: { type: 'dataset', slug: 'y' } }, region))
  assert.notEqual(key, modelCacheKey(base, { ...region, north: region.north + 1 }))
})

test('a bbox source keys on its taxon and dates, a dataset on its slug', () => {
  const bbox = modelCacheKey({ source: { type: 'bbox', taxon: 'Morchella', dateFrom: '2020-01-01' } }, region)
  assert.match(bbox, /Morchella/)
  assert.match(modelCacheKey({ source: { type: 'dataset', slug: 'my-finds' } }, region), /dataset:my-finds/)
})

// ── Cross-validation: the pure math ──────────────────────────────────────────

test('AUC is 1 for perfect separation, 0.5 for a coin toss', () => {
  // Presences all score above background.
  assert.equal(rocAuc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1)
  // Reversed: presences all score below.
  assert.equal(rocAuc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]), 0)
  // A flat predictor: every score tied, so no separation.
  assert.equal(rocAuc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.5)
})

test('AUC needs both classes present', () => {
  assert.equal(rocAuc([0.9, 0.8], [1, 1]), null)
  assert.equal(rocAuc([0.1, 0.2], [0, 0]), null)
})

test('a known middling case scores between a half and one', () => {
  // One misranked pair out of four presence/background pairs → 0.75.
  const auc = rocAuc([0.9, 0.4, 0.6, 0.1], [1, 1, 0, 0])
  assert.equal(auc, 0.75)
})

test('a spatial block keeps its points together across folds', () => {
  // Two points in the same 0.25° block get the same fold; a far-away point may
  // differ. The point is that the fold is a property of the block, not the point.
  const a = assignFold(-105.30, 40.10, { seed: 3 })
  const b = assignFold(-105.28, 40.12, { seed: 3 }) // same 0.25° block
  assert.equal(a, b)
  assert.ok(a >= 0 && a < CV_FOLDS)
})

test('folded points carry presence and a fold in range', () => {
  const pres = [[-105.3, 40.1], [-105.2, 40.0]]
  const bg = randomBackground({ north: 40.3, south: 39.9, east: -105.0, west: -105.5 }, 10, 7)
  const folded = foldPoints(pres, bg, { seed: 1 })
  assert.equal(folded.length, 12)
  assert.equal(folded.filter((p) => p.presence === 1).length, 2)
  assert.ok(folded.every((p) => p.fold >= 0 && p.fold < CV_FOLDS))
})

test('the background draw is deterministic for a seed and inside the region', () => {
  const region = { north: 40.3, south: 39.9, east: -105.0, west: -105.5 }
  const a = randomBackground(region, 20, 42)
  const b = randomBackground(region, 20, 42)
  assert.deepEqual(a, b)
  assert.ok(a.every(([lon, lat]) => lon >= region.west && lon <= region.east
    && lat >= region.south && lat <= region.north))
})

test('the summary averages per-fold AUC with the spread beside it', () => {
  // Two folds, each perfectly separable → mean 1, sd 0.
  const rows = [
    { fold: 0, presence: 1, prob: 0.9 }, { fold: 0, presence: 0, prob: 0.1 },
    { fold: 1, presence: 1, prob: 0.8 }, { fold: 1, presence: 0, prob: 0.2 },
  ]
  const cv = crossValidationSummary(rows, { folds: 2 })
  assert.equal(cv.auc, 1)
  assert.equal(cv.sd, 0)
  assert.equal(cv.folds, 2)
  assert.equal(cv.grade, 'excellent')
})

test('a fold with only one class is dropped, not scored as zero', () => {
  const rows = [
    { fold: 0, presence: 1, prob: 0.9 }, { fold: 0, presence: 0, prob: 0.1 },
    { fold: 1, presence: 1, prob: 0.8 }, { fold: 1, presence: 1, prob: 0.7 }, // no background
  ]
  const cv = crossValidationSummary(rows, { folds: 2 })
  assert.equal(cv.folds, 1)
})

test('no scorable fold summarises to null rather than a fake number', () => {
  assert.equal(crossValidationSummary([], { folds: 4 }), null)
})

// ── The Earth Engine builder, against a stub ─────────────────────────────────

/**
 * A stub that records every method reached for, so the builder can be checked
 * for the shape of its Earth Engine calls without a session. Same idea as the
 * ee-tile-layers stub.
 */
function stubEe() {
  const calls = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => {
      if (prop === 'then') return undefined
      calls.push(String(prop))
      return chain
    },
    apply: () => chain,
  })
  // A helper that records its own name when called, so a static method like
  // Image.cat or FeatureCollection.randomPoints shows up in `calls` too.
  const rec = (name) => (...args) => { calls.push(name); return chain }
  const ee = new Proxy({}, {
    get: (t, prop) => {
      calls.push(String(prop))
      if (prop === 'Filter') return { lt: rec('lt'), eq: rec('eq'), neq: rec('neq') }
      if (prop === 'Image') {
        const img = rec('Image')
        img.cat = rec('cat')
        return img
      }
      if (prop === 'Terrain') return { slope: rec('slope'), aspect: rec('aspect') }
      if (prop === 'Classifier') return { amnhMaxent: rec('amnhMaxent') }
      if (prop === 'FeatureCollection') {
        const fc = rec('FeatureCollection')
        fc.randomPoints = rec('randomPoints')
        return fc
      }
      if (prop === 'Geometry') return { Rectangle: rec('Rectangle'), Point: rec('Point') }
      if (prop === 'Feature') return rec('Feature')
      return chain
    },
  })
  return { ee, calls }
}

test('predictorStack assembles one image per predictor', () => {
  const { ee, calls } = stubEe()
  predictorStack(ee, ['elevation', 'ndvi'])
  assert.ok(calls.includes('cat'), 'the bands are concatenated into one image')
})

test('the climate normals are available predictors and build', () => {
  // The per-date weather layers return to the model as their long-run means.
  assert.ok(['precip_normal', 'temp_normal'].every((k) => PREDICTOR_KEYS.includes(k)))
  const spec = normaliseModelSpec({ region, predictors: ['elevation', 'precip_normal', 'temp_normal'] })
  assert.deepEqual(spec.predictors, ['elevation', 'precip_normal', 'temp_normal'])
  const { ee } = stubEe()
  // Builds without throwing against the stub — the band-name and mean chain is
  // followed, not evaluated.
  assert.doesNotThrow(() => predictorStack(ee, spec.predictors))
})

test('cross-validation trains and classifies once per fold and selects three columns', () => {
  const { ee, calls } = stubEe()
  const stack = predictorStack(ee, DEFAULT_PREDICTORS)
  const points = foldPoints(
    [[-105.3, 40.1], [-105.2, 40.0]],
    randomBackground({ north: 40.3, south: 39.9, east: -105.0, west: -105.5 }, 8, 7),
    { seed: 1 },
  )
  crossValidate(ee, { stack, points, predictors: DEFAULT_PREDICTORS, folds: CV_FOLDS })
  // One train and one classify per fold, held out with neq/eq filters, merged
  // and reduced to the three columns the summary reads.
  assert.equal(calls.filter((c) => c === 'train').length, CV_FOLDS)
  assert.equal(calls.filter((c) => c === 'classify').length, CV_FOLDS)
  assert.ok(calls.includes('neq') && calls.includes('sampleRegions') && calls.includes('select'))
})

test('the builder trains MaxEnt on presences plus random background and classifies', () => {
  const { ee, calls } = stubEe()
  const out = buildSuitabilityImage(ee, {
    presences: {}, predictors: DEFAULT_PREDICTORS, background: 1000, region, seed: 1,
  })
  // The whole method chain a MaxEnt projection needs.
  for (const method of ['sampleRegions', 'randomPoints', 'amnhMaxent', 'train', 'classify', 'clip']) {
    assert.ok(calls.includes(method), `builder never called ${method}`)
  }
  // And it returns something paintable.
  assert.ok(out.image)
  assert.equal(out.vis.min, 0)
  assert.equal(out.vis.max, 1)
  assert.ok(Array.isArray(out.vis.palette))
})
