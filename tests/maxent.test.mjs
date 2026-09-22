/* The modelling core: predictor registry, spec validation, the background plan,
 * and the Earth Engine builder exercised against a stub.
 *
 * The point of splitting maxent.mjs so the only EE-touching function takes `ee`
 * as an argument is that all of this runs without a live Earth Engine session.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  DEFAULT_PREDICTORS, MAX_BACKGROUND, MIN_BACKGROUND, MIN_PREDICTORS,
  backgroundPlan, buildSuitabilityImage, normaliseModelSpec, predictorStack,
  suitabilityLegend,
} from '../netlify/lib/maxent.mjs'
import { SpecError } from '../netlify/lib/ee-pipeline.mjs'

const region = { north: 40.25, south: 39.55, east: -105.15, west: -105.8 }

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
      if (prop === 'Filter') return { lt: rec('lt'), eq: rec('eq') }
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
      if (prop === 'Geometry') return { Rectangle: rec('Rectangle') }
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
