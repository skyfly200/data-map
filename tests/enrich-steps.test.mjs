import test from 'node:test'
import assert from 'node:assert/strict'
import {
  STAGES, STAGE_KEYS, STAGE_GROUPS, DEFAULT_STAGES, cleanStages, describeStages, normaliseSpec,
} from '../netlify/lib/ee-pipeline.mjs'
import { EE_TILE_LAYERS, resolveLayer } from '../netlify/lib/ee-tile-layers.mjs'
import { stepGroups, resolveDefault, sameSteps } from '../composables/useEnrichSteps.ts'

const mapSteps = STAGE_KEYS.filter((k) => STAGES[k].layer)

test('every step has a group the picker knows', () => {
  for (const key of STAGE_KEYS) assert.ok(STAGE_GROUPS.includes(STAGES[key].group), `${key} has group ${STAGES[key].group}`)
})

test('map-layer steps name a real, static, single-value layer', () => {
  assert.ok(mapSteps.length >= 5)
  for (const key of mapSteps) {
    const s = STAGES[key]
    const layer = EE_TILE_LAYERS[s.layer]
    assert.ok(layer, `${key} → ${s.layer} is not a map layer`)
    assert.equal(s.perDate, false, `${key} must not move with the date`)
    assert.equal(s.bands.length, 1, `${key} writes one column`)
    assert.ok(!layer.rgb && !layer.prepare && !layer.reference, `${key} is not a plain single-band layer`)
    // Fixed parameters must be ones the layer accepts.
    assert.doesNotThrow(() => resolveLayer(s.layer, s.layerParams || {}))
    if (s.classes) assert.equal(layer.legend?.type, 'classes')
  }
})

test('map-layer steps do not reuse a column another step writes', () => {
  const seen = new Map()
  for (const key of STAGE_KEYS) {
    for (const band of STAGES[key].bands) {
      assert.ok(!seen.has(band), `${band} written by both ${seen.get(band)} and ${key}`)
      seen.set(band, key)
    }
  }
})

test('a job may ask for map-layer steps', () => {
  const spec = normaliseSpec({
    stages: ['stand_size', 'hand', 'terrain'],
    source: { type: 'dataset', slug: 'foray-points' },
  })
  assert.deepEqual(spec.stages, ['terrain', 'hand', 'stand_size'])
})

test('the standard set is unchanged', () => {
  assert.deepEqual(DEFAULT_STAGES, ['terrain', 'landcover', 'soil_moisture', 'precip', 'temperature', 'ndvi'])
})

test('cleanStages drops unknown steps and orders by catalogue', () => {
  assert.deepEqual(cleanStages(['hand', 'gone', 'terrain', 'hand']), ['terrain', 'hand'])
  assert.deepEqual(cleanStages(null), [])
})

test('describeStages records each step with its source and columns', () => {
  const steps = describeStages(['terrain', 'field_capacity', 'nope'])
  assert.equal(steps.length, 2)
  assert.equal(steps[0].label, 'Terrain')
  assert.equal(steps[1].layer, 'field-capacity')
  assert.deepEqual(steps[1].layerParams, { depth: '0' })
  assert.deepEqual(steps[1].bands, ['field_capacity_pct'])
  assert.ok(steps.every((s) => s.asset))
})

test('saved defaults fall back to the standard set when empty or stale', () => {
  assert.deepEqual(resolveDefault(null), DEFAULT_STAGES)
  assert.deepEqual(resolveDefault(['gone']), DEFAULT_STAGES)
  assert.deepEqual(resolveDefault(['hand', 'terrain']), ['terrain', 'hand'])
  assert.ok(sameSteps(['ndvi', 'terrain'], ['terrain', 'ndvi']))
  assert.ok(!sameSteps(['terrain'], ['terrain', 'ndvi']))
})

test('the picker groups cover every step once', () => {
  const keys = stepGroups().flatMap((g) => g.steps.map((s) => s.key))
  assert.deepEqual([...keys].sort(), [...STAGE_KEYS].sort())
})

test('a sampled map-layer value is read under the band or the reducer name', async () => {
  const { mapLayerValue } = await import('../netlify/lib/ee-runner.mjs')
  assert.equal(mapLayerValue({ hand_m: 4.5 }, 'hand_m', 'first'), 4.5)
  assert.equal(mapLayerValue({ first: 3 }, 'hand_m', 'first'), 3)
  assert.equal(mapLayerValue({}, 'hand_m', 'first'), null)
  const classes = [{ label: 'Large diameter' }, { label: 'Medium diameter' }]
  assert.equal(mapLayerValue({ mode: 2 }, 'stand_size', 'mode', classes), 'Medium diameter')
  assert.equal(mapLayerValue({ mode: 9 }, 'stand_size', 'mode', classes), null)
})
