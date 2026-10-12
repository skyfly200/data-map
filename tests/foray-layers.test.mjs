/* Foray planner model layers (WANT-17 phases 2, 3 and 5): weighting, predictor
 * choice, the training envelope, the EE builders on a stub, and the endpoint
 * against an in-memory Supabase fake. */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  MAX_LAYER_MODELS, MAX_SAMPLE_POINTS, buildEnsembleImage, buildHabitatImage, habitatPredictors,
  normaliseWeights, parseModelWeights, parseSamplePoints, shapeSamples,
} from '../netlify/lib/foray-layers.mjs'
import { handleForayLayers } from '../netlify/functions/foray-layers.mjs'

const A = '11111111-1111-1111-1111-111111111111'
const B = '22222222-2222-2222-2222-222222222222'
const C = '33333333-3333-3333-3333-333333333333'

test('parseModelWeights reads id:weight pairs, defaults to 1 and rejects bad input', () => {
  assert.deepEqual(parseModelWeights(`${A}:0.5, ${B}`), [{ id: A, weight: 0.5 }, { id: B, weight: 1 }])
  assert.deepEqual(parseModelWeights(`${A}:2,${A}:3`), [{ id: A, weight: 2 }])
  assert.throws(() => parseModelWeights(''), /at least one/)
  assert.throws(() => parseModelWeights(`${A}:-1`), /positive/)
  assert.throws(() => parseModelWeights('drop table'), /not a model id/)
  const many = Array.from({ length: MAX_LAYER_MODELS + 1 }, (_, i) => `${String(i).padStart(8, '0')}-0000-0000-0000-000000000000`)
  assert.throws(() => parseModelWeights(many.join(',')), /At most/)
})

test('normaliseWeights scales to sum 1', () => {
  const w = normaliseWeights([{ id: 'a', weight: 3 }, { id: 'b', weight: 1 }])
  assert.deepEqual(w.map((m) => m.weight), [0.75, 0.25])
})

const ranges = (o) => Object.fromEntries(Object.entries(o).map(([k, [p25, p75, min, max]]) => [k, { p25, p75, min, max }]))

test('habitatPredictors normalises each model over its own predictors, then averages by weight', () => {
  const { predictors, skipped } = habitatPredictors([
    // Small model: elevation is all of it.
    { id: 'a', weight: 1, contributions: { elevation: 40 }, ranges: ranges({ elevation: [2400, 2900, 2000, 3300] }) },
    // Big model: elevation is a quarter, ndvi three quarters.
    { id: 'b', weight: 1, contributions: { elevation: 10, ndvi: 30 }, ranges: ranges({ elevation: [2600, 3100, 2200, 3500], ndvi: [0.4, 0.7, 0.1, 0.9] }) },
  ])
  assert.deepEqual(skipped, [])
  const byKey = Object.fromEntries(predictors.map((p) => [p.key, p]))
  // elevation: (1 + 0.25) / 2 = 0.625; ndvi: 0.75 / 2 = 0.375 -> already sums to 1.
  assert.ok(Math.abs(byKey.elevation.weight - 0.625) < 1e-9)
  assert.ok(Math.abs(byKey.ndvi.weight - 0.375) < 1e-9)
  // Range is the union of IQRs; envelope the union of min..max.
  assert.deepEqual([byKey.elevation.p25, byKey.elevation.p75, byKey.elevation.min, byKey.elevation.max], [2400, 3100, 2000, 3500])
})

test('habitatPredictors skips models without ranges and ignores unknown predictors', () => {
  const { predictors, skipped } = habitatPredictors([
    { id: 'a', weight: 1, contributions: { elevation: 50 }, ranges: null },
    { id: 'b', weight: 1, contributions: { elevation: 50, bogus: 50 }, ranges: ranges({ elevation: [1, 2, 0, 3], bogus: [1, 2, 0, 3] }) },
  ])
  assert.deepEqual(skipped, ['a'])
  assert.deepEqual(predictors.map((p) => [p.key, p.weight]), [['elevation', 1]])
})

test('habitatPredictors keeps only the top predictors, renormalised', () => {
  const contributions = { elevation: 30, slope: 25, ndvi: 20, aspect: 15, twi: 10 }
  const r = ranges(Object.fromEntries(Object.keys(contributions).map((k) => [k, [1, 2, 0, 3]])))
  const { predictors } = habitatPredictors([{ id: 'a', weight: 1, contributions, ranges: r }], { top: 2 })
  assert.deepEqual(predictors.map((p) => p.key), ['elevation', 'slope'])
  assert.ok(Math.abs(predictors[0].weight + predictors[1].weight - 1) < 1e-9)
})

function stubEe() {
  const calls = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => { if (prop === 'then') return undefined; calls.push(String(prop)); return chain },
    apply: () => chain,
  })
  const rec = (name) => () => { calls.push(name); return chain }
  const ee = new Proxy({}, {
    get: (t, prop) => {
      if (prop === 'Image') { const img = rec('Image'); img.cat = rec('cat'); img.constant = rec('constant'); return img }
      if (prop === 'ImageCollection') return rec('ImageCollection')
      if (prop === 'Terrain') return { slope: rec('slope'), aspect: rec('aspect') }
      return chain
    },
  })
  return { ee, calls }
}

test('buildHabitatImage scores in-range predictors and adds the envelope band', () => {
  const { ee, calls } = stubEe()
  buildHabitatImage(ee, [{ key: 'elevation', weight: 1, p25: 1, p75: 2, min: 0, max: 3 }])
  for (const m of ['gte', 'lte', 'sum', 'lt', 'gt', 'max', 'addBands']) assert.ok(calls.includes(m), m)
})

test('buildEnsembleImage divides by the weight of the surfaces that cover a pixel', () => {
  const { ee, calls } = stubEe()
  buildEnsembleImage(ee, [{ image: ee.Image(), weight: 0.5 }, { image: ee.Image(), weight: 0.5 }], [])
  for (const m of ['unmask', 'mask', 'divide', 'updateMask', 'constant']) assert.ok(calls.includes(m), m)
})

test('parseSamplePoints validates and shapeSamples keeps input order', () => {
  assert.deepEqual(parseSamplePoints([[-105, 40], ['-104.5', '39.5']]), [[-105, 40], [-104.5, 39.5]])
  assert.throws(() => parseSamplePoints([]), /at least one/)
  assert.throws(() => parseSamplePoints([[200, 0]]), /lon\/lat/)
  assert.throws(() => parseSamplePoints(Array(MAX_SAMPLE_POINTS + 1).fill([0, 0])), /At most/)
  const fc = { features: [{ properties: { i: 1, habitat: 0.8, outside: 0 } }, { properties: { i: 0, outside: 1 } }] }
  assert.deepEqual(shapeSamples(fc, 3, 'habitat'), [
    { i: 0, value: null, outside: true }, { i: 1, value: 0.8, outside: false }, { i: 2, value: null, outside: null },
  ])
})

// ── Endpoint ─────────────────────────────────────────────────────────────────

function fakeClient({ configs, jobs }) {
  const query = (table) => {
    const filters = []
    let orFilter = null
    const rows = () => {
      let r = table === 'model_configs' ? configs : table === 'ee_jobs' ? jobs : []
      for (const [k, v] of filters) r = r.filter((x) => v.includes(x[k]))
      if (orFilter) r = r.filter((x) => x.owner_id === orFilter || x.visibility === 'public')
      return r
    }
    const b = {
      select: () => b,
      in: (k, v) => (filters.push([k, v.map(String)]), b),
      or: (expr) => { orFilter = /owner_id\.eq\.([^,]+)/.exec(expr)[1]; return b },
      then: (res, rej) => Promise.resolve({ data: rows(), error: null }).then(res, rej),
    }
    return b
  }
  return { from: query }
}

const ME = 'user-me'
const world = () => fakeClient({
  configs: [
    { id: A, owner_id: ME, title: 'Chanterelle', visibility: 'private', model_runs: [{ job_id: 'j1', created_at: '2026-10-01' }], model_results: [] },
    { id: B, owner_id: 'someone', title: 'Porcini', visibility: 'public', model_runs: [{ job_id: 'j2', created_at: '2026-10-01' }], model_results: [] },
    { id: C, owner_id: 'someone', title: 'Secret', visibility: 'private', model_runs: [{ job_id: 'j3', created_at: '2026-10-01' }], model_results: [] },
  ],
  jobs: [
    { id: 'j1', user_id: ME, kind: 'model', status: 'succeeded', params: { source: { type: 'bbox', taxon: 'Cantharellus' } },
      result_meta: { cv: { auc: 0.81 }, contributions: { elevation: 60, ndvi: 40 }, predictorRanges: ranges({ elevation: [1, 2, 0, 3], ndvi: [0.3, 0.6, 0, 1] }) } },
    { id: 'j2', user_id: 'someone', kind: 'model', status: 'succeeded', params: { source: { type: 'bbox', taxon: 'Boletus' } }, result_meta: {} },
    { id: 'j3', user_id: 'someone', kind: 'model', status: 'succeeded', params: {}, result_meta: {} },
  ],
})

const deps = (client, calls = []) => ({
  client,
  auth: { ok: true, user: { id: ME }, admin: false },
  eeReady: () => true,
  loadFeatures: async () => [{ geometry: { coordinates: [-105, 40] } }],
  runner: {
    mint: async (args) => { calls.push(['mint', args]); return { template: 'https://t/{z}/{x}/{y}', meta: { layer: args.layer } } },
    sample: async (args) => { calls.push(['sample', args]); return { samples: args.points.map((_, i) => ({ i, value: 0.5, outside: false })), used: [], skipped: [] } },
  },
})

const get = (q) => new Request(`http://x/f${q}`)

test('GET lists own and public models only, with what each can feed', async () => {
  const res = await handleForayLayers(get(''), deps(world()))
  const body = await res.json()
  assert.equal(res.status, 200)
  assert.deepEqual(body.models.map((m) => m.title).sort(), ['Chanterelle', 'Porcini'])
  const ch = body.models.find((m) => m.id === A)
  assert.deepEqual({ taxon: ch.taxon, auc: ch.auc, ranges: ch.ranges, own: ch.own, usable: ch.usable }, { taxon: 'Cantharellus', auc: 0.81, ranges: true, own: true, usable: true })
  assert.equal(body.models.find((m) => m.id === B).ranges, false)
  assert.equal(JSON.stringify(body).includes('result_meta'), false, 'no job internals leak')
})

test('GET ?layer mints with the loaded models; a hidden model is 404', async () => {
  const calls = []
  const ok = await handleForayLayers(get(`?layer=habitat&models=${A}:2`), deps(world(), calls))
  assert.equal(ok.status, 200)
  const [, args] = calls[0]
  assert.equal(args.layer, 'habitat')
  assert.deepEqual(args.models[0].contributions, { elevation: 60, ndvi: 40 })
  assert.equal(args.models[0].features, undefined, 'habitat needs no presences')
  const hidden = await handleForayLayers(get(`?layer=habitat&models=${C}`), deps(world()))
  assert.equal(hidden.status, 404)
  const bad = await handleForayLayers(get(`?layer=nope&models=${A}`), deps(world()))
  assert.equal(bad.status, 400)
})

test('ensemble loads presences; POST samples points', async () => {
  const calls = []
  await handleForayLayers(get(`?layer=ensemble&models=${A},${B}`), deps(world(), calls))
  assert.equal(calls[0][1].models.length, 2)
  assert.ok(calls[0][1].models.every((m) => m.features?.length === 1 && m.spec))
  const res = await handleForayLayers(new Request('http://x/f', { method: 'POST', body: JSON.stringify({ layer: 'habitat', models: `${A}`, points: [[-105, 40], [-105.1, 40.1]] }) }), deps(world(), calls))
  const body = await res.json()
  assert.equal(body.samples.length, 2)
})

test('503 when Earth Engine is not configured', async () => {
  const d = deps(world()); d.eeReady = () => false
  const res = await handleForayLayers(get(`?layer=habitat&models=${A}`), d)
  assert.equal(res.status, 503)
})
