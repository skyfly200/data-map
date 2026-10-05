/* Allowed-area validation and the access-sets composable (fake fetch). */
import test from 'node:test'
import assert from 'node:assert/strict'

import {
  AreaError, assertPolygonGeometry, parseAreaGeoJSON, ringToPolygon, validateAreaInput, canManage, scopeLabel,
} from '../composables/areaGeometry.ts'
import { parseSetDetail, useAccessSets } from '../composables/useAccessSets.ts'

const SQ = { type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1], [0, 0]]] }
const PT = { type: 'Point', coordinates: [0, 0] }
const fc = (...geoms) => ({ type: 'FeatureCollection', features: geoms.map((g, i) => ({ type: 'Feature', geometry: g, properties: i ? { name: `n${i}` } : {} })) })

test('polygon geometry: accepts closed polygons and multipolygons only', () => {
  assertPolygonGeometry(SQ)
  assertPolygonGeometry({ type: 'MultiPolygon', coordinates: [SQ.coordinates] })
  assert.throws(() => assertPolygonGeometry(PT), AreaError)
  assert.throws(() => assertPolygonGeometry({ type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1], [2, 2]]] }), AreaError)
  assert.throws(() => assertPolygonGeometry({ type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 99], [0, 0]]] }), AreaError)
})

test('ringToPolygon swaps lat/lng and closes the ring', () => {
  const g = ringToPolygon([[10, 20], [10, 21], [11, 21]])
  assert.deepEqual(g.coordinates[0][0], [20, 10])
  assert.deepEqual(g.coordinates[0].at(-1), [20, 10])
  assert.throws(() => ringToPolygon([[1, 1], [2, 2]]), AreaError)
})

test('parseAreaGeoJSON: names, skipping non-polygons, errors', () => {
  const r = parseAreaGeoJSON(JSON.stringify(fc(SQ, SQ, PT)))
  assert.equal(r.areas.length, 2)
  assert.equal(r.skipped, 1)
  assert.deepEqual(r.areas.map((a) => a.name), ['Area 1', 'n1'])
  assert.throws(() => parseAreaGeoJSON('nope'), /not valid JSON/)
  assert.throws(() => parseAreaGeoJSON(fc(PT)), /No valid polygons/)
  assert.throws(() => parseAreaGeoJSON('x'.repeat(20), 10), /larger/)
  assert.equal(parseAreaGeoJSON(SQ).areas.length, 1)
})

test('validateAreaInput', () => {
  const ok = validateAreaInput({ name: ' A ', geometry: SQ })
  assert.equal(ok.name, 'A'); assert.equal(ok.fee_status, 'unknown'); assert.equal(ok.collecting, 'unknown')
  assert.throws(() => validateAreaInput({ name: '', geometry: SQ }), /name/)
  assert.throws(() => validateAreaInput({ name: 'a', geometry: SQ, collecting: 'yes' }), /ollecting/)
  assert.throws(() => validateAreaInput({ name: 'a', geometry: null }), AreaError)
  assert.equal(scopeLabel('club'), 'Club area'); assert.equal(scopeLabel('user'), 'Your area')
  assert.ok(canManage('admin') && !canManage('member'))
})

test('parseSetDetail tolerates missing fields', () => {
  const d = parseSetDetail({ set: { id: 's' }, areas: { features: [{ id: 'a1', geometry: SQ, properties: { name: 'X' } }, { geometry: SQ }] } })
  assert.equal(d.areas[0].properties.id, 'a1')
  assert.equal(d.areas[1].properties.name, 'Unnamed area')
  assert.deepEqual(parseSetDetail(null).areas, [])
})

// ── composable with fake fetch ───────────────────────────────────────────────
function setup(handler) {
  const store = new Map()
  globalThis.useState = (k, init) => { if (!store.has(k)) store.set(k, { value: init() }); return store.get(k) }
  globalThis.useAuth = () => ({ accessToken: async () => 'tok' })
  const calls = []
  globalThis.fetch = async (url, opts = {}) => {
    calls.push({ url, method: opts.method, headers: opts.headers, body: opts.body && JSON.parse(opts.body) })
    const { status = 200, body } = handler(url, opts)
    return { ok: status < 400, status, json: async () => body }
  }
  return { api: useAccessSets(), calls }
}

test('refresh sends bearer auth and stores sets', async () => {
  const { api, calls } = setup(() => ({ body: { ok: true, sets: [{ id: '1', name: 'S', scope: 'user', area_count: 0, role: 'owner' }] } }))
  await api.refresh()
  assert.equal(calls[0].headers.authorization, 'Bearer tok')
  assert.equal(calls[0].method, 'GET')
  assert.equal(api.sets.value.length, 1)
})

test('404 marks the endpoint unavailable without an error banner', async () => {
  const { api } = setup(() => ({ status: 404, body: null }))
  await api.refresh()
  assert.equal(api.unavailable.value, true)
  assert.equal(api.error.value, '')
  assert.deepEqual(api.sets.value, [])
})

test('addArea validates first, posts, then reloads set and detail', async () => {
  const { api, calls } = setup((url, opts) => {
    if (opts.method === 'POST') return { body: { ok: true } }
    if (url.includes('set_id=s1')) return { body: { ok: true, set: { id: 's1', name: 'S' }, areas: fc(SQ) } }
    return { body: { ok: true, sets: [] } }
  })
  await assert.rejects(api.addArea('s1', { name: '', geometry: SQ }), /name/)
  assert.equal(calls.length, 0)
  await api.addArea('s1', { name: 'Spot', geometry: SQ, collecting: 'allowed' })
  assert.equal(calls[0].body.action, 'add_area')
  assert.equal(calls[0].body.set_id, 's1')
  assert.equal(calls[0].body.collecting, 'allowed')
  assert.equal(api.areas.value.length, 1)
})

test('createSet requires a club for club scope; server errors surface', async () => {
  const { api, calls } = setup(() => ({ status: 403, body: { ok: false, error: 'Not allowed.' } }))
  await assert.rejects(api.createSet('x', 'club'), /Choose a club/)
  await assert.rejects(api.createClub('c'), /Not allowed/)
  assert.equal(api.error.value, 'Not allowed.')
  assert.equal(calls.length, 1)
})

test('importGeojson rejects bad files before any request and sends polygons only', async () => {
  const { api, calls } = setup(() => ({ body: { ok: true, sets: [] } }))
  await assert.rejects(async () => api.importGeojson('s', JSON.stringify(PT)), AreaError)
  assert.equal(calls.length, 0)
  const r = await api.importGeojson('s', JSON.stringify(fc(SQ, PT)))
  assert.equal(calls[0].body.action, 'import_geojson')
  assert.equal(calls[0].body.geojson.features.length, 1)
  assert.equal(r.skipped, 1)
})

test('club actions use the contract field names', async () => {
  const { api, calls } = setup(() => ({ body: { ok: true, sets: [] } }))
  await api.addMember('c1', ' a@b.org ')
  await api.removeMember('c1', 'u1')
  assert.deepEqual(calls.filter((c) => c.method === 'POST').map((c) => c.body), [
    { action: 'add_member', club_id: 'c1', email: 'a@b.org' },
    { action: 'remove_member', club_id: 'c1', user_id: 'u1' },
  ])
})
