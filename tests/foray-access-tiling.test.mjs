import test from 'node:test'
import assert from 'node:assert/strict'
import { COLORADO_BBOX, parseAccessResponse } from '../composables/forayPlanner.ts'
import { fetchAccess, tileBbox, clearAccessCache, MAX_SPLIT_DEPTH } from '../composables/useAccess.ts'

const feat = (id, x, y) => ({
  type: 'Feature', properties: { id, name: id, public_access: 'open', fee_status: 'free', collecting: 'allowed' },
  geometry: { type: 'Polygon', coordinates: [[[x, y], [x + 0.1, y], [x + 0.1, y + 0.1], [x, y + 0.1], [x, y]]] },
})
const body = (features, o = {}) => ({ ok: true, areas: { type: 'FeatureCollection', features }, truncated: { areas: false, lines: false }, loaded: true, ...o })
const bboxOf = (u) => u.split('bbox=')[1].split(',').map(Number)
const mk = (handler) => {
  const urls = []
  const f = async (u) => {
    urls.push(u)
    const r = handler(bboxOf(u), u)
    if (r instanceof Error) throw r
    return { ok: r.status !== 500, json: async () => r.body }
  }
  return { f, urls }
}
const inside = (b, x, y) => x >= b[0] && x < b[2] && y >= b[1] && y < b[3]

test('contract shape: FeatureCollection areas + top-level loaded', () => {
  const p = parseAccessResponse(body([feat('a', -105, 40)], { truncated: { areas: true } }))
  assert.equal(p.status, 'loaded'); assert.equal(p.areas.length, 1); assert.equal(p.truncated, true)
  assert.equal(parseAccessResponse(body([feat('a', -105, 40)], { loaded: false })).status, 'not-loaded')
  assert.equal(parseAccessResponse({ ok: false, error: 'x' }).status, 'unavailable')
})

test('tiles never exceed 3 deg and cover the bbox', () => {
  const t = tileBbox(COLORADO_BBOX)
  assert.equal(t.length, 6)
  for (const b of t) assert.ok(b[2] - b[0] <= 3 && b[3] - b[1] <= 3)
  assert.equal(Math.min(...t.map((b) => b[0])), COLORADO_BBOX[0])
  assert.equal(Math.max(...t.map((b) => b[2])), COLORADO_BBOX[2])
})

test('statewide: tiled requests, de-duped by id, loaded, cached', async () => {
  clearAccessCache()
  const { f, urls } = mk((b) => ({ body: body([feat('shared', -105.5, 39.0), feat(`t${b[0]}_${b[1]}`, b[0] + 0.1, b[1] + 0.1)]) }))
  const r = await fetchAccess(COLORADO_BBOX, f)
  assert.equal(urls.length, 6)
  assert.equal(r.status, 'loaded')
  assert.equal(r.areas.length, 7)
  await fetchAccess(COLORADO_BBOX, f)
  assert.equal(urls.length, 6, 'cached per tile')
})

test('truncated tile is subdivided; still truncated at max depth -> partial', async () => {
  clearAccessCache()
  const small = [-105.3, 40.0, -104.3, 41.0]
  let calls = 0
  const a = mk((b) => { calls++; return { body: body([feat(`s${calls}`, b[0], b[1])], { truncated: { areas: b[2] - b[0] > 0.6 } }) } })
  const r = await fetchAccess(small, a.f)
  assert.equal(r.status, 'loaded'); assert.equal(calls, 5)
  clearAccessCache()
  const b2 = mk((b) => ({ body: body([feat(`x${b.join()}`, b[0], b[1])], { truncated: { areas: true } }) }))
  const r2 = await fetchAccess(small, b2.f)
  assert.equal(r2.status, 'partial')
  assert.equal(b2.urls.length, (4 ** (MAX_SPLIT_DEPTH + 1) - 1) / 3)
})

test('one tile failing -> partial; all failing -> unavailable; failures not cached', async () => {
  clearAccessCache()
  let fail = true
  const { f } = mk((b) => (fail && inside(b, -108, 38) ? { status: 500 } : { body: body([feat(`t${b[0]}_${b[1]}`, b[0] + 0.1, b[1] + 0.1)]) }))
  const r = await fetchAccess(COLORADO_BBOX, f)
  assert.equal(r.status, 'partial'); assert.equal(r.areas.length, 5)
  fail = false
  assert.equal((await fetchAccess(COLORADO_BBOX, f)).status, 'loaded')
  clearAccessCache()
  assert.equal((await fetchAccess(COLORADO_BBOX, mk(() => new Error('offline')).f)).status, 'unavailable')
})

test('concurrency is limited', async () => {
  clearAccessCache()
  let cur = 0, max = 0
  const f = async (u) => { cur++; max = Math.max(max, cur); await new Promise((r) => setTimeout(r, 5)); cur--; return { ok: true, json: async () => body([feat(u, -105, 40)]) } }
  await fetchAccess(COLORADO_BBOX, f)
  assert.ok(max <= 4 && max > 1)
})
