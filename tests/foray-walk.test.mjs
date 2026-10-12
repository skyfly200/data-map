import test from 'node:test'
import assert from 'node:assert/strict'
import {
  ROADSIDE_M, SHORT_M, driveTarget, isAccessible, keysToFetch, passesWalk, walkMetres, walkText, walkTier,
} from '../composables/forayWalk.ts'
import { overpassPointsQuery, parseOsmPoints } from '../netlify/lib/access-ingest.mjs'
import { MAX_POINTS, handleAccessNearest, parsePoints } from '../netlify/functions/access-nearest.mjs'

const P = (m, extra = {}) => ({ m, name: 'Lot', lon: -105.3, lat: 40.1, ...extra })
const info = (parking, road_m = null, accessible = null) => ({ parking, accessible, trailhead: null, road_m })

test('walk tier: nearer of parking and road, with quarter-mile and mile cut-offs', () => {
  assert.equal(walkMetres(info(P(900), 300)), 300)
  assert.equal(walkTier(info(P(ROADSIDE_M)), true), 'roadside')
  assert.equal(walkTier(info(P(ROADSIDE_M + 1)), true), 'short')
  assert.equal(walkTier(info(null, SHORT_M + 1), true), 'hike')
  assert.equal(walkTier(info(null, null), true), 'far')
  assert.equal(walkTier(info(null, null), false), 'unknown')
  assert.equal(walkTier(null, true), 'unknown')
})

test('walk filter: max walk, accessible, unknown only when kept', () => {
  const f = (o) => ({ maxWalk: 'any', accessible: false, keepUnknown: false, ...o })
  assert.ok(passesWalk('unknown', false, f({})))
  assert.ok(passesWalk('short', false, f({ maxWalk: 'short' })))
  assert.ok(!passesWalk('hike', false, f({ maxWalk: 'short' })))
  assert.ok(!passesWalk('far', false, f({ maxWalk: 'hike' })))
  assert.ok(!passesWalk('unknown', false, f({ maxWalk: 'roadside' })))
  assert.ok(passesWalk('unknown', false, f({ maxWalk: 'roadside', keepUnknown: true })))
  assert.ok(!passesWalk('roadside', false, f({ accessible: true })))
  assert.ok(passesWalk('hike', true, f({ accessible: true })))
})

test('accessible parking, walk text and drive target', () => {
  assert.ok(isAccessible(info(P(100), null, P(200))))
  assert.ok(!isAccessible(info(P(100), null, P(2000))))
  assert.equal(walkText(info(P(965)), 'short'), '0.6 mi from parking')
  assert.equal(walkText(info(P(5000), 40), 'roadside'), '<0.1 mi from a road')
  assert.match(walkText(info(null), 'far'), /No parking or road within 3 mi/)
  assert.equal(walkText(null, 'unknown'), null)
  assert.deepEqual(driveTarget({ lat: 1, lon: 2 }, info(P(10))), { lat: 40.1, lon: -105.3, parking: true })
  assert.deepEqual(driveTarget({ lat: 1, lon: 2 }, null), { lat: 1, lon: 2, parking: false })
})

test('keysToFetch: skips cached, keeps order, caps', () => {
  const have = new Map([['b', 1]])
  assert.deepEqual(keysToFetch([{ key: 'a' }, { key: 'b' }, { key: 'c' }, { key: 'd' }], have, 2), ['a', 'c'])
})

test('parseOsmPoints: parking nodes and way centres, trailheads, skips private', () => {
  const rows = parseOsmPoints({ elements: [
    { type: 'node', id: 1, lat: 40.1, lon: -105.2, tags: { amenity: 'parking', fee: 'yes', wheelchair: 'yes', 'capacity:disabled': '2' } },
    { type: 'way', id: 2, center: { lat: 40.2, lon: -105.3 }, tags: { amenity: 'parking', name: 'TH lot' } },
    { type: 'node', id: 3, lat: 40.3, lon: -105.4, tags: { highway: 'trailhead', name: 'Mesa' } },
    { type: 'node', id: 4, lat: 40.3, lon: -105.4, tags: { amenity: 'parking', access: 'private' } },
    { type: 'way', id: 5, tags: { amenity: 'parking' } },
    { type: 'node', id: 6, lat: 40, lon: -105, tags: { amenity: 'bench' } },
  ] })
  assert.deepEqual(rows.map((r) => [r.osm_type, r.osm_id, r.kind]), [['node', 1, 'parking'], ['way', 2, 'parking'], ['node', 3, 'trailhead']])
  assert.equal(rows[0].fee, 'yes'); assert.equal(rows[0].wheelchair, 'yes'); assert.equal(rows[0].disabled_spaces, 2)
  assert.equal(rows[1].lat, 40.2); assert.equal(rows[1].disabled_spaces, null)
  const q = overpassPointsQuery([-105.5, 40, -105, 40.5])
  assert.match(q, /nwr\["amenity"="parking"\]\(40,-105\.5,40\.5,-105\)/)
  assert.match(q, /out center tags/)
})

test('access-nearest: validates, calls the RPC in order, fills gaps with nulls', async () => {
  assert.throws(() => parsePoints([]))
  assert.throws(() => parsePoints([[0, 95]]))
  assert.throws(() => parsePoints(Array.from({ length: MAX_POINTS + 1 }, () => [0, 0])))
  assert.deepEqual(parsePoints([[-105.1234567, 40.7654321]]), [[-105.12346, 40.76543]])

  const calls = []
  const client = { rpc: async (fn, args) => { calls.push([fn, args]); return { data: [{ i: 1, parking: { m: 120 }, road_m: 30 }], error: null } } }
  const post = (body, c = client) => handleAccessNearest(new Request('https://x/', { method: 'POST', body: JSON.stringify(body) }), c)
  const res = await post({ points: [[-105, 40], [-105.1, 40.1]] })
  assert.equal(res.status, 200)
  const d = await res.json()
  assert.equal(calls[0][0], 'access_nearest')
  assert.deepEqual(calls[0][1].p_points, [[-105, 40], [-105.1, 40.1]])
  assert.equal(d.results.length, 2)
  assert.deepEqual(d.results[0], { i: 0, parking: null, accessible: null, trailhead: null, road_m: null })
  assert.equal(d.results[1].parking.m, 120); assert.equal(d.results[1].road_m, 30)
  assert.equal((await post({ points: 'x' })).status, 400)
  assert.equal((await post({ points: [[0, 0]] }, null)).status, 503)
  assert.equal((await handleAccessNearest(new Request('https://x/'), client)).status, 405)
  const bad = { rpc: async () => ({ data: null, error: { message: 'boom' } }) }
  assert.equal((await post({ points: [[0, 0]] }, bad)).status, 500)
})
