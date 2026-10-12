import test from 'node:test'
import assert from 'node:assert/strict'
import {
  distanceMi, filterByDistance, formatDistance, parseLatLon, parseSavedOrigin, radiusLabel, sortByDistance, RADIUS_OPTIONS_MI,
} from '../composables/forayDistance.ts'

const boulder = { lat: 40.015, lon: -105.27 }
const denver = { lat: 39.739, lon: -104.99 }

test('distanceMi: Boulder to Denver is about 24 miles', () => {
  const d = distanceMi(boulder, denver)
  assert.ok(d > 22 && d < 26, String(d))
  assert.equal(distanceMi(boulder, boulder), 0)
})

test('filterByDistance: radius cut and distMi attached', () => {
  const cells = [{ key: 'a', ...boulder }, { key: 'b', ...denver }, { key: 'c', lat: 37.27, lon: -107.88 }]
  const r10 = filterByDistance(cells, boulder, 10)
  assert.deepEqual(r10.map((c) => c.key), ['a'])
  assert.equal(r10[0].distMi, 0)
  assert.deepEqual(filterByDistance(cells, boulder, 25).map((c) => c.key), ['a', 'b'])
  const any = filterByDistance(cells, boulder, 0)
  assert.equal(any.length, 3)
  assert.ok(any[2].distMi > 200)
})

test('filterByDistance: no origin passes everything with null distance', () => {
  const out = filterByDistance([{ key: 'a', ...denver }], null, 10)
  assert.equal(out.length, 1)
  assert.equal(out[0].distMi, null)
})

test('sortByDistance: nearest first, stable on ties, nulls last', () => {
  const out = sortByDistance([
    { key: 'x', distMi: 5 }, { key: 'y', distMi: null }, { key: 'z', distMi: 1 }, { key: 'w', distMi: 5 },
  ])
  assert.deepEqual(out.map((c) => c.key), ['z', 'x', 'w', 'y'])
})

test('parseLatLon: accepts comma or space, rejects out of range and text', () => {
  assert.deepEqual(parseLatLon('40.01, -105.27'), { lat: 40.01, lon: -105.27 })
  assert.deepEqual(parseLatLon(' 40 -105 '), { lat: 40, lon: -105 })
  assert.equal(parseLatLon('Boulder'), null)
  assert.equal(parseLatLon('95, 10'), null)
  assert.equal(parseLatLon(''), null)
})

test('formatDistance and radiusLabel', () => {
  assert.equal(formatDistance(0.4), '<1 mi')
  assert.equal(formatDistance(3.26), '3.3 mi')
  assert.equal(formatDistance(42.6), '43 mi')
  assert.equal(formatDistance(null), '')
  assert.equal(radiusLabel(25), '25 mi')
  assert.equal(radiusLabel(0), 'Any distance')
  assert.ok(RADIUS_OPTIONS_MI.includes(0))
})

test('parseSavedOrigin: valid round-trip, junk rejected', () => {
  const o = parseSavedOrigin(JSON.stringify({ ...boulder, label: 'Boulder', source: 'typed' }))
  assert.deepEqual(o, { ...boulder, label: 'Boulder', source: 'typed' })
  assert.equal(parseSavedOrigin(JSON.stringify({ lat: 'x', lon: 1 })), null)
  assert.equal(parseSavedOrigin('{bad'), null)
  assert.equal(parseSavedOrigin(null), null)
  assert.equal(parseSavedOrigin(JSON.stringify({ ...boulder, source: 'evil' })).source, 'typed')
})
