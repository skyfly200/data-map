import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { accessFor, distanceToLinesM, parseOsmWays, parsePadUs, padusAccessClass } from '../netlify/lib/access-ingest.mjs'
import { habitatScore, mergeContributions, normaliseContributions, outsideEnvelope, preferredRange, rankOpportunities } from '../netlify/lib/foray.mjs'

const fx = JSON.parse(readFileSync(new URL('./fixtures/access-fixtures.json', import.meta.url), 'utf8'))

test('PAD-US parse: polygons only, access class mapped', () => {
  const rows = parsePadUs(fx.padus)
  assert.equal(rows.length, 2)
  assert.equal(rows[0].access_class, 'open')
  assert.equal(rows[0].geom.type, 'MultiPolygon')
  assert.equal(padusAccessClass(undefined), 'unknown')
})

test('OSM parse: kinds, drops private/unknown/non-way', () => {
  const rows = parseOsmWays(fx.osm)
  assert.deepEqual(rows.map((r) => [r.osm_id, r.kind]), [[10, 'trail'], [11, 'road']])
})

test('access summary for a cell', () => {
  const areas = parsePadUs(fx.padus), lines = parseOsmWays(fx.osm)
  const a = accessFor([-105.3, 40.25], areas, lines)
  assert.equal(a.legal, 'open'); assert.equal(a.publicLand, true); assert.ok(a.trailM < 1)
  assert.equal(accessFor([-105.8, 40.1], areas, lines).publicLand, false)
  assert.ok(Math.abs(distanceToLinesM([-105.29, 40.25], lines.slice(0, 1)) - 850) < 50)
})

test('contributions normalise per model and merge', () => {
  assert.deepEqual(normaliseContributions({ a: 30, b: 10 }), { a: 0.75, b: 0.25 })
  const m = mergeContributions([{ contributions: { a: 50, b: 50 } }, { contributions: { a: 100 } }, { contributions: null }])
  assert.deepEqual(m, { a: 0.75, b: 0.25 })
})

test('habitat score and envelope', () => {
  const r = { a: preferredRange([1, 2, 3, 4, 5, 6, 7, 8]), b: preferredRange([10, 11, 12, 13]) }
  assert.equal(preferredRange([1, 2]), null)
  assert.equal(habitatScore({ a: 4, b: 99 }, { a: 0.75, b: 0.25 }, r), 0.75)
  assert.equal(habitatScore({}, { a: 1 }, r), null)
  assert.equal(outsideEnvelope({ a: 100 }, r), true)
  assert.equal(outsideEnvelope({ a: 4, b: 11 }, r), false)
})

test('opportunity ranking masks by access and prefers low effort', () => {
  const pub = { publicLand: true, trailM: 500 }
  const out = rankOpportunities([
    { id: 'x', promise: 0.9, finds: 20, access: pub },
    { id: 'y', promise: 0.8, finds: 0, access: pub },
    { id: 'z', promise: 1, finds: 0, access: { publicLand: false, trailM: 10 } },
    { id: 'w', promise: 1, finds: 0, access: { publicLand: true, trailM: 9000 } },
    { id: 'v', promise: 0.7, finds: 0, access: pub, envelope: true },
  ], { auc: 0.8 })
  assert.deepEqual(out.map((c) => c.id), ['y', 'v', 'x'])
  assert.equal(out[1].extrapolated, true); assert.equal(out[0].auc, 0.8)
})
