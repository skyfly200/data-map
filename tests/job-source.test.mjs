/* Which observations a job runs over, and what that costs.
 *
 * Selection is what a member will argue with — "why did my area miss that
 * find?" — so the boundary rules are pinned here rather than left to be
 * discovered against a live dataset.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { countDates, selectFeatures, withinBounds } from '../netlify/lib/job-source.mjs'
import { normaliseBounds } from '../netlify/lib/ee-pipeline.mjs'

const feature = (lat, lon, props = {}) => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lon, lat] },
  properties: props,
})

const box = { north: 40, south: 39, east: -105, west: -106 }

test('a point inside the box is selected and one outside is not', () => {
  assert.ok(withinBounds(feature(39.5, -105.5), box))
  assert.ok(!withinBounds(feature(41, -105.5), box))
  assert.ok(!withinBounds(feature(39.5, -104), box))
})

test('the edges of the box are inside it', () => {
  // Exclusive edges would silently drop the points a member dragged the box to
  // include, which is the complaint that is hardest to reproduce.
  for (const [lat, lon] of [[40, -105], [39, -106], [40, -106], [39, -105]]) {
    assert.ok(withinBounds(feature(lat, lon), box), `corner ${lat},${lon}`)
  }
})

test('a point with no usable geometry is not selected', () => {
  assert.ok(!withinBounds({ type: 'Feature', geometry: null }, box))
  assert.ok(!withinBounds(feature(NaN, -105.5), box))
  assert.ok(!withinBounds(null, box))
})

test('a box across the antimeridian selects both sides of it', () => {
  const wrapped = normaliseBounds({ north: 1, south: -1, east: -179, west: 179 })
  assert.ok(withinBounds(feature(0, 179.5), wrapped), 'east of the line')
  assert.ok(withinBounds(feature(0, -179.5), wrapped), 'west of the line')
  assert.ok(!withinBounds(feature(0, 100), wrapped), 'nowhere near it')
})

test('dates filter inclusively at both ends', () => {
  const feats = [
    feature(39.5, -105.5, { date: '2026-08-31' }),
    feature(39.5, -105.5, { date: '2026-09-01' }),
    feature(39.5, -105.5, { date: '2026-09-15' }),
    feature(39.5, -105.5, { date: '2026-09-30' }),
    feature(39.5, -105.5, { date: '2026-10-01' }),
  ]
  const got = selectFeatures(feats, { bounds: box, dateFrom: '2026-09-01', dateTo: '2026-09-30' })
  assert.deepEqual(got.map((f) => f.properties.date), ['2026-09-01', '2026-09-15', '2026-09-30'])
})

test('a record with no date is excluded once a date range is asked for', () => {
  const feats = [feature(39.5, -105.5, {}), feature(39.5, -105.5, { date: '2026-09-15' })]
  assert.equal(selectFeatures(feats, { bounds: box, dateFrom: '2026-09-01' }).length, 1)
  // With no range, a missing date is not a reason to drop the record.
  assert.equal(selectFeatures(feats, { bounds: box }).length, 2)
})

test('a taxon matches at whatever rank it names', () => {
  const feats = [
    feature(39.5, -105.5, { genus: 'Morchella', species: 'Morchella esculenta', order: 'Pezizales' }),
    feature(39.5, -105.5, { genus: 'Amanita', species: 'Amanita muscaria', order: 'Agaricales' }),
  ]
  assert.equal(selectFeatures(feats, { bounds: box, taxon: 'Morchella' }).length, 1)
  assert.equal(selectFeatures(feats, { bounds: box, taxon: 'Agaricales' }).length, 1)
  // Case should not decide whether a member's foray gets planned.
  assert.equal(selectFeatures(feats, { bounds: box, taxon: 'morchella' }).length, 1)
  assert.equal(selectFeatures(feats, { bounds: box, taxon: 'Boletus' }).length, 0)
})

test('a taxon matches a whole rank value, not part of one', () => {
  // "Amanita" must not pull in a genus called "Amanitopsis" via a substring.
  const feats = [feature(39.5, -105.5, { genus: 'Amanitopsis' })]
  assert.equal(selectFeatures(feats, { bounds: box, taxon: 'Amanita' }).length, 0)
})

test('distinct dates are what the dated stages are billed for', () => {
  const feats = [
    feature(0, 0, { date: '2026-09-01' }),
    feature(0, 0, { date: '2026-09-01' }),
    feature(0, 0, { date: '2026-09-02T10:00:00Z' }),
    feature(0, 0, {}),
  ]
  assert.equal(countDates(feats), 2)
  // Never zero: it multiplies a cost, and a zero there would price every dated
  // stage at nothing.
  assert.equal(countDates([]), 1)
  assert.equal(countDates([feature(0, 0, {})]), 1)
})
