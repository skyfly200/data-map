/* Which finds a layer built from observations is handed.
 *
 * The habitat-similarity layer averages the satellite embedding over these
 * sites, so what is pinned is that the right records are picked, duplicates of
 * one spot count once, and a large taxon is thinned to a stable, capped set.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { pickReferencePoints } from '../netlify/lib/reference-points.mjs'

const find = (species, lon, lat, extra = {}) => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lon, lat] },
  properties: { species, ...extra },
})

test('only records of the taxon are picked, by species or implied genus', () => {
  const features = [
    find('Morchella americana', -105.1, 39.1),
    find('Morchella sextelata', -105.2, 39.2),
    find('Cantharellus roseocanus', -105.3, 39.3),
  ]
  assert.equal(pickReferencePoints(features, 'Morchella').length, 2)
  assert.deepEqual(pickReferencePoints(features, 'Cantharellus roseocanus'), [[-105.3, 39.3]])
  assert.deepEqual(pickReferencePoints(features, 'Amanita'), [])
})

test('no taxon picks nothing rather than everything', () => {
  assert.deepEqual(pickReferencePoints([find('Morchella americana', -105, 39)], ''), [])
})

test('repeat records of one spot count once', () => {
  const features = [
    find('Morchella americana', -105.123401, 39.5),
    find('Morchella americana', -105.123402, 39.5),
    find('Morchella americana', -105.2, 39.5),
  ]
  assert.equal(pickReferencePoints(features, 'Morchella').length, 2)
})

test('records without usable coordinates are skipped', () => {
  const features = [
    { properties: { species: 'Morchella americana' } },
    find('Morchella americana', NaN, 39),
    find('Morchella americana', -500, 39),
    { properties: { species: 'Morchella americana', lon: -105.5, lat: 39.5 } },
  ]
  assert.deepEqual(pickReferencePoints(features, 'Morchella'), [[-105.5, 39.5]])
})

test('a large taxon is thinned to the cap, the same way every time', () => {
  const features = Array.from({ length: 2000 }, (_, i) =>
    find('Morchella americana', -106 + i / 1000, 39 + (i % 7) / 100))
  const a = pickReferencePoints(features, 'Morchella', 500)
  const b = pickReferencePoints([...features].reverse(), 'Morchella', 500)
  assert.equal(a.length, 500)
  assert.deepEqual(a, b)
  // Spread across the whole range rather than the first 500.
  assert.ok(a[a.length - 1][0] > -104.2)
})
