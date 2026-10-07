/* The strings the map writes for picked spots must be ones the server keeps
 * as-is, or each tap would mint under a different cache key than it renders. */

import test from 'node:test'
import assert from 'node:assert/strict'

import { spotFor, spotList } from '../composables/useMapSpotPicker.js'
import { resolveLayer } from '../netlify/lib/ee-tile-layers.mjs'

test('a tap is rounded the way the server normalises it', () => {
  const spots = [spotFor(39.123456789, -105.987654321), spotFor(40, -106)].join(';')
  assert.equal(spots, '39.12346,-105.98765;40.00000,-106.00000')
  const { params } = resolveLayer('habitat-similarity', { compare: 'spots', points: spots })
  assert.deepEqual(spotList(params.points).map((s) => s.split(',').map(Number)),
    spotList(spots).map((s) => s.split(',').map(Number)))
})

test('an empty or missing list has no spots', () => {
  assert.deepEqual(spotList(''), [])
  assert.deepEqual(spotList(undefined), [])
  assert.deepEqual(spotList('1,2;;3,4'), ['1,2', '3,4'])
})
