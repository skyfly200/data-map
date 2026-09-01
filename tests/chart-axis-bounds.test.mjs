import test from 'node:test'
import assert from 'node:assert/strict'

import { boundsFor, clampDomain } from '../composables/useChartFields.js'

// Charts pad their axes past the data so marks are not flush against the frame.
// Unclamped, that padding invents readings the quantity cannot take.

test('a compass aspect axis never leaves the circle', () => {
  // The reported symptom: an aspect axis drawn to 370-378°.
  assert.deepEqual(clampDomain([-18, 378], boundsFor('aspect'), [0, 360]), [0, 360])
})

test('day of year stays within the calendar', () => {
  assert.deepEqual(clampDomain([-18, 383], boundsFor('day_of_year'), [1, 365]), [1, 366])
})

test('bounded indices stay inside their range', () => {
  assert.deepEqual(clampDomain([-4, 94], boundsFor('slope'), [0, 90]), [0, 90])
  assert.deepEqual(clampDomain([-1.2, 1.2], boundsFor('ndvi'), [-1, 1]), [-1, 1])
  assert.deepEqual(clampDomain([-0.05, 1.05], boundsFor('soil_moisture'), [0, 1]), [0, 1])
})

test('a half-bounded field is clamped only where it has a bound', () => {
  // Elevation cannot go below zero, but has no ceiling worth imposing.
  assert.deepEqual(clampDomain([-500, 15000], boundsFor('elevation'), [0, 14000]), [0, 15000])
})

test('temperature is genuinely unbounded and is left alone', () => {
  assert.equal(boundsFor('tmax'), null)
  assert.deepEqual(clampDomain([-30, 110], boundsFor('tmax'), [-20, 100]), [-30, 110])
})

test('an unregistered field falls back to the sign of its own data', () => {
  // Counts and totals should not sprout a negative axis just from padding...
  assert.deepEqual(clampDomain([-5, 120], null, [0, 100]), [0, 120])
  // ...but a series that really does go negative keeps its room.
  assert.deepEqual(clampDomain([-20, 40], null, [-10, 30]), [-20, 40])
})

test('no values means no inference', () => {
  assert.deepEqual(clampDomain([-5, 5], null, []), [-5, 5])
})

test('clamping never collapses the domain to zero width', () => {
  // An empty chart falls back to [0, 1]; clamping that to a field whose floor is
  // 1 pinned both ends to 1, and every scale then divided by zero and painted
  // the SVG with NaN coordinates.
  const [lo, hi] = clampDomain([0, 1], boundsFor('day_of_year'), [])
  assert.ok(hi > lo, `degenerate domain [${lo}, ${hi}]`)

  // The same at the top end: a range pinned against a ceiling must widen
  // downward rather than upward, staying inside the bound.
  const [lo2, hi2] = clampDomain([360, 400], boundsFor('aspect'), [])
  assert.ok(hi2 > lo2, `degenerate domain [${lo2}, ${hi2}]`)
  assert.ok(hi2 <= 360, 'must not exceed the upper bound')
})

test('a normal domain is left exactly as it was', () => {
  assert.deepEqual(clampDomain([10, 90], boundsFor('slope'), [20, 80]), [10, 90])
})
