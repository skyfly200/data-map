/* Open Location Code encoding, checked against the reference test vectors.
 *
 * If these pass, the map's dropped-point plus code matches every other tool
 * that speaks OLC — a code that is subtly wrong is worse than none, because it
 * points somewhere real but not here.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { encodePlusCode } from '../composables/plusCode.js'

test('encodes the reference vectors at length 10', () => {
  // From the official Open Location Code test data.
  assert.equal(encodePlusCode(47.0000625, 8.0000625, 10), '8FVC2222+22')
  assert.equal(encodePlusCode(-41.2730625, 174.7859375, 10), '4VCPPQGP+Q9')
})

test('a shorter code is a coarser area, padded to the separator', () => {
  assert.equal(encodePlusCode(20.375, 2.775, 6), '7FG49Q00+')
})

test('a length-11 code refines the length-10 one rather than replacing it', () => {
  // The finer code is the coarser code with one more digit, so a reader can
  // shorten it by dropping from the right.
  const ten = encodePlusCode(37.539669, -122.375069, 10)
  const eleven = encodePlusCode(37.539669, -122.375069, 11)
  assert.equal(eleven.length, ten.length + 1)
  assert.equal(eleven.slice(0, ten.length), ten)
})

test('longitude wraps rather than clamping', () => {
  // 180 and -180 are the same meridian, so they encode to the same column.
  const a = encodePlusCode(0, 180, 10)
  const b = encodePlusCode(0, -180, 10)
  assert.equal(a, b)
})

test('latitude is clipped into range without throwing', () => {
  assert.equal(typeof encodePlusCode(95, 0, 10), 'string')
  assert.equal(typeof encodePlusCode(-95, 0, 10), 'string')
})
