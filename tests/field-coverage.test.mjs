/* Coverage notes for views grouped by a sparsely-resolved taxonomic rank.
 *
 * The taxonomy pass above species has stalled, so a chart of families is drawn
 * over a slice of the store. These check the arithmetic that decides when a view
 * has to say so, and what it says.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { SPARSE_RANKS, coverageNote, fieldCoverage } from '../composables/fieldCoverage.ts'

// Rows can be flat or GeoJSON features; the helper reads both.
const flat = (n, field, present) => Array.from({ length: n }, (_, i) => (
  i < present ? { [field]: 'x' } : {}))
const feats = (n, field, present) => Array.from({ length: n }, (_, i) => ({
  properties: i < present ? { [field]: 'x' } : {},
}))

test('coverage is the fraction of rows carrying the field', () => {
  const cov = fieldCoverage(flat(100, 'family', 4), 'family')
  assert.equal(cov.present, 4)
  assert.equal(cov.total, 100)
  assert.equal(cov.fraction, 0.04)
})

test('coverage reads a GeoJSON feature the same as a flat row', () => {
  assert.deepEqual(
    fieldCoverage(feats(50, 'order', 10), 'order'),
    fieldCoverage(flat(50, 'order', 10), 'order'),
  )
})

test('empty data is zero coverage, not a divide-by-zero', () => {
  assert.deepEqual(fieldCoverage([], 'family'), { present: 0, total: 0, fraction: 0 })
})

test('a thin rank gets a note naming the fraction and the count', () => {
  const note = coverageNote(flat(100, 'family', 4), 'family')
  assert.match(note, /4%/)
  assert.match(note, /4 of 100/)
  assert.match(note, /family/)
})

test('under one percent is written as <1, not rounded to 0', () => {
  const note = coverageNote(flat(1000, 'order', 3), 'order')
  assert.match(note, /<1%/)
})

test('a well-populated rank says nothing', () => {
  // Genus and species are complete, and are not sparse ranks anyway.
  assert.equal(coverageNote(flat(100, 'species', 100), 'species'), '')
  // A family populated past the threshold is effectively the whole.
  assert.equal(coverageNote(flat(100, 'family', 95), 'family'), '')
})

test('a non-taxonomic field never gets a note', () => {
  assert.equal(coverageNote(flat(100, 'cluster', 2), 'cluster'), '')
  assert.ok(!SPARSE_RANKS.has('genus'))
  assert.ok(!SPARSE_RANKS.has('species'))
  assert.ok(SPARSE_RANKS.has('family'))
})
