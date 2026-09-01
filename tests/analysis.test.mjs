import test from 'node:test'
import assert from 'node:assert/strict'

import { spearman, meanSd, median, fieldValue } from '../composables/statistics.js'

// The composable needs a Nuxt runtime; the statistics do not. These are the
// parts where being subtly wrong is invisible — a correlation that reads 0.9
// looks just as convincing whether or not the maths behind it is right.

test('spearman is +1 on any increasing relationship, linear or not', () => {
  const xs = [1, 2, 3, 4, 5, 6]
  assert.equal(spearman(xs, [1, 2, 3, 4, 5, 6]), 1)
  // Rank-based, so a bent-but-increasing curve is still a perfect 1 — this is
  // exactly why it is used here instead of Pearson.
  assert.equal(spearman(xs, [1, 4, 9, 16, 25, 36]), 1)
  assert.equal(spearman(xs, [0.1, 0.5, 9, 400, 5000, 99999]), 1)
})

test('spearman is -1 when one falls as the other rises', () => {
  assert.equal(spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1)
})

test('ties get average ranks', () => {
  // Every value tied means no variance in the ranks and no defined correlation,
  // rather than a spurious number.
  assert.equal(spearman([1, 1, 1, 1], [1, 2, 3, 4]), null)
  // A partial tie still yields a sensible positive coefficient.
  const rho = spearman([1, 2, 2, 3], [1, 2, 3, 4])
  assert.ok(rho > 0.8 && rho <= 1, `got ${rho}`)
})

test('spearman refuses samples too small to mean anything', () => {
  assert.equal(spearman([1, 2], [1, 2]), null)
  assert.equal(spearman([], []), null)
})

test('an unrelated pair scores near zero', () => {
  const xs = [1, 2, 3, 4, 5, 6, 7, 8]
  const ys = [5, 2, 8, 1, 7, 3, 6, 4]
  assert.ok(Math.abs(spearman(xs, ys)) < 0.5)
})

test('meanSd ignores gaps rather than treating them as zero', () => {
  // This is the bug that made the cluster heatmap meaningless: Number(null) is
  // 0, so a missing reading became a real zero and dragged the mean down.
  const { n, mean, sd } = meanSd([2, 4, null, 6, null])
  assert.equal(n, 3)
  assert.equal(mean, 4)
  assert.ok(Math.abs(sd - Math.sqrt(8 / 3)) < 1e-9)
})

test('meanSd on nothing reports nothing, not zero', () => {
  assert.deepEqual(meanSd([]), { n: 0, mean: null, sd: null })
  assert.deepEqual(meanSd([null, null]), { n: 0, mean: null, sd: null })
})

test('median handles both parities and ignores gaps', () => {
  assert.equal(median([3, 1, 2]), 2)
  assert.equal(median([4, 1, 3, 2]), 2.5)
  assert.equal(median([5, null, 1, null, 3]), 3)
  assert.equal(median([]), null)
})

test('median is not thrown off by an outlier the way a mean would be', () => {
  // Why the season-timing chart uses it: a few winter records must not move
  // "the middle of the season".
  const doys = [240, 245, 250, 255, 260, 5]
  assert.equal(median(doys), 247.5)
  const mean = doys.reduce((s, v) => s + v, 0) / doys.length
  assert.ok(mean < 220, 'the mean really is dragged down')
})

test('rain7 sums the seven-day history and needs at least one reading', () => {
  const row = { prcp_d0: 1, prcp_d1: 2, prcp_d2: 3 }
  assert.equal(fieldValue(row, 'rain7'), 6)
  // No readings at all is missing data, not zero rainfall.
  assert.equal(fieldValue({}, 'rain7'), null)
  // A single reading counts.
  assert.equal(fieldValue({ prcp_d3: 4 }, 'rain7'), 4)
})

test('fieldValue rejects blanks instead of coercing them to zero', () => {
  for (const bad of [null, undefined, '', 'not a number', NaN]) {
    assert.equal(fieldValue({ elevation: bad }, 'elevation'), null, `${bad} became a number`)
  }
  assert.equal(fieldValue({ elevation: '2500' }, 'elevation'), 2500)
  assert.equal(fieldValue({ elevation: 0 }, 'elevation'), 0)
})
