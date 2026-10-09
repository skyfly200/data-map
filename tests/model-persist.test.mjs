/* What a finished model writes back for the foray planner (WANT-17 phase 3):
 * the contributions and the predictor ranges, shaped for model_results. */

import test from 'node:test'
import assert from 'node:assert/strict'

import { contributionPayload } from '../netlify/lib/model-persist.mjs'
import { shapePredictorRanges } from '../netlify/lib/maxent.mjs'

test('shapePredictorRanges reads the repeated percentile lists in predictor order', () => {
  const raw = { p0: [1500, 2], p25: [2100, 8], p75: [2900, 20], p100: [3400, 41] }
  assert.deepEqual(shapePredictorRanges(raw, ['elevation', 'slope']), {
    elevation: { p25: 2100, p75: 2900, min: 1500, max: 3400 },
    slope: { p25: 8, p75: 20, min: 2, max: 41 },
  })
})

test('shapePredictorRanges drops a predictor masked at every presence', () => {
  const raw = { p0: [1500, null], p25: [2100, null], p75: [2900, null], p100: [3400, null] }
  assert.deepEqual(Object.keys(shapePredictorRanges(raw, ['elevation', 'ndvi'])), ['elevation'])
})

test('shapePredictorRanges returns null when nothing usable came back', () => {
  assert.equal(shapePredictorRanges(null, ['elevation']), null)
  assert.equal(shapePredictorRanges({ p0: [null], p25: [null], p75: [null], p100: [null] }, ['elevation']), null)
})

test('contributionPayload keeps ranges even without contributions', () => {
  const ranges = { elevation: { p25: 1, p75: 2, min: 0, max: 3 } }
  assert.deepEqual(contributionPayload({ predictorRanges: ranges }), { contributions: null, predictor_ranges: ranges })
  assert.deepEqual(contributionPayload({ contributions: { elevation: 60 } }), { contributions: { elevation: 60 }, predictor_ranges: null })
  assert.equal(contributionPayload({ contributions: {}, predictorRanges: null }), null)
  assert.equal(contributionPayload(null), null)
})
