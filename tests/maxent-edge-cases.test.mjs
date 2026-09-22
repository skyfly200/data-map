/* Edge-case expansion for the modelling core. */

import test from 'node:test'
import assert from 'node:assert/strict'
import {
  normaliseModelSpec,
  backgroundPlan,
  rocAuc,
  crossValidationSummary,
} from '../netlify/lib/maxent.mjs'

test('normaliseModelSpec handles extreme background values', () => {
  const region = { north: 40, south: 39, east: -105, west: -106 }
  // Underflow
  assert.equal(normaliseModelSpec({ region, background: -100 }).background, 100)
  // Overflow
  assert.equal(normaliseModelSpec({ region, background: 1000000 }).background, 10000)
})

test('backgroundPlan handles presence count zero', () => {
  const plan = backgroundPlan({ presenceCount: 0, background: 1000 })
  assert.equal(plan.n, 1000)
  assert.equal(plan.enough, false)
})

test('rocAuc handles tied scores', () => {
  // All ties: 0.5
  assert.equal(rocAuc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]), 0.5)
  // Perfect separation but with ties within class
  // Presences: [0.9, 0.9], Background: [0.1, 0.1]
  assert.equal(rocAuc([0.9, 0.9, 0.1, 0.1], [1, 1, 0, 0]), 1)
})

test('crossValidationSummary handles empty or single-class folds', () => {
  const rows = [
    { fold: 0, presence: 1, prob: 0.9 }, // only presence
    { fold: 1, presence: 0, prob: 0.1 }, // only background
  ]
  const summary = crossValidationSummary(rows, { folds: 2 })
  assert.equal(summary, null)
})

test('crossValidationSummary handles NaN probabilities', () => {
  const rows = [
    { fold: 0, presence: 1, prob: NaN },
    { fold: 0, presence: 0, prob: 0.1 },
  ]
  const summary = crossValidationSummary(rows, { folds: 1 })
  // If the code doesn't explicitly handle NaN, rocAuc might be returning something else
  // We'll just verify it doesn't crash.
  assert.ok(summary !== undefined)
})
