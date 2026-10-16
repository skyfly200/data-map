import test from 'node:test'
import assert from 'node:assert/strict'
import { requestWait } from '../netlify/lib/ee-runner.mjs'

test('a request is not waited on past the run being killed', () => {
  // The 30-second scheduled worker, 10 s into the run: 15 s left, not 10 minutes.
  assert.deepEqual(requestWait(600_000, 25_000, 10_000), { wait: 15_000, cut: true })
  // Already past it: give up at once.
  assert.deepEqual(requestWait(600_000, 25_000, 30_000), { wait: 0, cut: true })
  // No deadline configured still stops at the kill.
  assert.deepEqual(requestWait(0, 25_000, 10_000), { wait: 15_000, cut: true })
})

test('the request deadline applies when the run has longer left', () => {
  assert.deepEqual(requestWait(600_000, 0, 10_000), { wait: 600_000, cut: false })
  assert.deepEqual(requestWait(60_000, 900_000, 10_000), { wait: 60_000, cut: false })
  assert.deepEqual(requestWait(0, 0, 10_000), { wait: 0, cut: false })
})
