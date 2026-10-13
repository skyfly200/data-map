import test from 'node:test'
import assert from 'node:assert/strict'
import { packDated } from '../netlify/lib/ee-runner.mjs'

const pts = (n, date) => Array.from({ length: n }, (_, i) => ({ index: i, date }))

test('many dates share one request instead of one request each', () => {
  const byDate = new Map(Array.from({ length: 100 }, (_, d) => [`2024-01-${d}`, pts(3, d)]))
  const batches = packDated(byDate, { datesPerRequest: 30 })
  assert.equal(batches.length, 4)
  assert.deepEqual(batches.map((b) => b.dates.size), [30, 30, 30, 10])
})

test('a request never exceeds the chunk size in points', () => {
  const byDate = new Map([['a', pts(400, 'a')], ['b', pts(400, 'b')], ['c', pts(1200, 'c')]])
  const batches = packDated(byDate, { datesPerRequest: 30, chunkSize: 500 })
  for (const b of batches) assert.ok(b.points <= 500)
  const total = batches.reduce((n, b) => n + b.points, 0)
  assert.equal(total, 2000)
  // The big date is split across requests, in order.
  const cParts = batches.flatMap((b) => b.parts.filter((p) => p.date === 'c'))
  assert.deepEqual(cParts.map((p) => p.points.length), [500, 500, 200])
})
