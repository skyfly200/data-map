import test from 'node:test'
import assert from 'node:assert/strict'
import { sampleParts, PipelinePaused } from '../netlify/lib/ee-runner.mjs'

const part = (date, n) => ({ date, points: Array.from({ length: n }, (_, i) => ({ i })) })
// Rows are the part's date, so alignment is visible; any part dated 'bad' fails.
const request = (calls) => async (parts) => {
  calls.push(parts.map((p) => p.date))
  if (parts.some((p) => p.date === 'bad')) throw new Error('Image.select: Pattern \'B8\' did not match any bands.')
  return parts.flatMap((p) => p.points.map(() => ({ date: p.date })))
}

test('one failing date costs only its own points, not the whole batch', async () => {
  const calls = []
  const logs = []
  const skipped = { n: 0 }
  const parts = [part('a', 2), part('b', 1), part('bad', 3), part('c', 2)]
  const rows = await sampleParts(parts, request(calls), skipped, (m) => logs.push(m))
  assert.deepEqual(rows.map((r) => r.map((x) => x?.date ?? null)), [
    ['a', 'a'], ['b'], [null, null, null], ['c', 'c'],
  ])
  assert.equal(skipped.n, 3)
  assert.equal(logs.length, 1)
  assert.match(logs[0], /3 points for bad: Image\.select/)
  // Split in halves, not one request per date.
  assert.ok(calls.length <= 5)
})

test('a healthy batch is one request', async () => {
  const calls = []
  const skipped = { n: 0 }
  await sampleParts([part('a', 1), part('b', 1)], request(calls), skipped)
  assert.equal(calls.length, 1)
  assert.equal(skipped.n, 0)
})

test('running out of time pauses rather than skipping', async () => {
  const skipped = { n: 0 }
  await assert.rejects(
    sampleParts([part('a', 1), part('b', 1)], async () => { throw new PipelinePaused() }, skipped),
    PipelinePaused,
  )
  assert.equal(skipped.n, 0)
})
