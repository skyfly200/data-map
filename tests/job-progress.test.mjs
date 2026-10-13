import test from 'node:test'
import assert from 'node:assert/strict'
import { jobPercent, jobStalled, STALLED_AFTER_MS } from '../composables/jobProgress.ts'
import { MAX_ATTEMPTS, STALLED_JOB_ERROR, reclaimDecision } from '../netlify/lib/job-queue.mjs'

test('progress is stored as a fraction and shown as a percent', () => {
  assert.equal(jobPercent({ progress: 0.42 }), 42)
  assert.equal(jobPercent({ progress: 1 }), 100)
  assert.equal(jobPercent({ progress: 0 }), 0)
  assert.equal(jobPercent({ progress: null }), 0)
  assert.equal(jobPercent(null), 0)
  // An already-percent value is not multiplied past 100.
  assert.equal(jobPercent({ progress: 55 }), 55)
  assert.equal(jobPercent({ progress: 250 }), 100)
})

test('a running job with no recent report reads as stalled', () => {
  const now = Date.parse('2026-10-13T12:00:00Z')
  const ago = (ms) => new Date(now - ms).toISOString()
  assert.equal(jobStalled({ status: 'running', locked_at: ago(30_000) }, now), false)
  assert.equal(jobStalled({ status: 'running', locked_at: ago(STALLED_AFTER_MS + 1000) }, now), true)
  assert.equal(jobStalled({ status: 'running', started_at: ago(STALLED_AFTER_MS + 1000) }, now), true)
  assert.equal(jobStalled({ status: 'queued', locked_at: ago(STALLED_AFTER_MS * 5) }, now), false)
  assert.equal(jobStalled({ status: 'running' }, now), false)
})

test('a queued job is claimed as its first attempt', () => {
  assert.deepEqual(reclaimDecision({ status: 'queued', params: {} }), { attempt: 1 })
})

test('a stale running job is retried once, then failed', () => {
  assert.deepEqual(reclaimDecision({ status: 'running', params: { attempts: 1 } }), { attempt: 2 })
  // Claimed before attempts were counted: it has had one run already.
  assert.deepEqual(reclaimDecision({ status: 'running', params: {} }), { attempt: 2 })
  assert.deepEqual(reclaimDecision({ status: 'running', params: { attempts: MAX_ATTEMPTS } }), { fail: STALLED_JOB_ERROR })
})

test('a paused run continues only when it got further than the last one', async () => {
  const { continuationDecision, checkpointProgress, NO_PROGRESS_ERROR, MAX_CONTINUATIONS } = await import('../netlify/lib/job-queue.mjs')
  const cp = { stagesDone: ['terrain', 'landcover'], datesDone: { soil_moisture: ['2024-01-01', '2024-01-02'] } }
  assert.equal(checkpointProgress(cp), 4)
  assert.deepEqual(continuationDecision({}, cp), { progress: 4, continuations: 1 })
  assert.deepEqual(continuationDecision({ checkpoint_progress: 4, continuations: 1 }, cp), { fail: NO_PROGRESS_ERROR })
  assert.deepEqual(continuationDecision({ checkpoint_progress: 2, continuations: MAX_CONTINUATIONS }, cp), { fail: NO_PROGRESS_ERROR })
})
