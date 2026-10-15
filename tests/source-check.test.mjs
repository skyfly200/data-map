import test from 'node:test'
import assert from 'node:assert/strict'
import { bboxIssue, bboxSource } from '../composables/sourceCheck.ts'
import { QueueError, checkSource } from '../netlify/lib/job-queue.mjs'

const box = { north: 41, south: 37, west: -109, east: -102, dateFrom: '2024-01-01', dateTo: '2025-01-01', taxon: '' }

test('a sound box has no issue', () => {
  assert.equal(bboxIssue(box), '')
})

test('an emptied edge is named, not read as zero', () => {
  assert.equal(bboxIssue({ ...box, north: '' }), 'Fill in the north edge.')
  assert.equal(bboxIssue({ ...box, west: null, east: undefined }), 'Fill in the west, east edges.')
})

test('the server rules are the ones applied', () => {
  assert.match(bboxIssue({ ...box, north: 36 }), /north edge must be above its south/)
  assert.match(bboxIssue({ ...box, north: 95 }), /north must be between -90 and 90/)
  assert.match(bboxIssue({ ...box, west: -130 }), /larger than 20 degrees/)
  assert.match(bboxIssue({ ...box, dateFrom: '2025-06-01' }), /start date is after the end date/)
  assert.match(bboxIssue({ ...box, dateFrom: '2015-01-01' }), /longer than five years/)
})

test('turning the date filter off drops the dates', () => {
  assert.equal(bboxIssue({ ...box, dateFrom: '2025-06-01', noDateFilter: true }), '')
  assert.equal(bboxSource({ ...box, noDateFilter: true }).dateFrom, undefined)
})

const member = { tier: 'member', ee_max_points: 100 }

test('the dry run reports points without queueing', async () => {
  const out = await checkSource({ profile: member, spec: { kind: 'enrich', source: { type: 'bbox', bounds: box } }, counter: async () => ({ points: 12, dates: 3 }) })
  assert.deepEqual(out, { points: 12, dates: 3 })
})

test('the dry run refuses an empty area with the reason', async () => {
  const counter = async () => ({ points: 0, breakdown: { total: 50, inBounds: 0, inDates: 0 } })
  await assert.rejects(checkSource({ profile: member, spec: { kind: 'enrich', source: { type: 'bbox', bounds: box } }, counter }),
    (err) => err instanceof QueueError && err.code === 'empty_source' && /No observations fall inside/.test(err.message))
})

test('the dry run refuses more points than the member may sample', async () => {
  const counter = async () => ({ points: 500, dates: 1 })
  await assert.rejects(checkSource({ profile: member, spec: { kind: 'enrich', source: { type: 'bbox', bounds: box } }, counter }),
    (err) => err.code === 'too_many_points')
})
