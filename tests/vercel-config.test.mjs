/* vercel.json generation.
 *
 * Vercel's Hobby plan rejects a deployment outright when any cron runs more
 * than once a day, so the generator relaxes sub-daily schedules unless told the
 * account is on Pro, and the committed vercel.json must match its output.
 */

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'

import { buildVercelConfig, hobbySchedule } from '../scripts/gen-vercel-config.mjs'

test('schedules that already run at most daily are left alone', () => {
  assert.equal(hobbySchedule('0 3 * * *'), '0 3 * * *')
  assert.equal(hobbySchedule('15 4 * * 1'), '15 4 * * 1')
})

test('sub-daily schedules become once a day', () => {
  assert.equal(hobbySchedule('* * * * *'), '0 0 * * *')
  assert.equal(hobbySchedule('0 */6 * * *'), '0 0 * * *')
  assert.equal(hobbySchedule('*/5 2 * * *'), '0 2 * * *')
  assert.equal(hobbySchedule('30 1,13 * * *'), '30 1 * * *')
})

test('the Pro plan keeps the Netlify schedules', () => {
  const pro = buildVercelConfig({ plan: 'pro' })
  assert.ok(pro.crons.some((c) => c.schedule === '* * * * *'))
})

test('the committed vercel.json is what the generator writes for Hobby', () => {
  const committed = JSON.parse(readFileSync('vercel.json', 'utf8'))
  assert.deepEqual(committed, buildVercelConfig())
  for (const { schedule } of committed.crons) {
    assert.equal(hobbySchedule(schedule), schedule, `${schedule} runs more than daily`)
  }
})
