import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { toDaily, vercelSchedule } from '../scripts/vercel-cron.mjs'

test('sub-daily schedules become one run a day', () => {
  assert.equal(toDaily('* * * * *'), '0 0 * * *')
  assert.equal(toDaily('0 */6 * * *'), '0 0 * * *')
  assert.equal(toDaily('*/15 3 * * *'), '0 3 * * *')
})

test('a schedule that is already daily or rarer is left alone', () => {
  assert.equal(toDaily('30 4 * * *'), '30 4 * * *')
  assert.equal(toDaily('0 2 * * 1'), '0 2 * * 1')
})

test('VERCEL_PLAN=pro keeps the Netlify schedule', () => {
  assert.equal(vercelSchedule('* * * * *', 'pro'), '* * * * *')
  assert.equal(vercelSchedule('* * * * *', undefined), '0 0 * * *')
})

test('committed vercel.json has no cron Hobby would reject', () => {
  const { crons } = JSON.parse(readFileSync(new URL('../vercel.json', import.meta.url), 'utf8'))
  for (const c of crons) {
    const [min, hour] = c.schedule.split(/\s+/)
    assert.match(min, /^\d+$/, `${c.path}: ${c.schedule}`)
    assert.match(hour, /^\d+$/, `${c.path}: ${c.schedule}`)
  }
})
