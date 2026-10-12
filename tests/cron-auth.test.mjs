import test from 'node:test'
import assert from 'node:assert/strict'
import { isScheduledInvocation } from '../netlify/lib/cron-auth.mjs'

const req = (init = {}, url = 'https://x.test/.netlify/functions/ee-worker') => new Request(url, init)

test('a query string cannot claim to be the scheduler', async () => {
  assert.equal(await isScheduledInvocation(req({}, 'https://x.test/api/fn/ee-worker?scheduled=1'), {}), false)
})

test('the Netlify scheduler body is recognised off Vercel only', async () => {
  const init = { method: 'POST', body: JSON.stringify({ next_run: '2026-10-09T19:00:00Z' }) }
  assert.equal(await isScheduledInvocation(req(init), {}), true)
  assert.equal(await isScheduledInvocation(req(init), { VERCEL: '1' }), false)
})

test('a Vercel cron is recognised by CRON_SECRET', async () => {
  const env = { VERCEL: '1', CRON_SECRET: 's3cret' }
  assert.equal(await isScheduledInvocation(req({ headers: { authorization: 'Bearer s3cret' } }), env), true)
  assert.equal(await isScheduledInvocation(req({ headers: { authorization: 'Bearer nope' } }), env), false)
  assert.equal(await isScheduledInvocation(req(), env), false)
})

test('reading the body to check does not consume it', async () => {
  const r = req({ method: 'POST', body: '{"next_run":"x"}' })
  await isScheduledInvocation(r, {})
  assert.deepEqual(await r.json(), { next_run: 'x' })
})
