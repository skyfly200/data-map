// Submitting and listing Earth Engine jobs.
//
//   GET  /.netlify/functions/ee-jobs          the caller's jobs, newest first
//   GET  /.netlify/functions/ee-jobs?all=1    every job (admins only)
//   POST /.netlify/functions/ee-jobs          submit one; body is the spec
//
// Submitting is gated on membership read FRESH from the database rather than
// from the token, because this is the path that spends Earth Engine quota and
// a token's tier claim can be an hour behind a cancelled membership.

import { requireMemberFresh, requireUser } from '../lib/auth.mjs'
import { QueueError, listJobs, submitJob } from '../lib/job-queue.mjs'
import { SpecError } from '../lib/ee-pipeline.mjs'
import { measureSource } from '../lib/job-source.mjs'
import { earthEngineConfigured } from '../lib/ee-runner.mjs'
import { atLeast } from '../lib/tiers.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json' },
})

export default async function handler(request) {
  if (request.method === 'GET') {
    const auth = await requireUser(request)
    if (!auth.ok) return auth.response
    const all = new URL(request.url).searchParams.get('all') === '1'
    if (all && !atLeast(auth.tier, 'admin')) {
      return json({ ok: false, error: 'That is an administrator action.' }, 403)
    }
    try {
      const jobs = await listJobs(auth.user?.id, { all })
      return json({ ok: true, jobs, tier: auth.tier })
    } catch (err) {
      return json({ ok: false, error: err.message }, err.status || 500)
    }
  }

  if (request.method !== 'POST') return json({ ok: false, error: 'Use GET or POST.' }, 405)

  const auth = await requireMemberFresh(request)
  if (!auth.ok) return auth.response

  // Refused up front rather than after queueing: a job that can never run
  // should not sit in a queue looking like it might.
  if (!earthEngineConfigured()) {
    return json({
      ok: false,
      error: 'Earth Engine is not configured on this deployment. '
        + 'Set EARTHENGINE_SERVICE_ACCOUNT_KEY and EARTHENGINE_PROJECT.',
    }, 503)
  }

  let body
  try {
    body = await request.json()
  } catch {
    return json({ ok: false, error: 'Send the job spec as JSON.' }, 400)
  }

  try {
    const result = await submitJob({
      user: auth.user,
      profile: auth.profile,
      spec: body,
      counter: measureSource,
    })
    return json({ ok: true, ...result })
  } catch (err) {
    // A SpecError is the member's spec being wrong and its message is written
    // for them; a QueueError carries its own status; anything else is ours.
    if (err instanceof SpecError) return json({ ok: false, error: err.message }, 400)
    if (err instanceof QueueError) return json({ ok: false, error: err.message, code: err.code }, err.status)
    return json({ ok: false, error: 'Could not queue that job.' }, 500)
  }
}
