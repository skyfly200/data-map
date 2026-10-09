// Submitting and listing Earth Engine jobs.
//
//   GET  /.netlify/functions/ee-jobs          the caller's jobs, newest first
//   GET  /.netlify/functions/ee-jobs?all=1    every job (admins only)
//   POST /.netlify/functions/ee-jobs          submit one; body is the spec
//
// Submitting is gated on membership read FRESH from the database rather than
// from the token, because this is the path that spends Earth Engine quota and
// a token's tier claim can be an hour behind a cancelled membership.

import { adminClient, requireMemberFresh, requireUser } from '../lib/auth.mjs'
import { pokeWorker } from '../lib/worker-poke.mjs'
import { QueueError, listJobs, submitJob } from '../lib/job-queue.mjs'
import { SpecError } from '../lib/ee-pipeline.mjs'
import { DatasetAccessError, viewerFrom } from '../lib/dataset-access.mjs'
import { BaselineError, measureSource } from '../lib/job-source.mjs'
import { earthEngineConfigured } from '../lib/ee-runner.mjs'
import { atLeast } from '../lib/tiers.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json' },
})

export default async function handler(request) {
  if (request.method === 'GET') {
    const auth = await requireUser(request)
    if (!auth.ok) return auth.response
    const params = new URL(request.url).searchParams
    const all = params.get('all') === '1'
    const archived = params.get('archived') === '1'
    if (all && !atLeast(auth.tier, 'admin')) {
      return json({ ok: false, error: 'That is an administrator action.' }, 403)
    }
    try {
      const jobs = await listJobs(auth.user?.id, { all, archived })
      return json({ ok: true, jobs, tier: auth.tier })
    } catch (err) {
      return json({ ok: false, error: err.message }, err.status || 500)
    }
  }

  if (request.method === 'PATCH') {
    const auth = await requireUser(request)
    if (!auth.ok) return auth.response
    let body
    try { body = await request.json() } catch { return json({ ok: false, error: 'Send JSON.' }, 400) }
    const client = adminClient()
    if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)
    const userId = auth.user?.id

    if (body.action === 'archive') {
      const id = String(body.id || '').trim()
      if (!id) return json({ ok: false, error: 'Provide a job id.' }, 400)
      const { data: job } = await client.from('ee_jobs').select('id, user_id, status').eq('id', id).maybeSingle()
      if (!job || (job.user_id !== userId && !atLeast(auth.tier, 'admin'))) {
        return json({ ok: false, error: 'No such job.' }, 404)
      }
      if (job.status === 'queued' || job.status === 'running') {
        return json({ ok: false, error: 'Cannot archive a job that is still running.' }, 409)
      }
      const { error } = await client.from('ee_jobs').update({ archived_at: new Date().toISOString() }).eq('id', id)
      if (error) return json({ ok: false, error: error.message }, 500)
      return json({ ok: true, archived: id })
    }

    if (body.action === 'unarchive') {
      const id = String(body.id || '').trim()
      if (!id) return json({ ok: false, error: 'Provide a job id.' }, 400)
      const { data: job } = await client.from('ee_jobs').select('id, user_id').eq('id', id).maybeSingle()
      if (!job || (job.user_id !== userId && !atLeast(auth.tier, 'admin'))) {
        return json({ ok: false, error: 'No such job.' }, 404)
      }
      const { error } = await client.from('ee_jobs').update({ archived_at: null }).eq('id', id)
      if (error) return json({ ok: false, error: error.message }, 500)
      return json({ ok: true, unarchived: id })
    }

    // Archive all of a user's completed or failed jobs in one shot.
    if (body.action === 'archive_all') {
      const { error, count } = await client.from('ee_jobs')
        .update({ archived_at: new Date().toISOString() })
        .eq('user_id', userId)
        .is('archived_at', null)
        .in('status', ['succeeded', 'failed', 'cancelled'])
      if (error) return json({ ok: false, error: error.message }, 500)
      return json({ ok: true, archived: count ?? 0 })
    }

    return json({ ok: false, error: 'Unknown action. Use archive, unarchive, or archive_all.' }, 400)
  }

  if (request.method !== 'POST') return json({ ok: false, error: 'Use GET, PATCH or POST.' }, 405)

  const auth = await requireMemberFresh(request)
  if (!auth.ok) return auth.response

  // Refused up front rather than after queueing: a job that can never run
  // should not sit in a queue looking like it might.
  let body
  try {
    body = await request.json()
  } catch {
    return json({ ok: false, error: 'Send the job spec as JSON.' }, 400)
  }

  // Region loads (access_ingest) do not use Earth Engine.
  if (body?.kind !== 'access_ingest' && !earthEngineConfigured()) {
    return json({
      ok: false,
      error: 'Earth Engine is not configured on this deployment. '
        + 'Set EARTHENGINE_SERVICE_ACCOUNT_KEY and EARTHENGINE_PROJECT.',
    }, 503)
  }

  try {
    const result = await submitJob({
      user: auth.user,
      profile: auth.profile,
      spec: body,
      // Priced as the submitter, not as the server. A spec naming a dataset
      // they may not read is refused here, before it costs anything and before
      // it becomes a queued job that would read it later.
      counter: (spec) => measureSource(spec, {
        client: adminClient(),
        viewer: viewerFrom(auth),
      }),
    })
    // Only worth poking for a job that actually queued; a spec that resolved to
    // nothing to do has no worker to start.
    if (result?.job?.id) pokeWorker(request)
    return json({ ok: true, ...result })
  } catch (err) {
    // A SpecError is the member's spec being wrong and its message is written
    // for them; a QueueError carries its own status; anything else is ours.
    if (err instanceof SpecError) return json({ ok: false, error: err.message }, 400)
    // Baseline missing — server misconfiguration, not a bad request.
    if (err instanceof BaselineError) return json({ ok: false, error: err.message, code: err.code }, 503)
    // Naming a dataset they may not read. Reported as written — the message is
    // deliberately the same one an absent dataset gets.
    if (err instanceof DatasetAccessError) {
      return json({ ok: false, error: err.message, code: err.code }, err.status)
    }
    if (err instanceof QueueError) return json({ ok: false, error: err.message, code: err.code }, err.status)
    return json({ ok: false, error: 'Could not queue that job.' }, 500)
  }
}
