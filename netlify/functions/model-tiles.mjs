// Re-serve a stored model's suitability surface.
//
//   GET /.netlify/functions/model-tiles?job=<jobId>
//
// A model job durably stores what it fitted — predictors, region, source and the
// cross-validation score — in its row. What it cannot store durably is the tile
// template: that carries an Earth Engine map id, which expires, so a surface
// saved on the map goes blank after a while. This mints a fresh template from the
// stored model on demand, so the map is always available without re-running the
// whole job. Within the result cache's TTL it is a blob read; past it, one mint.
//
// The score is not recomputed — it is read back from the job's stored result and
// returned alongside, so the surface still arrives with the number it earned.

import { adminClient, requireUser } from '../lib/auth.mjs'
import { ownerViewer } from '../lib/job-queue.mjs'
import { loadSource } from '../lib/job-source.mjs'
import { earthEngineConfigured, remintSuitability } from '../lib/ee-runner.mjs'
import { effectiveTier } from '../lib/tiers.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

export default async function handler(request) {
  if (request.method !== 'GET') return json({ ok: false, error: 'Use GET.' }, 405)

  const auth = await requireUser(request)
  if (!auth.ok) return auth.response

  const reqUrl = new URL(request.url)
  const explicitJobId = reqUrl.searchParams.get('job')
  const configId = reqUrl.searchParams.get('model')

  if (!explicitJobId && !configId) return json({ ok: false, error: 'Name a model job or config.' }, 400)

  const client = adminClient()
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)

  // ?model=configId resolves to the latest succeeded job for that config.
  let jobId = explicitJobId
  if (!jobId && configId) {
    const { data: run, error: runErr } = await client
      .from('model_runs')
      .select('job_id')
      .eq('config_id', configId)
      .eq('status', 'succeeded')
      .order('created_at', { ascending: false })
      .limit(1)
      .maybeSingle()
    if (runErr) return json({ ok: false, error: runErr.message }, 500)
    if (!run?.job_id) return json({ ok: false, error: 'No succeeded run for that model config.' }, 404)
    jobId = run.job_id
  }

  const { data: job, error } = await client
    .from('ee_jobs')
    .select('id, user_id, kind, status, params, result_meta')
    .eq('id', jobId)
    .maybeSingle()
  if (error) return json({ ok: false, error: error.message }, 500)
  if (!job) return json({ ok: false, error: 'No such model.' }, 404)

  // A model is the member's own to re-serve; an admin may re-serve any. Read past
  // the token's tier so a demoted admin cannot, matching the rest of the app.
  const isAdmin = auth.user ? effectiveTier(await profileOf(client, auth.user.id)) === 'admin' : true
  if (auth.user && job.user_id !== auth.user.id && !isAdmin) {
    return json({ ok: false, error: 'That model is not yours.' }, 403)
  }
  if (job.kind !== 'model') return json({ ok: false, error: 'That job is not a model.' }, 400)
  if (job.status !== 'succeeded') return json({ ok: false, error: 'That model has not finished.' }, 409)

  if (!earthEngineConfigured()) {
    return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
  }

  try {
    const spec = job.params || {}
    // Resolved as the member who owns the job, not as the service account, so a
    // dataset source is read under the same rules the job ran under.
    const features = await loadSource(spec.source, { client, viewer: await ownerViewer(job) })
    if (!features.length) return json({ ok: false, error: "That model's source no longer has any observations." }, 404)

    const { template, meta } = await remintSuitability({ spec, features })
    // The score is durable on the job; carry it back so the surface still shows it.
    return json({ ok: true, template, meta: { ...meta, cv: job.result_meta?.cv ?? null } })
  } catch (err) {
    return json({ ok: false, error: String(err?.message || err).slice(0, 300) }, 502)
  }
}

/** A member's profile row, for the admin check. */
async function profileOf(client, userId) {
  const { data } = await client.from('profiles').select('tier, member_until').eq('user_id', userId).maybeSingle()
  return data
}
