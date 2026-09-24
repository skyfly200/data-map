// MaxEnt Modeling API
//
//   POST  /.netlify/functions/modeling/maxent/train     - Submit training job
//   GET   /.netlify/functions/modeling/maxent/results/:id - Fetch results for a job
//   GET   /.netlify/functions/modeling/maxent/models    - List user's saved models
//   DELETE /.netlify/functions/modeling/maxent/models/:id - Delete a model
//
// This function handles the lifecycle of a MaxEnt model: 
// 1. Validating the spec -> 2. Creating DB records -> 3. Triggering EE job.

import { adminClient, requireMemberFresh, requireUser } from '../lib/auth.mjs'
import { normaliseModelSpec, estimateModelUnits } from '../lib/maxent.mjs'
import { checkVisibility, viewerFrom } from '../lib/dataset-access.mjs'
import { runEvaluation, earthEngineConfigured } from '../lib/ee-runner.mjs'
import { loadSource, measureSource } from '../lib/job-source.mjs'
import { submitJob } from '../lib/job-queue.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

function fail(err) {
  return json({ ok: false, error: String(err?.message || err) }, err.status || 500)
}

/** Submit a training job to Earth Engine. */
async function train(client, auth, body) {
  const viewer = viewerFrom(auth)

  // 1. Validate and Normalise
  const spec = normaliseModelSpec(body)

  // 2. Create Configuration
  const visibility = checkVisibility(body.visibility, viewer)
  const { data: config, error: configErr } = await client.from('model_configs').insert({
    owner_id: viewer.userId,
    title: spec.title,
    description: body.description || null,
    predictors: spec.predictors,
    background_count: spec.background,
    effort_weighted: true,
    projection_region: spec.region,
    source_dataset_id: body.source_dataset_id,
    visibility,
  }).select().single()

  if (configErr) throw new Error(configErr.message)

  // 3. Submit via the standard job queue (same path as ee-jobs.mjs).
  const result = await submitJob({
    user: auth.user,
    profile: auth.profile,
    spec: { type: 'maxent', config_id: config.id, ...spec },
    counter: (s) => measureSource(s, { client, viewer }),
  })
  const jobId = result.job?.id
  if (!jobId) throw new Error('Job queue did not return a job ID.')

  // 4. Record the Run
  const { error: runErr } = await client.from('model_runs').insert({
    config_id: config.id,
    job_id: jobId,
    status: 'pending',
  })

  if (runErr) throw new Error(runErr.message)

  return json({ ok: true, config, jobId, status: 'submitted' })
}

/** Fetch results for a specific run. */
async function getResults(client, viewer, jobId) {
  const { data: run, error: runErr } = await client.from('model_runs')
    .select('*, model_configs(*)')
    .eq('job_id', jobId).maybeSingle()

  if (runErr) throw new Error(runErr.message)
  if (!run || (run.model_configs.owner_id !== viewer.userId)) {
    return json({ ok: false, error: 'Job not found or access denied.' }, 404)
  }

  const { data: result, error: resErr } = await client.from('model_results')
    .select('*').eq('run_id', run.id).maybeSingle()

  if (resErr) throw new Error(resErr.message)
  if (!result) return json({ ok: true, status: 'pending', jobId })

  return json({ ok: true, result, run })
}

/** List the user's saved models. */
async function listModels(client, viewer) {
  const { data, error } = await client.from('model_configs')
    .select('*, model_results(*)')
    .eq('owner_id', viewer.userId).order('created_at', { ascending: false })

  if (error) throw new Error(error.message)
  return json({ ok: true, models: data || [] })
}

/**
 * Run evaluation for a saved model run and return niche + response-curve data.
 *
 * The evaluation re-fits the model from the stored spec so it can predict
 * synthetic sweeps, then samples the presence and background covariates for the
 * niche-overlap visualisations. Heavy: expect ~30–60 seconds for a typical job.
 */
async function evaluate(client, viewer, jobId) {
  if (!earthEngineConfigured()) {
    return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
  }

  const { data: run, error: runErr } = await client.from('model_runs')
    .select('*, model_configs(*)')
    .eq('job_id', jobId).maybeSingle()

  if (runErr) throw new Error(runErr.message)
  if (!run || run.model_configs?.owner_id !== viewer.userId) {
    return json({ ok: false, error: 'Job not found or access denied.' }, 404)
  }
  if (run.status !== 'succeeded') {
    return json({ ok: false, error: 'Evaluation is only available for succeeded jobs.' }, 409)
  }

  const config = run.model_configs
  const source = config.source_dataset_id
    ? { type: 'dataset', slug: config.source_dataset_id }
    : null
  if (!source) return json({ ok: false, error: 'This model has no linked source dataset.' }, 400)
  const spec = { predictors: config.predictors, background: config.background_count, region: config.projection_region }
  const features = await loadSource(source, { client, viewer })
  if (!features.length) throw new Error('That model has no usable observations.')

  const result = await runEvaluation({ spec, features })
  return json({ ok: true, jobId, ...result })
}

/** Delete a model configuration. */
async function removeModel(client, viewer, body) {
  const id = String(body.id || '').trim()
  if (!id) return json({ ok: false, error: 'Missing model ID.' }, 400)

  const { data: row, error: readErr } = await client.from('model_configs')
    .select('id, owner_id').eq('id', id).maybeSingle()

  if (readErr) throw new Error(readErr.message)
  if (!row || row.owner_id !== viewer.userId) {
    return json({ ok: false, error: 'Model not found or access denied.' }, 404)
  }

  const { error } = await client.from('model_configs').delete().eq('id', id)
  if (error) throw new Error(error.message)

  return json({ ok: true, deleted: id })
}

export default async function handler(request) {
  const client = adminClient()
  if (!client) return json({ ok: false, error: 'Supabase not configured.' }, 503)

  try {
    const url = new URL(request.url)
    const path = url.pathname
    const method = request.method

    // Simplified routing for the prototype
    if (method === 'GET' && path.includes('/modeling/maxent/models')) {
      const auth = await requireUser(request)
      if (!auth.ok) return auth.response
      return await listModels(client, viewerFrom(auth))
    }

    if (method === 'GET' && path.includes('/modeling/maxent/results')) {
      const auth = await requireUser(request)
      if (!auth.ok) return auth.response
      const jobId = path.split('/').pop()
      return await getResults(client, viewerFrom(auth), jobId)
    }

    if (method === 'POST' && path.includes('/modeling/maxent/train')) {
      const auth = await requireMemberFresh(request)
      if (!auth.ok) return auth.response
      return await train(client, auth, await request.json())
    }

    if (method === 'DELETE' && path.includes('/modeling/maxent/models')) {
      const auth = await requireMemberFresh(request)
      if (!auth.ok) return auth.response
      return await removeModel(client, viewerFrom(auth), await request.json())
    }

    // GET /.netlify/functions/modeling/maxent/evaluate/:jobId
    // Returns niche distribution and response-curve data for a completed model run.
    if (method === 'GET' && path.includes('/modeling/maxent/evaluate')) {
      const auth = await requireMemberFresh(request)
      if (!auth.ok) return auth.response
      const jobId = path.split('/').pop()
      return await evaluate(client, viewerFrom(auth), jobId)
    }

    return json({ ok: false, error: 'Not Found' }, 404)
  } catch (err) {
    return fail(err)
  }
}
