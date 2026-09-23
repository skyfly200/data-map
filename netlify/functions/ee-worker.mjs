// The worker that runs queued Earth Engine jobs.
//
// Scheduled, and also pokeable by an admin. Each invocation claims one job and
// runs it to completion, writing progress into the row as it goes; the next
// invocation takes the next job. One at a time on purpose — Earth Engine
// throttles a pile-up of concurrent requests, and scripts/ee_enrich.py found
// that the throttled failures show up as silently empty columns rather than as
// errors, which is the worst way for this to go wrong.
//
// The 300-second cap is the ceiling for one job. Anything longer needs an Earth
// Engine export task, which is a different shape of thing: submitted, not
// awaited. A job that would exceed this is refused at submission by the point
// limit rather than being started and killed halfway.

import { adminClient, requireAdmin } from '../lib/auth.mjs'
import {
  claimNextJob, failJob, finishJob, isCancelled, ownerViewer, planFor, reportProgress,
} from '../lib/job-queue.mjs'
import { explainEmpty, explainSelection, loadSource } from '../lib/job-source.mjs'
import { loadBaseline } from '../lib/baseline.mjs'
import { runModel, runPipeline } from '../lib/ee-runner.mjs'
import { uploadJson } from '../lib/datasets-store.mjs'
import { notifyJobSettled } from '../lib/notify.mjs'
import { logCronRun } from '../lib/cron-jobs.mjs'

// Every minute, so a job that misses the poke below still starts within a
// minute rather than up to five. The poke on submit is the fast path; this is
// the backstop for when it does not arrive (a cold submit, a dropped fetch).
export const config = { timeout: 300, schedule: '* * * * *' }

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json' },
})

// Progress is written to the database, and a job over many dates would
// otherwise write hundreds of times. Only report when the bar has visibly
// moved, or when the stage changes.
function throttled(jobId) {
  let lastFraction = -1
  let lastStage = null
  return async ({ fraction, stage, message }) => {
    if (stage === lastStage && fraction - lastFraction < 0.01 && fraction < 1) return
    lastFraction = fraction
    lastStage = stage
    await reportProgress(jobId, { fraction, stage, message })
  }
}

export default async function handler(request) {
  // A scheduled invocation carries no user. A manual poke has to be an admin, so
  // the queue cannot be driven by anyone who finds the URL — except the
  // submit endpoint's server-to-server poke, which carries a shared secret so a
  // member's job can start the worker the moment it is queued without making the
  // member an admin. The secret is optional: without it, only the cron and an
  // admin drive the queue, as before.
  const pokeSecret = process.env.WORKER_POKE_SECRET
  const poked = Boolean(pokeSecret) && request.headers.get('x-worker-secret') === pokeSecret
  const scheduled = request.headers.get('x-netlify-event') === 'schedule'
    || new URL(request.url).searchParams.get('scheduled') === '1'
  if (!scheduled && !poked) {
    const auth = await requireAdmin(request)
    if (!auth.ok) return auth.response
  }

  const workerId = `${process.env.AWS_LAMBDA_LOG_STREAM_NAME || 'worker'}-${Date.now()}`
  const t0 = Date.now()
  const job = await claimNextJob(workerId)
  if (!job) return json({ ok: true, claimed: null, message: 'Nothing queued.' })

  const spec = job.params || {}
  let spent = 0
  const _jobStart = Date.now()
  console.log(`[ee-worker] starting job ${job.id} (kind: ${spec.kind || 'enrich'})`)

  try {
    // Resolved as the member who submitted it, not as the worker. The worker's
    // client is the service role, which row-level security does not apply to,
    // so a dataset source has to be checked here or not at all — and the check
    // is re-run rather than trusted from submission time, because a job can
    // outlive the access that queued it.
    const features = await loadSource(spec.source, {
      client: adminClient(),
      viewer: await ownerViewer(job),
    })
    if (!features.length) {
      let msg
      if (spec.source?.type === 'dataset') {
        msg = `Dataset "${spec.source.slug}" exists but contains no observations.`
      } else {
        const baseline = await loadBaseline()
        const breakdown = explainSelection(baseline?.features || [], spec.source)
        msg = explainEmpty(spec.source, {
          total: (baseline?.features || []).length,
          inBounds: breakdown.inBounds,
          inDates: breakdown.inDates,
        })
      }
      throw new Error(msg)
    }

    // Cancelling is a member writing to their own row; the worker notices here
    // and between stages rather than being interrupted.
    if (await isCancelled(job.id)) return json({ ok: true, claimed: job.id, cancelled: true })

    const onProgress = throttled(job.id)

    // A model job produces a raster — a fitted suitability surface — not a table
    // of sampled points. Its result is a tile template stored in result_meta;
    // there is no GeoJSON file to write.
    if (spec.kind === 'model') {
      const { template, meta } = await runModel({ spec, features, onProgress })
      spent = job.estimated_units || 0
      await finishJob(job.id, { resultPath: null, costUnits: spent, meta: { ...meta, template } })
      // The member has almost certainly left the page by now; let them know.
      await notifyJobSettled({ ...job, status: 'succeeded', result_meta: { ...meta, template } })
      console.log(`[ee-worker] model job ${job.id} succeeded in ${Date.now() - _jobStart}ms (presences: ${meta.presences})`)
      await logCronRun(adminClient(), { jobId: 'ee-worker', status: 'ok', durationMs: Date.now() - t0, details: { jobId: job.id, kind: 'model', presences: meta.presences } })
      return json({ ok: true, claimed: job.id, model: true, presences: meta.presences })
    }

    const plan = planFor(job)
    const result = await runPipeline({ spec, features, plan, onProgress })
    spent = job.estimated_units || 0

    // Written under the job id, so a result is always traceable to the run that
    // produced it and two jobs cannot overwrite each other.
    const resultPath = `jobs/${job.user_id}/${job.id}.geojson`
    await uploadJson(resultPath, { type: 'FeatureCollection', features: result.features })

    const meta = {
      features: result.features.length,
      sampled: result.sampled,
      bands: result.bands,
      stages: spec.stages,
      skipped: result.skipped,
    }
    await finishJob(job.id, { resultPath, costUnits: spent, meta })
    await notifyJobSettled({ ...job, status: 'succeeded', result_meta: meta })
    console.log(`[ee-worker] job ${job.id} succeeded in ${Date.now() - _jobStart}ms (features: ${result.features.length})`)
    await logCronRun(adminClient(), { jobId: 'ee-worker', status: 'ok', durationMs: Date.now() - t0, details: { jobId: job.id, kind: 'enrich', features: result.features.length } })
    return json({ ok: true, claimed: job.id, features: result.features.length })
  } catch (err) {
    // Charged for what it spent before breaking: Earth Engine billed those
    // requests whether or not anything came back.
    console.error(`[ee-worker] job ${job.id} failed after ${Date.now() - _jobStart}ms:`, err?.message || err)
    await failJob(job.id, err, { costUnits: spent })
    await notifyJobSettled({ ...job, status: 'failed', error: String(err?.message || err) })
    await logCronRun(adminClient(), { jobId: 'ee-worker', status: 'error', durationMs: Date.now() - t0, details: { jobId: job.id, error: String(err?.message || err) } })
    return json({ ok: false, claimed: job.id, error: String(err.message || err) }, 200)
  }
}
