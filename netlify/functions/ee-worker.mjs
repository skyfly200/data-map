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

import { requireAdmin } from '../lib/auth.mjs'
import {
  claimNextJob, failJob, finishJob, isCancelled, planFor, reportProgress,
} from '../lib/job-queue.mjs'
import { loadSource } from '../lib/job-source.mjs'
import { runPipeline } from '../lib/ee-runner.mjs'
import { uploadJson } from '../lib/datasets-store.mjs'

export const config = { timeout: 300, schedule: '*/5 * * * *' }

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
  // A scheduled invocation carries no user. A manual poke has to be an admin,
  // so the queue cannot be driven by anyone who finds the URL.
  const scheduled = request.headers.get('x-netlify-event') === 'schedule'
    || new URL(request.url).searchParams.get('scheduled') === '1'
  if (!scheduled) {
    const auth = await requireAdmin(request)
    if (!auth.ok) return auth.response
  }

  const workerId = `${process.env.AWS_LAMBDA_LOG_STREAM_NAME || 'worker'}-${Date.now()}`
  const job = await claimNextJob(workerId)
  if (!job) return json({ ok: true, claimed: null, message: 'Nothing queued.' })

  const spec = job.params || {}
  let spent = 0

  try {
    const features = await loadSource(spec.source)
    if (!features.length) throw new Error('That source no longer has any observations.')

    // Cancelling is a member writing to their own row; the worker notices here
    // and between stages rather than being interrupted.
    if (await isCancelled(job.id)) return json({ ok: true, claimed: job.id, cancelled: true })

    const plan = planFor(job)
    const onProgress = throttled(job.id)

    const result = await runPipeline({ spec, features, plan, onProgress })
    spent = job.estimated_units || 0

    // Written under the job id, so a result is always traceable to the run that
    // produced it and two jobs cannot overwrite each other.
    const resultPath = `jobs/${job.user_id}/${job.id}.geojson`
    await uploadJson(resultPath, { type: 'FeatureCollection', features: result.features })

    await finishJob(job.id, {
      resultPath,
      costUnits: spent,
      meta: {
        features: result.features.length,
        sampled: result.sampled,
        bands: result.bands,
        stages: spec.stages,
        skipped: result.skipped,
      },
    })
    return json({ ok: true, claimed: job.id, features: result.features.length })
  } catch (err) {
    // Charged for what it spent before breaking: Earth Engine billed those
    // requests whether or not anything came back.
    await failJob(job.id, err, { costUnits: spent })
    return json({ ok: false, claimed: job.id, error: String(err.message || err) }, 200)
  }
}
