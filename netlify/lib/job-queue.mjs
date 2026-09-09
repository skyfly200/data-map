// The job queue, as rows in Postgres.
//
// Earth Engine work does not fit inside a request. An interactive sample is
// seconds to a minute, a job over a few thousand points is minutes, and neither
// is something to hold an HTTP connection open for. So submitting a job writes
// a row, a worker claims it and reports progress into the same row, and the
// member's browser watches that row. Nothing is held in memory between
// requests, which is what makes it survive a function cold start.
//
// The claim is the part worth being careful about: two workers must never run
// one job, or the society pays twice for the same result. See claimNextJob.

import { adminClient } from './auth.mjs'
import { estimateUnits, summariseUsage, checkQuota } from './quotas.mjs'
import { STAGES, normaliseSpec, progressPlan } from './ee-pipeline.mjs'

/** A job left running longer than this is assumed dead and may be reclaimed. */
export const STALE_LOCK_MS = 15 * 60 * 1000

export class QueueError extends Error {
  constructor(message, { status = 400, code = '' } = {}) {
    super(message)
    this.status = status
    this.code = code
  }
}

function db() {
  const client = adminClient()
  if (!client) throw new QueueError('The job queue needs Supabase to be configured.', { status: 503 })
  return client
}

/**
 * How many points and dates a spec will touch.
 *
 * Needed before the job runs, to price it. For a dataset source the count is
 * known; for a bounding box it has to be estimated from the observations
 * already loaded, which is what `counter` provides.
 */
export async function measureSpec(spec, counter) {
  const measured = await counter(spec)
  return {
    points: Math.max(0, Number(measured?.points) || 0),
    dates: Math.max(1, Number(measured?.dates) || 1),
  }
}

/**
 * Submit a job, or explain why not.
 *
 * The order matters: normalise first so a malformed spec is refused before it
 * costs a database read, then measure, then check the quota, then insert. The
 * quota check reads the profile fresh rather than trusting the token's tier —
 * this is a path that spends money, and a token can be up to an hour stale.
 */
export async function submitJob({ user, profile, spec: rawSpec, counter }) {
  const spec = normaliseSpec(rawSpec)
  const client = db()

  const { points, dates } = await measureSpec(spec, counter)
  if (!points) throw new QueueError('That area and date range contain no observations to enrich.')

  const estimate = estimateUnits({ points, dates, stages: spec.stages }, STAGES)

  // Everything this member has run, for the daily count, the month's spend and
  // how many jobs they already have in flight.
  const { data: history } = await client
    .from('ee_jobs')
    .select('status, created_at, cost_units, estimated_units')
    .eq('user_id', user.id)
    // A calendar month back covers every window the quota rules look at.
    .gte('created_at', new Date(Date.now() - 32 * 86400000).toISOString())

  const usage = summariseUsage(history || [])
  const verdict = checkQuota({ profile, usage, running: usage.running, estimate, points })
  if (!verdict.ok) throw new QueueError(verdict.message, { status: 403, code: verdict.code })

  const { data, error } = await client.from('ee_jobs').insert({
    user_id: user.id,
    kind: spec.kind,
    params: { ...spec, points, dates },
    title: spec.title || null,
    estimated_units: estimate,
    status: 'queued',
  }).select().single()

  if (error) throw new QueueError(`Could not queue that job: ${error.message}`, { status: 500 })
  return { job: data, estimate, points, dates, remaining: verdict.remaining }
}

/**
 * Take the oldest queued job, exclusively.
 *
 * The update is conditional on the row still being queued, and Postgres
 * serialises the two writers, so of two workers reaching for the same row
 * exactly one gets a row back and the other gets nothing and moves on. Doing
 * this as a read followed by a write would let both think they had won.
 *
 * A job whose lock has gone stale is reclaimed the same way: a worker that died
 * mid-job would otherwise leave it running forever.
 */
export async function claimNextJob(workerId) {
  const client = db()
  const staleBefore = new Date(Date.now() - STALE_LOCK_MS).toISOString()

  const { data: candidates } = await client
    .from('ee_jobs')
    .select('id, status, locked_at')
    .or(`status.eq.queued,and(status.eq.running,locked_at.lt.${staleBefore})`)
    .order('created_at', { ascending: true })
    .limit(5)

  for (const candidate of candidates || []) {
    const query = client.from('ee_jobs')
      .update({
        status: 'running',
        locked_by: workerId,
        locked_at: new Date().toISOString(),
        started_at: new Date().toISOString(),
        progress: 0,
        stage: null,
        message: 'Starting…',
      })
      .eq('id', candidate.id)

    // The condition that makes the claim exclusive: still queued, or still
    // holding a lock that has expired.
    const guarded = candidate.status === 'queued'
      ? query.eq('status', 'queued')
      : query.eq('status', 'running').lt('locked_at', staleBefore)

    const { data } = await guarded.select().maybeSingle()
    if (data) return data
    // Lost the race for this one; try the next candidate.
  }
  return null
}

/** Write progress back to the row the member is watching. */
export async function reportProgress(jobId, { fraction, stage, message }) {
  const client = adminClient()
  if (!client) return
  await client.from('ee_jobs').update({
    progress: Math.min(1, Math.max(0, Number(fraction) || 0)),
    stage: stage || null,
    message: message || null,
    // Refreshed on every report so a long stage does not look like a dead
    // worker to the stale-lock sweep above.
    locked_at: new Date().toISOString(),
  }).eq('id', jobId)
}

export async function finishJob(jobId, { resultPath, meta, costUnits }) {
  const client = db()
  const { error } = await client.from('ee_jobs').update({
    status: 'succeeded',
    progress: 1,
    stage: 'done',
    message: null,
    result_path: resultPath,
    result_meta: meta || null,
    cost_units: costUnits ?? 0,
    finished_at: new Date().toISOString(),
    locked_by: null,
  }).eq('id', jobId)
  if (error) throw new QueueError(error.message, { status: 500 })
}

export async function failJob(jobId, error, { costUnits = 0 } = {}) {
  const client = adminClient()
  if (!client) return
  await client.from('ee_jobs').update({
    status: 'failed',
    // What actually went wrong, trimmed: an Earth Engine stack trace in a UI
    // helps nobody, and the first line is nearly always the useful part.
    error: String(error?.message || error).slice(0, 500),
    // Charged for what it managed to spend before failing. Earth Engine billed
    // those requests whether or not the job produced anything.
    cost_units: costUnits,
    finished_at: new Date().toISOString(),
    locked_by: null,
  }).eq('id', jobId)
}

/** Was this job cancelled while it was running? Checked between stages. */
export async function isCancelled(jobId) {
  const client = adminClient()
  if (!client) return false
  const { data } = await client.from('ee_jobs').select('status').eq('id', jobId).maybeSingle()
  return data?.status === 'cancelled'
}

/** A member's jobs, newest first. */
export async function listJobs(userId, { limit = 50, all = false } = {}) {
  const client = db()
  let query = client.from('ee_jobs')
    .select('id, user_id, kind, title, params, status, progress, stage, message, '
      + 'estimated_units, cost_units, result_path, result_meta, error, created_at, started_at, finished_at')
    .order('created_at', { ascending: false })
    .limit(Math.min(200, Math.max(1, limit)))
  if (!all) query = query.eq('user_id', userId)
  const { data, error } = await query
  if (error) throw new QueueError(error.message, { status: 500 })
  return data || []
}

/** The plan a job will follow, for the progress bar and the cost estimate. */
export function planFor(job) {
  const spec = job?.params || {}
  return progressPlan(spec.stages || [], { points: spec.points || 0, dates: spec.dates || 1 })
}
