// Whether a member may run the job they just asked for.
//
// Earth Engine bills the project, not the caller. Every member's query runs
// under one service account against one pool of quota, so without a limit here
// a single large job degrades the platform for the whole society. These are the
// limits an admin sets per member on the profiles row.
//
// All pure, so the arithmetic that decides whether someone is over budget can
// be tested without a database or a live Earth Engine session.

import { effectiveTier } from './tiers.mjs'

/** What a new profile gets. Mirrors the defaults in migration 002. */
export const DEFAULT_LIMITS = {
  ee_quota_monthly: 500,
  ee_jobs_per_day: 20,
  ee_max_points: 5000,
  ee_max_concurrent: 1,
}

/** Points per Earth Engine request; matches CHUNK_SIZE in scripts/ee_enrich.py. */
export const CHUNK_SIZE = 500

/**
 * Roughly what a job will cost, in units of "one Earth Engine request".
 *
 * The pattern the Python pipeline established is one reduceRegions per chunk of
 * points per query group, so cost scales with points and with how many distinct
 * images a stage has to touch: a static layer like elevation is one pass, while
 * a seven-day rainfall history is one per day.
 *
 * It is an estimate and it is meant to be a slight over-count. Refusing a job
 * that would have just fitted is a smaller harm than admitting one that blows
 * the month's quota in a single run.
 */
export function estimateUnits({ points = 0, stages = [], dates = 1 } = {}, catalogue = {}) {
  const chunks = Math.max(1, Math.ceil(points / CHUNK_SIZE))
  const days = Math.max(1, dates)
  let total = 0
  for (const key of stages) {
    const stage = catalogue[key]
    if (!stage) continue
    // A stage sampling an image that moves with the record's date cannot batch
    // across dates: it costs at least one request per distinct date as well as
    // one per chunk of points. A static layer like elevation pays neither.
    const groups = stage.perDate ? Math.max(chunks, days) : chunks
    total += groups * (stage.passes || 1)
  }
  return total
}

/**
 * May this job run?
 *
 * Returns `{ ok }` or `{ ok: false, code, message }`, where the message is
 * written to be shown to the member: someone who has hit a limit needs to know
 * which one and when it lifts, not that a request failed.
 */
export function checkQuota({ profile, usage = {}, running = 0, estimate = 0, points = 0, now = new Date() } = {}) {
  const tier = effectiveTier(profile, now)
  if (tier === 'free') {
    return {
      ok: false,
      code: 'not_a_member',
      message: profile?.member_until
        ? 'Your membership has lapsed. Renew it to run pipeline jobs.'
        : 'Running pipeline jobs is a membership benefit.',
    }
  }

  const limits = { ...DEFAULT_LIMITS, ...(profile || {}) }

  // Admins are still metered — an admin running a runaway job spends the same
  // quota as anyone else — but they are not held to the per-member ceilings,
  // which exist to divide a shared pool fairly rather than to protect it.
  const enforced = tier !== 'admin'

  if (enforced && points > limits.ee_max_points) {
    return {
      ok: false,
      code: 'too_many_points',
      message: `That job covers ${points.toLocaleString()} points and your limit is `
        + `${limits.ee_max_points.toLocaleString()}. Narrow the area or the date range.`,
    }
  }

  if (enforced && running >= limits.ee_max_concurrent) {
    return {
      ok: false,
      code: 'already_running',
      message: limits.ee_max_concurrent === 1
        ? 'You already have a job running. It will need to finish first.'
        : `You already have ${running} jobs running, which is your limit.`,
    }
  }

  if (enforced && (usage.jobsToday || 0) >= limits.ee_jobs_per_day) {
    return {
      ok: false,
      code: 'daily_limit',
      message: `You have started ${usage.jobsToday} jobs today, which is your daily limit. `
        + 'It resets at midnight UTC.',
    }
  }

  const spent = usage.unitsThisMonth || 0
  if (spent + estimate > limits.ee_quota_monthly) {
    const left = Math.max(0, limits.ee_quota_monthly - spent)
    return {
      ok: false,
      code: 'over_quota',
      message: `That job needs about ${estimate} units and you have ${left} left this month `
        + `of ${limits.ee_quota_monthly}. An admin can raise your quota.`,
    }
  }

  return { ok: true, tier, estimate, remaining: limits.ee_quota_monthly - spent - estimate }
}

/** Usage rolled up the way checkQuota wants it, from a member's job rows. */
export function summariseUsage(jobs = [], now = new Date()) {
  const monthStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), 1))
  const dayStart = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()))
  let unitsThisMonth = 0
  let jobsToday = 0
  let running = 0

  for (const job of jobs) {
    const at = new Date(job.created_at)
    if (!Number.isFinite(at.getTime())) continue
    // A cancelled or failed job spent something but is not held against the
    // daily count: a member should not be locked out for the day by a pipeline
    // that broke on them.
    const counted = job.status === 'running' || job.status === 'succeeded'
    if (counted && at >= monthStart) unitsThisMonth += job.cost_units || job.estimated_units || 0
    if (counted && at >= dayStart) jobsToday += 1
    if (job.status === 'running' || job.status === 'queued') running += 1
  }

  return { unitsThisMonth, jobsToday, running }
}
