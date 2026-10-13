// How a running Earth Engine job reads to the person watching it.
//
// The worker stores progress as a fraction (0–1), not a percentage, and
// refreshes locked_at every time it reports. A page that prints the fraction
// as a percent shows a job stuck at "0%" however far along it is, and a job
// whose worker was cut off keeps status 'running' until the stale-lock sweep
// reaches it, so both are worth turning into something a person can read.

/** No report from the worker for this long reads as stalled. */
export const STALLED_AFTER_MS = 3 * 60 * 1000

interface JobLike {
  status?: string | null
  progress?: number | null
  locked_at?: string | null
  started_at?: string | null
}

/** Whole-number percent, 0–100, from the stored fraction. */
export function jobPercent(job: JobLike | null | undefined): number {
  const p = Number(job?.progress)
  if (!Number.isFinite(p) || p <= 0) return 0
  // A value above 1 was already written as a percentage; keep it rather than
  // multiplying it past 100.
  return Math.min(100, Math.round(p > 1 ? p : p * 100))
}

/** A running job whose worker has not reported for STALLED_AFTER_MS. */
export function jobStalled(job: JobLike | null | undefined, now = Date.now()): boolean {
  if (job?.status !== 'running') return false
  const last = Date.parse(job.locked_at || job.started_at || '')
  return Number.isFinite(last) && now - last > STALLED_AFTER_MS
}
