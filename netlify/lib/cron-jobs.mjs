// Static catalogue of scheduled Netlify functions.
// Used by the admin panel to show schedules and recent logs alongside each job.

export const CRON_JOBS = [
  {
    id: 'refresh-observations',
    name: 'Observation refresh',
    schedule: '0 */6 * * *',
    description: 'Fetches new iNaturalist sightings and merges them into the observations dataset in Supabase Storage (or Netlify Blob Store when Supabase is not configured).',
  },
  {
    id: 'ee-worker',
    name: 'Earth Engine worker',
    schedule: '* * * * *',
    description: 'Claims and runs one queued Earth Engine enrichment or modeling job per invocation. Logs only when a job is claimed.',
  },
]

/**
 * Write one log row to cron_logs via the Supabase client.
 * Best-effort: a log failure never throws — the function result matters more.
 */
export async function logCronRun(client, { jobId, status, durationMs, details = {} }) {
  if (!client) return
  try {
    await client.from('cron_logs').insert({
      job_id: jobId,
      fired_at: new Date().toISOString(),
      duration_ms: durationMs ?? null,
      status,
      details,
    })
  } catch {
    // intentionally swallowed
  }
}
