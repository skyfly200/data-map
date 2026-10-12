// Write-back of a finished model job's contributions onto model_results
// (WANT-17 blocker). Best-effort: a job with no model_runs row (an ad-hoc run
// that was never saved as a config) or with neither contributions nor predictor
// ranges is simply skipped.

/** Pick the persistable fields out of a model job's result meta. */
export function contributionPayload(meta) {
  const filled = (v) => v && typeof v === 'object' && Object.keys(v).length > 0
  const c = filled(meta?.contributions) ? meta.contributions : null
  const r = filled(meta?.predictorRanges) ? meta.predictorRanges : null
  if (!c && !r) return null
  return { contributions: c, predictor_ranges: r }
}

export async function persistModelContributions(client, jobId, meta) {
  const payload = contributionPayload(meta)
  if (!payload) return { written: false, reason: 'no contributions or ranges' }
  const { data: run, error } = await client.from('model_runs').select('id').eq('job_id', String(jobId)).maybeSingle()
  if (error) return { written: false, reason: error.message }
  if (!run) return { written: false, reason: 'no model_runs row' }
  const { error: upErr } = await client.from('model_results').update(payload).eq('run_id', run.id)
  return upErr ? { written: false, reason: upErr.message } : { written: true }
}
