// Submitting Earth Engine jobs and watching them run.
//
// A job outlives the request that started it, so this is a poll rather than a
// stream: the server writes progress into the job row and the browser reads it
// back. Polling stops as soon as nothing is in flight, so a page left open on a
// finished job costs nothing.
//
// Reads go through Supabase directly, where row-level security already limits a
// member to their own rows. Writes go through the function, which is where the
// quota is checked and the spec is validated.

const POLL_MS = 3000

/** Statuses that will not change on their own. */
const SETTLED = new Set(['succeeded', 'failed', 'cancelled'])

export function useEeJobs() {
  const { $supabase } = useNuxtApp()
  const { accessToken } = useAuth()

  const jobs = useState('ee-jobs', () => [])
  const loading = useState('ee-jobs-loading', () => false)
  const error = useState('ee-jobs-error', () => '')
  const submitting = useState('ee-jobs-submitting', () => false)

  const active = computed(() => jobs.value.filter((j) => !SETTLED.has(j.status)))

  async function refresh({ all = false } = {}) {
    if (!$supabase) return []
    loading.value = true
    error.value = ''
    try {
      let query = $supabase.from('ee_jobs')
        .select('id, user_id, kind, title, params, status, progress, stage, message, '
          + 'estimated_units, cost_units, result_path, result_meta, error, '
          + 'created_at, started_at, finished_at')
        .order('created_at', { ascending: false })
        .limit(50)
      // RLS already narrows this to the caller's rows unless they are an admin,
      // so `all` is about what an admin asks for rather than about permission.
      if (!all) {
        const { data: session } = await $supabase.auth.getSession()
        const uid = session.session?.user?.id
        if (uid) query = query.eq('user_id', uid)
      }
      const { data, error: err } = await query
      if (err) throw err
      jobs.value = data || []
      return jobs.value
    } catch (e) {
      error.value = /schema cache|does not exist/i.test(e?.message || '')
        ? 'The job tables are missing. Run supabase_migrations/002_membership_jobs_and_admin.sql.'
        : (e?.message || 'Could not read your jobs.')
      return []
    } finally {
      loading.value = false
    }
  }

  async function submit(spec) {
    submitting.value = true
    error.value = ''
    try {
      const token = await accessToken()
      const res = await fetch('/.netlify/functions/ee-jobs', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          ...(token ? { authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify(spec),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok || !data.ok) throw new Error(messageFrom(data, res.status))
      await refresh()
      return data
    } catch (e) {
      // The function's messages are written for the member — a quota that ran
      // out, an area with nothing in it — so they are shown as they arrive.
      error.value = e.message || 'Could not submit that job.'
      throw e
    } finally {
      submitting.value = false
    }
  }

  /**
   * Cancel a job of your own.
   *
   * Written straight to the row: the RLS policy allows exactly this transition
   * and nothing else, and the worker checks the status between stages. A job
   * mid-stage finishes that stage first, which is why this says "cancelling".
   */
  async function cancel(id) {
    if (!$supabase) return
    const { error: err } = await $supabase.from('ee_jobs')
      .update({ status: 'cancelled' }).eq('id', id)
    if (err) error.value = err.message
    await refresh()
  }

  /** The finished output, as GeoJSON, ready to drop onto the map. */
  async function fetchResult(job) {
    if (!job?.result_path || !$supabase) return null
    const bucket = 'datasets'
    const { data, error: err } = await $supabase.storage.from(bucket).download(job.result_path)
    if (err || !data) throw new Error('That result could not be read. It may have been cleaned up.')
    return JSON.parse(await data.text())
  }

  // Poll only while something is actually moving.
  let timer = null
  function stopPolling() { if (timer) { clearInterval(timer); timer = null } }
  function startPolling() {
    if (timer || !import.meta.client) return
    timer = setInterval(async () => {
      await refresh()
      if (!active.value.length) stopPolling()
    }, POLL_MS)
  }

  // Anything in flight keeps the poll alive; a page opened on settled jobs
  // never starts one.
  watch(active, (list) => {
    if (list.length) startPolling()
    else stopPolling()
  })
  if (import.meta.client) onScopeDispose(stopPolling)

  return {
    jobs, active, loading, error, submitting,
    refresh, submit, cancel, fetchResult, startPolling, stopPolling,
  }
}
