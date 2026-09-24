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

export interface EeJob {
  id: string
  user_id: string
  kind: string
  title: string
  params: any
  status: string
  progress: number
  stage: string
  message: string
  estimated_units: number
  cost_units: number
  result_path: string | null
  result_meta: any
  error: string | null
  created_at: string
  started_at: string | null
  finished_at: string | null
  archived_at: string | null
}

function messageFrom(data: any, status: number): string {
  return data?.error || data?.message || `Earth Engine jobs error (${status})`
}

export function useEeJobs() {
  const { $supabase } = useNuxtApp()
  const { accessToken } = useAuth()

  const jobs = useState<EeJob[]>('ee-jobs', () => [])
  const loading = useState('ee-jobs-loading', () => false)
  const error = useState('ee-jobs-error', () => '')
  const submitting = useState('ee-jobs-submitting', () => false)

  const active = computed(() => jobs.value.filter((j) => !SETTLED.has(j.status)))

  const FIELDS = 'id, user_id, kind, title, params, status, progress, stage, message, '
    + 'estimated_units, cost_units, result_path, result_meta, error, '
    + 'created_at, started_at, finished_at, archived_at'

  async function refresh({ all = false } = {}): Promise<EeJob[]> {
    if (!$supabase) return []
    loading.value = true
    error.value = ''
    try {
      let query = $supabase.from('ee_jobs')
        .select(FIELDS)
        .is('archived_at', null)
        .order('created_at', { ascending: false })
        .limit(50)
      if (!all) {
        const { data: session } = await $supabase.auth.getSession()
        const uid = session.session?.user?.id
        if (uid) query = query.eq('user_id', uid)
      }
      const { data, error: err } = await query
      if (err) throw err
      jobs.value = data || []
      return jobs.value
    } catch (e: any) {
      error.value = /schema cache|does not exist/i.test(e?.message || '')
        ? 'The job tables are missing. Run supabase_migrations/002_membership_jobs_and_admin.sql.'
        : (e?.message || 'Could not read your jobs.')
      return []
    } finally {
      loading.value = false
    }
  }

  async function refreshArchive(): Promise<EeJob[]> {
    if (!$supabase) return []
    const { data: session } = await $supabase.auth.getSession()
    const uid = session.session?.user?.id
    if (!uid) return []
    const { data } = await $supabase.from('ee_jobs')
      .select(FIELDS)
      .not('archived_at', 'is', null)
      .eq('user_id', uid)
      .order('archived_at', { ascending: false })
      .limit(100)
    return data || []
  }

  async function submit(spec: any) {
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
    } catch (e: any) {
      // Don't set shared error.value here — that state drives the "Your jobs"
      // section. The caller (jobs.vue) handles submit errors with its own ref.
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
  async function cancel(id: string) {
    if (!$supabase) return
    const { error: err } = await $supabase.from('ee_jobs')
      .update({ status: 'cancelled' }).eq('id', id)
    if (err) error.value = err.message
    await refresh()
  }

  /**
   * The finished output, as GeoJSON, ready to drop onto the map.
   *
   * Downloaded straight from storage with the member's own session, which the
   * policy in migration 005 scopes to their own jobs/<uid>/ prefix. That covers
   * their own results and nothing else — a dataset somebody shared with them is
   * read through /.netlify/functions/datasets instead, because sharing lives in
   * the row and a path-prefix policy cannot see it.
   */
  async function fetchResult(job: EeJob): Promise<any | null> {
    if (!job?.result_path || !$supabase) return null
    const bucket = useRuntimeConfig().public.datasetsBucket || 'datasets'
    const { data, error: err } = await $supabase.storage.from(bucket).download(job.result_path)
    if (err || !data) throw new Error('That result could not be read. It may have been cleaned up.')
    return JSON.parse(await data.text())
  }

  /**
   * A fresh suitability template for a stored model.
   *
   * The template on a model's result_meta carries an Earth Engine map id that
   * expires; this re-mints it from the durably-stored model (cached server-side),
   * so viewing a model's map always works rather than drawing blank tiles.
   */
  async function modelTiles(job: EeJob): Promise<{ template: string; meta: any } | null> {
    const token = await accessToken()
    const res = await fetch(`/.netlify/functions/model-tiles?job=${encodeURIComponent(job.id)}`, {
      headers: token ? { authorization: `Bearer ${token}` } : {},
    })
    const body = await res.json()
    if (!res.ok || !body.ok) throw new Error(body.error || 'That model surface could not be re-served.')
    return { template: body.template, meta: body.meta }
  }

  async function patch(body: Record<string, unknown>) {
    const token = await accessToken()
    const res = await fetch('/.netlify/functions/ee-jobs', {
      method: 'PATCH',
      headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify(body),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(data.error || `Archive action failed (${res.status})`)
    return data
  }

  async function archive(id: string) {
    await patch({ action: 'archive', id })
    jobs.value = jobs.value.filter(j => j.id !== id)
  }

  async function unarchive(id: string) {
    await patch({ action: 'unarchive', id })
  }

  async function archiveAll() {
    await patch({ action: 'archive_all' })
    jobs.value = jobs.value.filter(j => j.status === 'queued' || j.status === 'running')
  }

  // Poll only while something is actually moving.
  let timer: ReturnType<typeof setInterval> | null = null
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
    refresh, refreshArchive, submit, cancel, fetchResult, modelTiles,
    archive, unarchive, archiveAll,
    startPolling, stopPolling,
  }
}
