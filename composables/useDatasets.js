// A member's saved datasets: naming a job result so another job can use it.
//
// Thin wrapper over /.netlify/functions/datasets. Everything here goes through
// the function rather than through PostgREST, because a dataset row carries the
// storage path of its file and is therefore created server-side only — see the
// reasoning in supabase_migrations/005_dataset_storage_access.sql.

import { MEMBER_VISIBILITIES } from '~/netlify/lib/dataset-access.mjs'

export const VISIBILITY_LABELS = {
  private: 'Only me',
  members: 'FRMS members',
  public: 'Anyone',
}

export function useDatasets() {
  const { accessToken } = useAuth()

  // Shared, so the jobs page's source picker and its dataset list are looking
  // at the same rows rather than each fetching their own copy.
  const datasets = useState('datasets-mine', () => [])
  const available = useState('datasets-available', () => [])
  const loading = useState('datasets-loading', () => false)
  const error = useState('datasets-error', () => '')

  async function call(path, options = {}) {
    const token = await accessToken()
    const res = await fetch(`/.netlify/functions/datasets${path}`, {
      ...options,
      headers: {
        'content-type': 'application/json',
        ...(token ? { authorization: `Bearer ${token}` } : {}),
        ...(options.headers || {}),
      },
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `That request failed (${res.status}).`)
    }
    return data
  }

  /** The caller's own datasets. */
  async function refresh() {
    loading.value = true
    error.value = ''
    try {
      datasets.value = (await call('')).datasets || []
    } catch (e) {
      error.value = e.message
      datasets.value = []
    } finally {
      loading.value = false
    }
    return datasets.value
  }

  /** Everything they may run a job over: their own, plus what is shared. */
  async function refreshAvailable() {
    try {
      available.value = (await call('?available=1')).datasets || []
    } catch (e) {
      error.value = e.message
      available.value = []
    }
    return available.value
  }

  /**
   * Name a finished job, so other jobs can reference it.
   *
   * Private unless told otherwise — the server decides that, not this, so the
   * default holds for every caller rather than for this one.
   */
  async function saveJob(job, { title, description = '', visibility } = {}) {
    error.value = ''
    const data = await call('', {
      method: 'POST',
      body: JSON.stringify({
        action: 'save',
        job_id: job.id,
        title: title || job.title || '',
        description,
        ...(visibility ? { visibility } : {}),
      }),
    })
    await refresh()
    return data.dataset
  }

  async function update(id, patch) {
    error.value = ''
    const data = await call('', {
      method: 'POST',
      body: JSON.stringify({ action: 'update', id, ...patch }),
    })
    await refresh()
    return data.dataset
  }

  async function remove(id) {
    error.value = ''
    await call('', { method: 'POST', body: JSON.stringify({ action: 'delete', id }) })
    await refresh()
  }

  /** The GeoJSON behind one, by slug. Works for shared datasets too. */
  async function fetchGeojson(slug) {
    return (await call(`?slug=${encodeURIComponent(slug)}`)).geojson
  }

  /** Import an Earth Engine asset by its asset path (e.g., "users/username/project/dataset") */
  async function importAsset(assetPath, { title = '', description = '', visibility = 'private' } = {}) {
    error.value = ''
    if (!assetPath || !assetPath.trim()) {
      throw new Error('Please provide an Earth Engine asset path')
    }
    const data = await call('', {
      method: 'POST',
      body: JSON.stringify({
        action: 'import_asset',
        asset_path: assetPath.trim(),
        title: title || assetPath.split('/').pop() || 'EE Asset',
        description,
        visibility,
      }),
    })
    await refresh()
    return data.dataset
  }

  return {
    datasets, available, loading, error,
    visibilities: MEMBER_VISIBILITIES,
    refresh, refreshAvailable, saveJob, update, remove, fetchGeojson, importAsset,
  }
}
