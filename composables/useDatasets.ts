// A member's saved datasets: naming a job result so another job can use it.
//
// Thin wrapper over /.netlify/functions/datasets. Everything here goes through
// the function rather than through PostgREST, because a dataset row carries the
// storage path of its file and is therefore created server-side only — see the
// reasoning in supabase_migrations/005_dataset_storage_access.sql.

import { MEMBER_VISIBILITIES } from '~/netlify/lib/dataset-access.mjs'

export interface Dataset {
  id: string
  slug: string
  title: string
  description: string
  visibility: string
  created_at: string
}

export const VISIBILITY_LABELS: Record<string, string> = {
  private: 'Only me',
  members: 'FRMS members',
  public: 'Anyone',
}

export function useDatasets() {
  const { accessToken } = useAuth()

  // Shared, so the jobs page's source picker and its dataset list are looking
  // at the same rows rather than each fetching their own copy.
  const datasets = useState<Dataset[]>('datasets-mine', () => [])
  const available = useState<Dataset[]>('datasets-available', () => [])
  const loading = useState<boolean>('datasets-loading', () => false)
  const error = useState<string>('datasets-error', () => '')

  async function call<T>(path: string, options: RequestInit = {}): Promise<T> {
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
    return data as T
  }

  /** The caller's own datasets. */
  async function refresh(): Promise<Dataset[]> {
    loading.value = true
    error.value = ''
    try {
      const data = await call<{ datasets: Dataset[] }>('')
      datasets.value = data.datasets || []
    } catch (e: any) {
      error.value = e.message
      datasets.value = []
    } finally {
      loading.value = false
    }
    return datasets.value
  }

  /** Everything they may run a job over: their own, plus what is shared. */
  async function refreshAvailable(): Promise<Dataset[]> {
    try {
      const data = await call<{ datasets: Dataset[] }>('?available=1')
      available.value = data.datasets || []
    } catch (e: any) {
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
  async function saveJob(job: { id: string, title?: string }, { title, description = '', visibility }: { title?: string, description?: string, visibility?: string } = {}) {
    error.value = ''
    const data = await call<{ dataset: Dataset }>('', {
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

  async function update(id: string, patch: Partial<Dataset>) {
    error.value = ''
    const data = await call<{ dataset: Dataset }>('', {
      method: 'POST',
      body: JSON.stringify({ action: 'update', id, ...patch }),
    })
    await refresh()
    return data.dataset
  }

  async function remove(id: string) {
    error.value = ''
    await call<{ ok: boolean }>('', { method: 'POST', body: JSON.stringify({ action: 'delete', id }) })
    await refresh()
  }

  /** The GeoJSON behind one, by slug. Works for shared datasets too. */
  async function fetchGeojson(slug: string): Promise<any> {
    const data = await call<{ geojson: any }>(`?slug=${encodeURIComponent(slug)}`)
    return data.geojson
  }

  /** Import an Earth Engine asset by its asset path (e.g., "users/username/project/dataset") */
  async function importAsset(assetPath: string, { title = '', description = '', visibility = 'private' }: { title?: string, description?: string, visibility?: string } = {}) {
    error.value = ''
    if (!assetPath || !assetPath.trim()) {
      throw new Error('Please provide an Earth Engine asset path')
    }
    const data = await call<{ dataset: Dataset }>('', {
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
