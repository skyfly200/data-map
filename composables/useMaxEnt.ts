// State management and API communication for the MaxEnt Modeling Suite.
// 
// This composable coordinates the training process: 
// 1. Configuration -> 2. Job Submission -> 3. Polling for Results.

import { markRaw } from 'vue'

export interface MaxEntConfig {
  id: string
  title: string
  description?: string
  predictors: string[]
  background_count: number
  effort_weighted: boolean
  projection_region?: any
  source_dataset_id?: string
  visibility: string
  created_at: string
  updated_at: string
}

export interface MaxEntResult {
  id: string
  auc: number
  auc_sd: number
  grade: string
  suitability_asset_path: string
  results_json_path?: string
  created_at: string
}

export interface MaxEntRun {
  id: string
  job_id: string
  status: 'pending' | 'running' | 'succeeded' | 'failed'
  started_at: string
  finished_at?: string
  error_message?: string
  run_meta?: any
  model_configs: MaxEntConfig
}

export function useMaxEnt() {
  const models = useState<MaxEntConfig[]>('maxent-models', () => [])
  const activeJob = useState<MaxEntRun | null>('maxent-active-job', () => null)
  const pending = useState<boolean>('maxent-pending', () => false)
  const error = useState<string>('maxent-error', () => '')

  /** Fetch the user's saved model configurations. */
  async function fetchModels() {
    try {
      const res = await fetch('/.netlify/functions/modeling/maxent/models')
      if (!res.ok) throw new Error(`Failed to fetch models: ${res.statusText}`)
      const data = await res.json()
      if (data.ok) models.value = data.models
    } catch (e: any) {
      error.value = e.message
    }
  }

  /** Submit a new training job. */
  async function trainModel(spec: any) {
    pending.value = true
    error.value = ''
    try {
      const res = await fetch('/.netlify/functions/modeling/maxent/train', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(spec),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Training submission failed')
      
      activeJob.value = {
        id: data.config.id,
        job_id: data.jobId,
        status: 'pending',
        started_at: new Date().toISOString(),
        model_configs: data.config,
      } as MaxEntRun

      // Start polling for results
      pollJobStatus(data.jobId)
      return { ok: true, jobId: data.jobId }
    } catch (e: any) {
      error.value = e.message
      return { ok: false, error: e.message }
    } finally {
      pending.value = false
    }
  }

  /** Poll for job completion and fetch results. */
  async function pollJobStatus(jobId: string) {
    const timer = setInterval(async () => {
      try {
        const res = await fetch(`/.netlify/functions/modeling/maxent/results/${jobId}`)
        const data = await res.json()
        
        if (data.ok && data.result) {
          clearInterval(timer)
          activeJob.value = { ...activeJob.value!, status: 'succeeded' }
          await fetchModels() // Refresh models list
        } else if (data.ok && data.status === 'pending') {
          activeJob.value = { ...activeJob.value!, status: 'running' }
        } else if (!data.ok) {
          clearInterval(timer)
          activeJob.value = { ...activeJob.value!, status: 'failed', error_message: data.error }
        }
      } catch (e: any) {
        console.error('Polling error:', e)
      }
    }, 5000)
  }

  /** Delete a saved model configuration. */
  async function deleteModel(id: string) {
    try {
      const res = await fetch('/.netlify/functions/modeling/maxent/models', {
        method: 'DELETE',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ id }),
      })
      const data = await res.json()
      if (!data.ok) throw new Error(data.error || 'Delete failed')
      models.value = models.value.filter(m => m.id !== id)
      return { ok: true }
    } catch (e: any) {
      error.value = e.message
      return { ok: false, error: e.message }
    }
  }

  return {
    models, activeJob, pending, error,
    fetchModels, trainModel, deleteModel,
  }
}
