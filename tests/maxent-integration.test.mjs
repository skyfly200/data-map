import test from 'node:test'
import assert from 'node:assert/strict'
import { 
  normaliseModelSpec, 
  estimateModelUnits 
} from '../netlify/lib/maxent.mjs'

/**
 * Integration tests for the MaxEnt pipeline logic.
 * Since the handler in modeling-maxent.mjs is a default export that encapsulates
 * its dependencies, we test the logic by simulating the helper functions 
 * and the overall flow.
 */

// ── Mocks ─────────────────────────────────────────────────────────────────────

const mockUser = { userId: 'user-123', role: 'member' }
const mockViewer = { userId: 'user-123' }

const createMockClient = () => {
  const state = {
    model_configs: [],
    model_runs: [],
    model_results: [],
  }

  const from = (table) => ({
    select: (cols) => ({
      eq: (col, val) => {
        const rows = state[table].filter(r => r[col] === val)
        return {
          maybeSingle: async () => ({ data: rows[0] || null, error: null }),
          execute: async () => ({ data: rows, error: null }),
          order: (col2, { ascending }) => ({
            eq: (col3, val3) => ({
              maybeSingle: async () => {
                const filtered = rows.filter(r => r[col3] === val3)
                return { data: filtered[0] || null, error: null }
              }
            })
          })
        }
      },
      order: (col, { ascending }) => ({
        eq: (col2, val) => ({
          maybeSingle: async () => {
            const rows = state[table].filter(r => r[col2] === val)
            return { data: rows[0] || null, error: null }
          }
        })
      })
    }),
    insert: (data) => {
      const id = Math.random().toString(36).substr(2, 9)
      const row = { ...data, id }
      state[table].push(row)
      return { 
        data: row, 
        error: null, 
        select: () => ({
          single: async () => ({ data: row, error: null })
        })
      }
    },
    delete: () => ({
      eq: (col, val) => ({
        execute: async () => {
          const idx = state[table].findIndex(r => r[col] === val)
          if (idx === -1) return { error: new Error('Not found') }
          state[table].splice(idx, 1)
          return { error: null }
        }
      })
    })
  })

  return { from, state }
}

const mockEeRunner = {
  submitMaxEntJob: async (spec) => ({ id: 'job-abc-123', status: 'running' })
}

// ── Pipeline Logic Tests ─────────────────────────────────────────────────────

test('MaxEnt Pipeline Integration', async (t) => {
  const client = createMockClient()

  await t.test('Training flow: Spec -> Config -> Job -> Run', async () => {
    const body = {
      title: 'Test Model',
      region: { north: 40, south: 39, east: -105, west: -106 },
      predictors: ['elevation', 'slope', 'ndvi'],
      background: 1000,
      visibility: 'private',
      source_dataset_id: 'ds-123'
    }

    // 1. Validate Spec
    const spec = normaliseModelSpec(body)
    assert.strictEqual(spec.background, 1000)
    assert.ok(spec.predictors.includes('elevation'))

    // 2. Create Config
    const { data: config } = await client.from('model_configs').insert({
      owner_id: mockUser.userId,
      title: spec.title,
      predictors: spec.predictors,
      background_count: spec.background,
      projection_region: spec.region,
      source_dataset_id: body.source_dataset_id,
      visibility: 'private'
    }).select().single()
    
    assert.ok(config.id)
    assert.strictEqual(config.owner_id, 'user-123')

    // 3. Trigger Job
    const job = await mockEeRunner.submitMaxEntJob(spec)
    assert.strictEqual(job.id, 'job-abc-123')

    // 4. Record Run
    const { error: runErr } = await client.from('model_runs').insert({
      config_id: config.id,
      job_id: job.id,
      status: 'pending'
    }).select().single()
    
    assert.strictEqual(runErr, null)

    // Verify state
    assert.strictEqual(client.state.model_configs.length, 1)
    assert.strictEqual(client.state.model_runs.length, 1)
  })

  await t.test('Results flow: JobID -> Run -> Result', async () => {
    const jobId = 'job-abc-123'
    
    // Setup: find the run record first
    const { data: run } = await client.from('model_runs')
      .select().eq('job_id', jobId).maybeSingle()
    
    assert.ok(run, 'Run should exist in state')

    await client.from('model_results').insert({
      run_id: run.id,
      auc: 0.85,
      metrics: { folds: 5, mean: 0.82, sd: 0.05 }
    })

    // Test retrieval
    const { data: foundRun } = await client.from('model_runs')
      .select('*, model_configs(*)').eq('job_id', jobId).maybeSingle()

    assert.ok(foundRun)
    const { data: config } = await client.from('model_configs')
      .select().eq('id', foundRun.config_id).maybeSingle()
    
    assert.ok(config)
    assert.strictEqual(config.owner_id, 'user-123')
    
    const { data: result } = await client.from('model_results')
      .select('*').eq('run_id', foundRun.id).maybeSingle()

    assert.ok(result, 'Result should be found for the run ID')
    assert.strictEqual(result.auc, 0.85)
  })
  await t.test('Management flow: List and Delete', async () => {
    // List
    const { data: models } = await client.from('model_configs')
      .select('*, model_results(*)').eq('owner_id', mockUser.userId).execute()

    assert.strictEqual(models.length, 1)
    // Delete
    const targetId = models[0].id
    const { error: delErr } = await client.from('model_configs').delete().eq('id', targetId).execute()
    assert.strictEqual(delErr, null)
    
    assert.strictEqual(client.state.model_configs.length, 0)
  })
})
