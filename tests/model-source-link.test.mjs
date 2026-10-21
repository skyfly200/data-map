// A model names its dataset by slug; model_configs stores the row's uuid.
// Storing the slug was the "invalid input syntax for type uuid" failure.

import test from 'node:test'
import assert from 'node:assert/strict'

import { linkedDataset } from '../netlify/functions/modeling-maxent.mjs'

const ID = '3f2b6c1e-8a4d-4c2e-9b7a-1d2e3f4a5b6c'
const owner = { userId: 'user-1', tier: 'member' }
const row = { id: ID, slug: 'pipeline-enrichment-3', owner_id: 'user-1', visibility: 'private', path: 'p' }

function client() {
  const calls = []
  return {
    calls,
    from: () => ({
      select: () => ({
        eq: (col, val) => {
          calls.push([col, val])
          return { maybeSingle: async () => ({ data: row, error: null }) }
        },
      }),
    }),
  }
}

test('a dataset source given by slug resolves to the row, looked up by slug', async () => {
  const c = client()
  const got = await linkedDataset(c, owner, { source: { type: 'dataset', slug: 'pipeline-enrichment-3' } })
  assert.equal(got.id, ID)
  assert.deepEqual(c.calls, [['slug', 'pipeline-enrichment-3']])
})

test('an older client sending a slug as source_dataset_id still works', async () => {
  const c = client()
  const got = await linkedDataset(c, owner, { source_dataset_id: 'pipeline-enrichment-3' })
  assert.equal(got.id, ID)
  assert.deepEqual(c.calls, [['slug', 'pipeline-enrichment-3']])
})

test('a real id is looked up by id', async () => {
  const c = client()
  await linkedDataset(c, owner, { source_dataset_id: ID })
  assert.deepEqual(c.calls, [['id', ID]])
})

test('an area source links no dataset', async () => {
  const c = client()
  assert.equal(await linkedDataset(c, owner, { source: { type: 'bbox', bounds: {} } }), null)
  assert.equal(c.calls.length, 0)
})
