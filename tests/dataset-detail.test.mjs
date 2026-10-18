/* What the dataset detail view is told about a dataset.
 *
 * The parts worth pinning: provenance is a whitelist, so nothing a request
 * sends lands in the row unless it is named here; a parent the viewer may not
 * read is hidden rather than named; and the column summary files each band
 * under the stage that produced it.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  buildDetail, cleanProvenance, fetchProvenance, inferOrigin, jobSettings, stageCatalog,
} from '../netlify/lib/dataset-detail.mjs'
import { columnSummary } from '../netlify/lib/dataset-columns.mjs'

const OWNER = 'user-1'
const asOwner = { userId: OWNER, tier: 'member' }
const asOther = { userId: 'user-2', tier: 'member' }

test('provenance keeps only named fields, cleaned', () => {
  const p = cleanProvenance({
    origin: 'inat', taxon: 'Boletus', date_from: '2024-05-01T00:00', lat: '40.1',
    radius_km: 'far', path: 'species/../secret', owner_id: 'x',
  })
  assert.deepEqual(p, { origin: 'inat', taxon: 'Boletus', date_from: '2024-05-01', lat: 40.1 })
})

test('provenance without a known origin is not stored', () => {
  assert.equal(cleanProvenance({ origin: 'somewhere', taxon: 'x' }), null)
  assert.equal(cleanProvenance('inat'), null)
  assert.equal(cleanProvenance([{ origin: 'inat' }]), null)
})

test('a fetch records the query it ran', () => {
  const now = new Date('2026-10-18T00:00:00Z')
  const p = fetchProvenance({
    origin: 'gbif', taxon: 'Boletus rubriceps', d1: '2020-01-01', d2: null,
    lat: 40, lng: -105, radius: 500, quality: 'human observation', records: 313, now,
  })
  assert.deepEqual(p, {
    origin: 'gbif', taxon: 'Boletus rubriceps', date_from: '2020-01-01', lat: 40, lng: -105,
    radius_km: 500, quality: 'human observation', records: 313, fetched_at: now.toISOString(),
  })
})

test('origin is inferred from the path for rows saved before provenance', () => {
  assert.equal(inferOrigin({ job_id: 'j' }), 'job')
  assert.equal(inferOrigin({ path: 'species/gbif-boletus.geojson' }), 'gbif')
  assert.equal(inferOrigin({ path: 'species/u1/boletus-123.geojson' }), 'inat')
  assert.equal(inferOrigin({ path: 'species/u1/gbif-boletus-123.geojson' }), 'gbif')
  assert.equal(inferOrigin({ path: 'datasets/u/x.geojson', description: 'DwC-A import — 3 occurrences' }), 'gbif_archive')
  assert.equal(inferOrigin({ path: 'datasets/u/x.geojson', description: 'Imported from Earth Engine asset: a/b' }), 'ee_asset')
  assert.equal(inferOrigin({ path: 'datasets/u/x.geojson' }), 'upload')
})

test('job settings drop the worker bookkeeping', () => {
  const s = jobSettings({ kind: 'enrich', stages: ['terrain'], attempts: 2, checkpoint_path: 'x', continuations: 1 })
  assert.deepEqual(s, { kind: 'enrich', stages: ['terrain'] })
})

const row = (extra = {}) => ({
  id: 'd1', owner_id: OWNER, slug: 'enriched', title: 'Enriched', path: 'jobs/u/x.geojson',
  visibility: 'members', job_id: 'j1', provenance: null, ...extra,
})
const job = {
  id: 'j1', kind: 'enrich', status: 'succeeded', created_at: 't',
  params: { kind: 'enrich', source: { type: 'dataset', slug: 'raw' }, stages: ['terrain'], attempts: 1 },
  result_meta: { bands: ['elevation'], skipped: {} },
}

test('detail names a parent the viewer can read', () => {
  const parent = { slug: 'raw', title: 'Raw fetch', owner_id: OWNER, visibility: 'private' }
  const d = buildDetail({ row: row(), job, parent, viewer: asOwner })
  assert.deepEqual(d.from, { slug: 'raw', title: 'Raw fetch' })
  assert.equal(d.origin, 'job')
  assert.equal(d.job.settings.attempts, undefined)
  assert.equal(d.dataset.path, undefined, 'the storage path stays on the server')
  assert.equal(d.dataset.mine, true)
})

test('detail hides a parent the viewer cannot read', () => {
  const parent = { slug: 'raw', title: 'Owner only', owner_id: OWNER, visibility: 'private' }
  const d = buildDetail({ row: row(), job, parent, viewer: asOther })
  assert.deepEqual(d.from, { hidden: true })
  assert.equal(d.dataset.mine, false)
})

test('enrichment runs link the dataset each was saved as', () => {
  const run = { ...job, id: 'j2', params: { ...job.params, source: { type: 'dataset', slug: 'enriched' } } }
  const derived = [{ job_id: 'j2', slug: 'enriched-2', title: 'Again', owner_id: OWNER, visibility: 'private' }]
  const d = buildDetail({ row: row({ job_id: null, path: 'species/x.geojson' }), enrichments: [run], derived, viewer: asOwner })
  assert.deepEqual(d.enrichments[0].saved_as, { slug: 'enriched-2', title: 'Again' })
  assert.equal(d.origin, 'inat')
  assert.equal(d.inferred, true)
})

test('columns are grouped by the stage that produced them', () => {
  const stages = stageCatalog()
  const features = [
    { properties: { species: 'a', elevation: 10, slope: 1, prcp_d1: 0, prcp_d0: 2, images: [] } },
    { properties: { species: 'b', elevation: null, slope: 2, prcp_d0: 1, prcp_d1: '' } },
  ]
  const { total, groups } = columnSummary(features, stages)
  assert.equal(total, 2)
  const keys = groups.map((g) => g.key)
  assert.deepEqual(keys, ['terrain', 'precip', 'observation'])
  const terrain = groups[0].columns
  assert.deepEqual(terrain.map((c) => [c.name, c.present]), [['elevation', 1], ['slope', 2]])
  assert.deepEqual(groups[1].columns.map((c) => c.name), ['prcp_d0', 'prcp_d1'], 'bands in catalogue order')
  assert.deepEqual(groups[2].columns.map((c) => c.name), ['species'], 'images left out')
})
