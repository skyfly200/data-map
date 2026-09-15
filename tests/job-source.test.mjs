/* What a job is allowed to run over.
 *
 * The dataset source is the interesting half. It used to build a storage path
 * straight from the slug the caller sent, with no check that they were allowed
 * to read it — harmless while only admins made datasets, and a way to read
 * anyone's private work the moment members own them. These tests are the
 * regression.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { DatasetAccessError } from '../netlify/lib/dataset-access.mjs'
import { countDates, loadSource, measureSource, selectFeatures, withinBounds }
  from '../netlify/lib/job-source.mjs'

const point = (lon, lat, props = {}) => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lon, lat] },
  properties: props,
})

// ── Selecting from a bounding box ────────────────────────────────────────────

const BOX = { west: -105.5, east: -105.0, south: 40.0, north: 40.5 }

test('a point inside the box is selected and one outside is not', () => {
  assert.ok(withinBounds(point(-105.2, 40.2), BOX))
  assert.ok(!withinBounds(point(-104.0, 40.2), BOX))
  assert.ok(!withinBounds(point(-105.2, 41.0), BOX))
})

test('a point with no usable coordinates is not inside anything', () => {
  assert.ok(!withinBounds(point('x', 40.2), BOX))
  assert.ok(!withinBounds({ properties: {} }, BOX))
  assert.ok(!withinBounds(null, BOX))
})

test('a box across the antimeridian still contains its points', () => {
  // normaliseBounds pushes east past 180 for a wrapped box, so a point at -179
  // has to be found at +181.
  const wrapped = { west: 170, east: 190, south: -10, north: 10 }
  assert.ok(withinBounds(point(175, 0), wrapped))
  assert.ok(withinBounds(point(-179, 0), wrapped), 'a point just past the line is inside')
  assert.ok(!withinBounds(point(0, 0), wrapped))
})

test('a taxon matches at any rank', () => {
  const features = [
    point(-105.2, 40.2, { date: '2026-09-01', genus: 'Morchella' }),
    point(-105.2, 40.2, { date: '2026-09-01', order: 'Agaricales' }),
    point(-105.2, 40.2, { date: '2026-09-01', genus: 'Amanita' }),
  ]
  assert.equal(selectFeatures(features, { bounds: BOX, taxon: 'Morchella' }).length, 1)
  assert.equal(selectFeatures(features, { bounds: BOX, taxon: 'agaricales' }).length, 1)
  assert.equal(selectFeatures(features, { bounds: BOX, taxon: '' }).length, 3)
})

test('a date range excludes records outside it, and ones with no date at all', () => {
  const features = [
    point(-105.2, 40.2, { date: '2026-08-01' }),
    point(-105.2, 40.2, { date: '2026-09-15' }),
    point(-105.2, 40.2, {}),
  ]
  const got = selectFeatures(features, { bounds: BOX, dateFrom: '2026-09-01', dateTo: '2026-09-30' })
  assert.equal(got.length, 1)
})

test('dates are counted distinctly, since that is what the dated stages cost per', () => {
  assert.equal(countDates([
    point(0, 0, { date: '2026-09-01' }),
    point(0, 0, { date: '2026-09-01' }),
    point(0, 0, { date: '2026-09-02' }),
  ]), 2)
  // Never zero: a job over undated points still makes one pass.
  assert.equal(countDates([point(0, 0, {})]), 1)
  assert.equal(countDates([]), 1)
})

// ── A dataset as the source ──────────────────────────────────────────────────

const OWNER = 'user-1'
const OTHER = 'user-2'

const row = {
  id: 'd1', slug: 'autumn-2026', owner_id: OWNER, visibility: 'private',
  path: 'jobs/user-1/job-9.geojson',
}

const clientReturning = (found) => ({
  from: () => ({
    select: () => ({ eq: () => ({ maybeSingle: async () => ({ data: found, error: null }) }) }),
  }),
})

const source = { type: 'dataset', slug: 'autumn-2026' }

test('the owner gets their dataset, read from the path on the row', async () => {
  const asked = []
  const read = async (path) => {
    asked.push(path)
    return { features: [point(-105.2, 40.2, { date: '2026-09-01' })] }
  }
  const features = await loadSource(source, {
    client: clientReturning(row), viewer: { userId: OWNER, tier: 'member' }, read,
  })
  assert.equal(features.length, 1)
  // The regression: the path comes off the row, not from the slug the caller
  // sent. A row is the only way to name a stored file.
  assert.deepEqual(asked, ['jobs/user-1/job-9.geojson'])
})

test('another member cannot read a private dataset by naming its slug', async () => {
  let readCalled = false
  const read = async () => { readCalled = true; return { features: [] } }
  await assert.rejects(
    () => loadSource(source, {
      client: clientReturning(row), viewer: { userId: OTHER, tier: 'member' }, read,
    }),
    DatasetAccessError,
  )
  assert.equal(readCalled, false, 'storage must not be touched for a dataset they cannot read')
})

test('a members-visible dataset is readable by another member', async () => {
  const shared = { ...row, visibility: 'members' }
  const read = async () => ({ features: [point(-105.2, 40.2, {})] })
  const features = await loadSource(source, {
    client: clientReturning(shared), viewer: { userId: OTHER, tier: 'member' }, read,
  })
  assert.equal(features.length, 1)
})

test('a lapsed member cannot read a members-visible dataset', async () => {
  // The viewer carries the effective tier, so a lapsed membership arrives here
  // as 'free' and is refused without this file knowing about dates.
  const shared = { ...row, visibility: 'members' }
  await assert.rejects(
    () => loadSource(source, {
      client: clientReturning(shared), viewer: { userId: OTHER, tier: 'free' }, read: async () => ({}),
    }),
    DatasetAccessError,
  )
})

test('a registered dataset whose file is gone is our fault, not a refusal', async () => {
  // Distinct from "no such dataset" on purpose: same-looking failures here
  // would hide a broken deployment behind a permissions message.
  const err = await loadSource(source, {
    client: clientReturning(row), viewer: { userId: OWNER, tier: 'member' }, read: async () => null,
  }).catch((e) => e)
  assert.ok(err instanceof DatasetAccessError)
  assert.equal(err.status, 500)
  assert.equal(err.code, 'no_file')
})

test('measuring a dataset source is scoped the same way as loading it', async () => {
  // Pricing runs before the job is queued and reads the whole source to count
  // it. An unscoped count would leak the size of a private dataset.
  await assert.rejects(
    () => measureSource({ source }, {
      client: clientReturning(row), viewer: { userId: OTHER, tier: 'member' }, read: async () => ({}),
    }),
    DatasetAccessError,
  )

  const measured = await measureSource({ source }, {
    client: clientReturning(row),
    viewer: { userId: OWNER, tier: 'member' },
    read: async () => ({ features: [
      point(0, 0, { date: '2026-09-01' }),
      point(0, 0, { date: '2026-09-02' }),
    ] }),
  })
  assert.deepEqual(measured, { points: 2, dates: 2 })
})
