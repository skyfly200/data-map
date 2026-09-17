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
import {
  countDates, explainEmpty, explainSelection, loadSource, matchesTaxon, measureSource,
  selectFeatures, withinBounds, withinDates,
} from '../netlify/lib/job-source.mjs'

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

// ── Matching a taxon against a dataset with no rank columns ──────────────────
//
// The committed baseline predates the taxonomy work: it carries a `species`
// binomial and nothing else. Against that, a rank-column match found nothing
// for every genus anybody would type — so a job for "Amanita" in an area with
// ten thousand observations in it was refused, and refused with a message
// blaming the area and the dates, which were both fine.

test('a rank column is the answer where it exists', () => {
  const props = { kingdom: 'Fungi', order: 'Agaricales', genus: 'Amanita', species: 'Amanita muscaria' }
  for (const t of ['Fungi', 'Agaricales', 'Amanita', 'Amanita muscaria', 'AMANITA', ' amanita ']) {
    assert.ok(matchesTaxon(props, t), `${t} did not match`)
  }
  assert.ok(!matchesTaxon(props, 'Boletus'))
  assert.ok(!matchesTaxon(props, 'Morchella'))
})

test('a genus matches a binomial when the dataset has no genus column', () => {
  // This is the whole bug: 48,000 records with species and no genus.
  const legacy = { species: 'Caloboletus conifericola' }
  assert.ok(matchesTaxon(legacy, 'Caloboletus'), 'a genus query found nothing in a legacy dataset')
  assert.ok(matchesTaxon(legacy, 'Caloboletus conifericola'))
  assert.ok(!matchesTaxon(legacy, 'conifericola'), 'the epithet is not the genus')
  assert.ok(!matchesTaxon(legacy, 'Boletus'), 'a different genus matched')
})

test('a real genus column wins over the binomial guess', () => {
  // Splitting a name on its spaces is the guesswork the rank columns replaced,
  // so it only applies where there is no column to disagree with. A record
  // whose genus says one thing and whose species string says another is
  // answered by the column.
  const props = { genus: 'Amanita', species: 'Lepiota naucina' }
  assert.ok(matchesTaxon(props, 'Amanita'))
  assert.ok(!matchesTaxon(props, 'Lepiota'), 'the binomial overrode the genus column')
})

test('an empty taxon matches everything, and an empty record matches nothing', () => {
  assert.ok(matchesTaxon({ species: 'Amanita muscaria' }, ''))
  assert.ok(matchesTaxon({}, '   '))
  assert.ok(matchesTaxon({}, null))
  assert.ok(!matchesTaxon({}, 'Amanita'))
  assert.ok(!matchesTaxon({ species: '' }, 'Amanita'))
})

test('a bbox selection still filters by taxon through the legacy path', () => {
  const features = [
    { geometry: { coordinates: [-105, 40] }, properties: { date: '2026-08-01', species: 'Amanita muscaria' } },
    { geometry: { coordinates: [-105, 40] }, properties: { date: '2026-08-01', species: 'Boletus edulis' } },
  ]
  const source = { bounds: { north: 41, south: 39, west: -106, east: -104 }, taxon: 'Amanita' }
  assert.equal(selectFeatures(features, source).length, 1)
  assert.equal(selectFeatures(features, { ...source, taxon: '' }).length, 2)
})

// ── Saying which filter emptied the selection ────────────────────────────────

test('each filter is counted separately, so the empty one can be named', () => {
  const features = [
    { geometry: { coordinates: [-105, 40] }, properties: { date: '2026-08-01', species: 'Amanita muscaria' } },
    { geometry: { coordinates: [-105, 40] }, properties: { date: '2020-08-01', species: 'Boletus edulis' } },
    { geometry: { coordinates: [0, 0] }, properties: { date: '2026-08-01', species: 'Amanita muscaria' } },
  ]
  const seen = explainSelection(features, {
    bounds: { north: 41, south: 39, west: -106, east: -104 },
    dateFrom: '2026-01-01', dateTo: '2026-12-31', taxon: 'Amanita',
  })
  assert.equal(seen.total, 3)
  assert.equal(seen.inBounds, 2)
  assert.equal(seen.inDates, 1)
  assert.equal(seen.selected.length, 1)
})

test('the refusal names the area when nothing is in it', () => {
  const msg = explainEmpty({ taxon: '' }, { total: 48233, inBounds: 0, inDates: 0 })
  assert.match(msg, /No observations fall inside that area/)
  assert.match(msg, /current map view/, 'it does not say what to do instead')
})

test('the refusal names the dates, with how many were in the area', () => {
  const msg = explainEmpty({ dateFrom: '2030-01-01', dateTo: '2030-12-31' },
    { total: 48233, inBounds: 10084, inDates: 0 })
  assert.match(msg, /10,084 observations are in that area/)
  assert.match(msg, /2030-01-01/)
  assert.match(msg, /2030-12-31/)
  assert.match(msg, /Widen the date range/)
})

test('the refusal names the taxon, with how many survived the dates', () => {
  // The case that actually happened, and the one the old message described
  // least accurately of the three.
  const msg = explainEmpty({ taxon: 'Amanita', dateFrom: '2026-07-01', dateTo: '2026-09-17' },
    { total: 48233, inBounds: 10084, inDates: 296 })
  assert.match(msg, /296 observations are in that area and date range/)
  assert.match(msg, /“Amanita”/)
  assert.match(msg, /Clear the taxon/, 'it does not say what to do instead')
  assert.ok(!/date range\./.test(msg.replace(/in that area and date range/, '')),
    'it still blames the dates')
})

test('a dataset that could not be read is reported as our fault, not theirs', () => {
  const msg = explainEmpty({}, { total: 0, inBounds: 0, inDates: 0 })
  assert.match(msg, /could not be read/)
  assert.match(msg, /our side/)
})

test('with no breakdown it falls back to the old wording rather than throwing', () => {
  // A dataset source has no breakdown to give, and neither does an older
  // counter, so the message has to survive its absence.
  assert.match(explainEmpty({}, null), /no observations to enrich/)
  assert.match(explainEmpty(), /no observations to enrich/)
})

// ── Dates ────────────────────────────────────────────────────────────────────

test('an undated record is outside any date range, and inside no range at all', () => {
  const undated = { properties: {} }
  assert.ok(withinDates(undated, {}), 'no range means every record is in it')
  assert.ok(!withinDates(undated, { dateFrom: '2020-01-01' }))
  assert.ok(!withinDates(undated, { dateTo: '2020-01-01' }))
})

test('a date range includes both of its ends', () => {
  const on = (d) => ({ properties: { date: d } })
  const range = { dateFrom: '2026-07-01', dateTo: '2026-09-17' }
  assert.ok(withinDates(on('2026-07-01'), range))
  assert.ok(withinDates(on('2026-09-17'), range))
  assert.ok(!withinDates(on('2026-06-30'), range))
  assert.ok(!withinDates(on('2026-09-18'), range))
  // A full timestamp is cut to its date, which is what the pipeline writes.
  assert.ok(withinDates(on('2026-08-01T14:22:00Z'), range))
})
