import test from 'node:test'
import assert from 'node:assert/strict'
import {
  INAT_DATASET_KEY, searchParams, partitionYears, dropReason, gbifRecordToFeature, mergeGbif, inatIdFrom,
} from '../netlify/lib/gbif-bulk.mjs'

const rec = (o = {}) => ({
  key: 1, datasetKey: 'herbarium', basisOfRecord: 'PRESERVED_SPECIMEN',
  decimalLatitude: 39.5, decimalLongitude: -106.1, species: 'Morchella snyderi', genus: 'Morchella',
  eventDate: '2019-06-02T00:00:00', year: 2019, coordinateUncertaintyInMeters: 100, issues: [], ...o,
})

test('search params cover the bbox, fungi and a year range', () => {
  const p = searchParams({ bbox: [-109, 37, -102, 41], years: [2000, 2010] })
  assert.equal(p.get('taxonKey'), '5')
  assert.equal(p.get('decimalLatitude'), '37,41')
  assert.equal(p.get('decimalLongitude'), '-109,-102')
  assert.equal(p.get('year'), '2000,2010')
})

test('year ranges are halved until each fits the search window', async () => {
  // 1 record per year, window of 10
  const count = async ([a, b]) => b - a + 1
  const parts = await partitionYears(count, [1990, 2019], 10)
  assert.ok(parts.every((p) => p.count <= 10))
  assert.equal(parts.reduce((a, p) => a + p.count, 0), 30)
  assert.equal(parts[0].years[0], 1990)
  assert.equal(parts.at(-1).years[1], 2019)
})

test('an empty range yields no queries', async () => {
  assert.deepEqual(await partitionYears(async () => 0, [1900, 2000]), [])
})

test('iNaturalist copies are dropped by dataset or occurrenceID', () => {
  assert.equal(dropReason(rec({ datasetKey: INAT_DATASET_KEY })), 'inat')
  assert.equal(dropReason(rec({ occurrenceID: 'https://www.inaturalist.org/observations/12345' })), 'inat')
  assert.equal(inatIdFrom('https://www.inaturalist.org/observations/12345'), 12345)
  assert.equal(dropReason(rec()), null)
})

test('coarse, nameless and suspect records are dropped', () => {
  assert.equal(dropReason(rec({ coordinateUncertaintyInMeters: 20000 })), 'uncertainty')
  assert.equal(dropReason(rec({ coordinateUncertaintyInMeters: null })), null)
  assert.equal(dropReason(rec({ species: null, genus: null })), 'no-name')
  assert.equal(dropReason(rec({ year: 1950 }), { minYear: 1970 }), 'year')
  assert.equal(dropReason(rec({ issues: ['COUNTRY_COORDINATE_MISMATCH'] })), 'coords-suspect')
})

test('a record becomes a feature in the dataset shape', () => {
  const f = gbifRecordToFeature(rec({ species: null }))
  assert.deepEqual(f.geometry.coordinates, [-106.1, 39.5])
  assert.equal(f.properties.uuid, 'gbif:1')
  assert.equal(f.properties.species, 'Morchella')
  assert.equal(f.properties.date, '2019-06-02')
  assert.equal(f.properties.day_of_year, 153)
  assert.equal(f.properties.source, 'gbif')
})

test('a partial date has no day of year', () => {
  const f = gbifRecordToFeature(rec({ eventDate: '1975' }))
  assert.equal(f.properties.date, null)
  assert.equal(f.properties.day_of_year, null)
})

test('merge keeps iNat features and replaces earlier GBIF ones', () => {
  const baseline = {
    type: 'FeatureCollection',
    features: [
      { properties: { uuid: 'abc', inat_id: 9 } },
      { properties: { uuid: 'gbif:old', source: 'gbif' } },
    ],
  }
  const fresh = [gbifRecordToFeature(rec({ key: 2 })), gbifRecordToFeature(rec({ key: 2 }))]
  const { collection, added, replaced } = mergeGbif(baseline, fresh)
  assert.equal(added, 1)
  assert.equal(replaced, 1)
  assert.deepEqual(collection.features.map((f) => f.properties.uuid), ['abc', 'gbif:2'])
})
