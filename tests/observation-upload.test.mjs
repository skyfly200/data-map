import test from 'node:test'
import assert from 'node:assert/strict'
import { parseCsv, toFeatureCollection, geojsonToFeatures, guessFormat, UploadError } from '../netlify/lib/observation-upload.mjs'

test('csv: detects iNat/GBIF columns, skips bad rows, handles quotes', () => {
  const csv = 'id,observed_on,scientific_name,latitude,longitude\n1,2024-05-01T10:00,"Morchella, sp.",40.1,-105.2\n2,2024-05-02,Amanita,abc,-105\n3,,X,95,10\n'
  const r = toFeatureCollection({ format: 'csv', content: csv })
  assert.equal(r.features.length, 1)
  assert.equal(r.skipped, 2)
  assert.deepEqual(r.features[0].geometry.coordinates, [-105.2, 40.1])
  assert.equal(r.features[0].properties.species, 'Morchella, sp.')
  assert.equal(r.features[0].properties.date, '2024-05-01')
})

test('csv: quoted newline, BOM, CRLF', () => {
  const { rows } = parseCsv('﻿lat,lon,name\r\n1,2,"a\nb"\r\n')
  assert.equal(rows[0].name, 'a\nb')
})

test('csv: missing lat column and explicit override', () => {
  assert.throws(() => toFeatureCollection({ format: 'csv', content: 'a,b\n1,2\n' }), (e) => e instanceof UploadError && e.code === 'no_lat_col')
  const r = toFeatureCollection({ format: 'csv', content: 'a,b\n1,2\n', cols: { lat_col: 'a', lon_col: 'b' } })
  assert.deepEqual(r.features[0].geometry.coordinates, [2, 1])
})

test('geojson: collection, non-point, invalid', () => {
  const r = geojsonToFeatures({ type: 'FeatureCollection', features: [
    { type: 'Feature', geometry: { type: 'Point', coordinates: [-105, 40] }, properties: { Species: 'A', date: '2024-01-02T00:00' } },
    { type: 'Feature', geometry: { type: 'Polygon', coordinates: [[[1, 2], [3, 4], [1, 2]]] }, properties: {} },
    { type: 'Feature', geometry: null, properties: {} },
  ] })
  assert.equal(r.features.length, 2)
  assert.equal(r.skipped, 1)
  assert.equal(r.features[0].properties.species, 'A')
  assert.equal(r.features[0].properties.date, '2024-01-02')
  assert.throws(() => geojsonToFeatures('nope'), UploadError)
})

test('limits and format', () => {
  assert.throws(() => toFeatureCollection({ format: 'csv', content: 'x'.repeat(3 * 1024 * 1024 + 1) }), (e) => e.code === 'too_large')
  assert.throws(() => toFeatureCollection({ format: 'kml', content: 'a' }), (e) => e.code === 'bad_format')
  assert.equal(guessFormat('a.geojson'), 'geojson')
  assert.equal(guessFormat('x', '{"type"'), 'geojson')
  assert.equal(guessFormat('x.csv'), 'csv')
})
