/* Getting data out of the app.
 *
 * Almost everything that can go wrong here is silent. A badly quoted CSV field
 * shifts every column after it in one row out of a thousand, and the file still
 * opens. A missing BOM mangles accented names in Excel and nowhere else. So
 * these are mostly about the encoding rather than the plumbing.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  GEOMETRY_COLUMNS, IDENTITY_COLUMNS,
  columnsOf, exportFilename, quoteCsv, toCsv, toGeoJson,
} from '../composables/dataExport.js'

const feature = (props = {}, lon = -105.2, lat = 40.1) => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lon, lat] },
  properties: props,
})

const rows = (csv) => csv.trimEnd().split('\r\n')

// ── Columns ──────────────────────────────────────────────────────────────────

test('columns are the union across features, not the first row', () => {
  // An enriched set is ragged: a job that ran different stages leaves rows with
  // different columns, and a reader should get all of them.
  const cols = columnsOf([
    feature({ id: 1, ndvi: 0.4 }),
    feature({ id: 2, slope: 12 }),
  ])
  assert.deepEqual(cols, ['id', 'ndvi', 'slope'])
})

test('identity columns come first, the rest sorted', () => {
  const cols = columnsOf([feature({ zebra: 1, ndvi: 2, scientific_name: 'x', date: 'y', id: 3 })])
  assert.deepEqual(cols.slice(0, 3), ['id', 'date', 'scientific_name'])
  assert.deepEqual(cols.slice(3), ['ndvi', 'zebra'])
  assert.ok(IDENTITY_COLUMNS.includes('id'))
})

test('no features means no columns rather than a throw', () => {
  assert.deepEqual(columnsOf([]), [])
  assert.deepEqual(columnsOf(), [])
  assert.deepEqual(columnsOf([{}, null]), [])
})

// ── CSV quoting ──────────────────────────────────────────────────────────────

test('a field with a comma is quoted', () => {
  assert.equal(quoteCsv('Boulder, Colorado'), '"Boulder, Colorado"')
})

test('a quote inside a field is doubled, and the field quoted', () => {
  assert.equal(quoteCsv('the "brown" one'), '"the ""brown"" one"')
})

test('a newline inside a field is quoted rather than ending the row', () => {
  // The one that shifts every subsequent row instead of every subsequent
  // column, and still opens cleanly enough that nobody notices.
  assert.equal(quoteCsv('line one\nline two'), '"line one\nline two"')
  assert.equal(quoteCsv('carriage\rreturn'), '"carriage\rreturn"')
})

test('an ordinary field is not quoted', () => {
  assert.equal(quoteCsv('Morchella'), 'Morchella')
  assert.equal(quoteCsv(''), '')
})

test('null and undefined are empty, not the strings "null" and "undefined"', () => {
  assert.equal(quoteCsv(null), '')
  assert.equal(quoteCsv(undefined), '')
})

test('numbers pass through, and non-finite ones become empty', () => {
  assert.equal(quoteCsv(0), '0')
  assert.equal(quoteCsv(-5), '-5')
  assert.equal(quoteCsv(0.4321), '0.4321')
  // NaN and Infinity are not values a spreadsheet can do anything with, and
  // "NaN" in a numeric column is worse than a blank.
  assert.equal(quoteCsv(NaN), '')
  assert.equal(quoteCsv(Infinity), '')
})

test('a negative number is not mistaken for a formula', () => {
  // The prefix guard must not reach numbers: -5 is a measurement somebody made.
  assert.equal(quoteCsv(-5), '-5')
  assert.equal(quoteCsv(-0.25), '-0.25')
})

test('text a spreadsheet would evaluate is neutralised, not altered', () => {
  // These fields carry text from iNaturalist, which is to say from the public.
  // A cell beginning =, +, - or @ is a formula in Excel, Sheets and Numbers.
  assert.equal(quoteCsv('=1+1'), "'=1+1")
  assert.equal(quoteCsv('=HYPERLINK("http://evil","click")'),
    '"\'=HYPERLINK(""http://evil"",""click"")"')
  assert.equal(quoteCsv('@SUM(A1:A9)'), "'@SUM(A1:A9)")
  assert.equal(quoteCsv('+1234'), "'+1234")
  assert.equal(quoteCsv('-lookup'), "'-lookup")
  // The value is preserved with a prefix rather than stripped: a reader that
  // does not evaluate formulas can still see what was recorded.
  assert.ok(quoteCsv('=1+1').includes('=1+1'))
})

test('a nested value keeps to one field as JSON', () => {
  assert.equal(quoteCsv({ a: 1 }), '"{""a"":1}"')
  assert.equal(quoteCsv([1, 2]), '"[1,2]"')
})

test('booleans are written as words', () => {
  assert.equal(quoteCsv(true), 'true')
  assert.equal(quoteCsv(false), 'false')
})

// ── The CSV itself ───────────────────────────────────────────────────────────

test('coordinates get their own columns, since a CSV has nowhere else', () => {
  const csv = toCsv([feature({ id: 1 }, -105.25, 40.12)])
  const [header, first] = rows(csv)
  assert.equal(header, 'longitude,latitude,id')
  assert.equal(first, '-105.25,40.12,1')
  assert.deepEqual(GEOMETRY_COLUMNS, ['longitude', 'latitude'])
})

test('a ragged set still lines up, with blanks where a row has nothing', () => {
  const csv = toCsv([feature({ id: 1, ndvi: 0.4 }), feature({ id: 2 })])
  const [header, a, b] = rows(csv)
  assert.equal(header, 'longitude,latitude,id,ndvi')
  assert.equal(a, '-105.2,40.1,1,0.4')
  assert.equal(b, '-105.2,40.1,2,')
})

test('columns can be narrowed, and the order given is the order written', () => {
  const csv = toCsv([feature({ id: 1, ndvi: 0.4, slope: 12 })], { columns: ['slope', 'id'] })
  assert.equal(rows(csv)[0], 'longitude,latitude,slope,id')
  assert.equal(rows(csv)[1], '-105.2,40.1,12,1')
})

test('geometry columns can be left out for a set that is not about place', () => {
  const csv = toCsv([feature({ id: 1 })], { geometry: false })
  assert.equal(rows(csv)[0], 'id')
})

test('rows end with CRLF and the file ends with one', () => {
  // Excel on Windows runs a whole LF-only file into a single row.
  const csv = toCsv([feature({ id: 1 }), feature({ id: 2 })])
  assert.ok(csv.endsWith('\r\n'))
  assert.equal(csv.split('\r\n').length, 4)   // header, two rows, trailing empty
})

test('an empty set still produces a header', () => {
  // A file with no header is one nobody can tell apart from a broken download.
  const csv = toCsv([], { columns: ['id', 'ndvi'] })
  assert.equal(csv, 'longitude,latitude,id,ndvi\r\n')
})

test('a missing geometry leaves the coordinate cells blank rather than throwing', () => {
  const csv = toCsv([{ properties: { id: 1 } }])
  assert.equal(rows(csv)[1], ',,1')
})

// ── GeoJSON ──────────────────────────────────────────────────────────────────

test('the output is a valid FeatureCollection', () => {
  const parsed = JSON.parse(toGeoJson([feature({ id: 1 })]))
  assert.equal(parsed.type, 'FeatureCollection')
  assert.equal(parsed.features.length, 1)
  assert.deepEqual(parsed.features[0].geometry.coordinates, [-105.2, 40.1])
})

test('narrowing columns keeps the geometry', () => {
  // Dropping it to save width leaves something that is no longer GeoJSON.
  const parsed = JSON.parse(toGeoJson([feature({ id: 1, ndvi: 0.4 })], { columns: ['ndvi'] }))
  assert.deepEqual(parsed.features[0].properties, { ndvi: 0.4 })
  assert.ok(parsed.features[0].geometry, 'geometry must survive a narrowed export')
})

test('narrowing does not invent keys a row never had', () => {
  const parsed = JSON.parse(toGeoJson([feature({ id: 1 })], { columns: ['id', 'ndvi'] }))
  assert.deepEqual(Object.keys(parsed.features[0].properties), ['id'])
})

test('a narrowed export does not mutate what it was given', () => {
  const source = feature({ id: 1, ndvi: 0.4 })
  toGeoJson([source], { columns: ['id'] })
  assert.deepEqual(source.properties, { id: 1, ndvi: 0.4 })
})

test('pretty output is indented and still parses the same', () => {
  const plain = toGeoJson([feature({ id: 1 })])
  const pretty = toGeoJson([feature({ id: 1 })], { pretty: true })
  assert.ok(pretty.includes('\n'))
  assert.ok(!plain.includes('\n'))
  assert.deepEqual(JSON.parse(pretty), JSON.parse(plain))
})

// ── Filenames ────────────────────────────────────────────────────────────────

test('a filename carries the view, the date and the right extension', () => {
  const at = new Date('2026-09-15T12:00:00Z')
  assert.equal(exportFilename('Autumn foray, Front Range', 'csv', { stamp: at }),
    'autumn-foray-front-range-2026-09-15.csv')
  assert.equal(exportFilename('Autumn foray', 'geojson', { stamp: at }),
    'autumn-foray-2026-09-15.geojson')
})

test('a title that slugifies to nothing still gives a usable name', () => {
  const at = new Date('2026-09-15T12:00:00Z')
  assert.equal(exportFilename('—', 'csv', { stamp: at }), 'export-2026-09-15.csv')
  assert.equal(exportFilename('', 'csv', { stamp: at }), 'export-2026-09-15.csv')
})
