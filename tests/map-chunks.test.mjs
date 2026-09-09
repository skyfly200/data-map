/* The chunk scheme has two halves that must agree exactly.
 *
 * scripts/mapChunks.mjs decides which file an observation is written into at
 * build time; composables/useMapChunks.js decides which file to ask for at run
 * time. If the two ever disagree about the cell a coordinate belongs to, the
 * map does not break loudly: it just quietly never finds anything, which is the
 * kind of failure that survives a manual look at the page. So the agreement is
 * asserted here rather than assumed.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  CELL_SIZE, DERIVED, OVERVIEW_FIELDS, buildOverview, cellKey, slim, splitFeatures,
} from '../scripts/mapChunks.mjs'
import { cellKeyFor, cellsForBounds, pickCells, MAX_CELLS_PER_MOVE } from '../composables/useMapChunks.js'

const point = (lat, lon, props = {}) => ({
  type: 'Feature',
  geometry: { type: 'Point', coordinates: [lon, lat] },
  properties: props,
})

test('the writer and the loader name the same cell', () => {
  // Including negatives and exact boundaries, where a floor and a truncation
  // part company and half the southern or western hemisphere goes missing.
  const coords = [
    [39.7, -105.2], [0, 0], [-0.1, -0.1], [-33.9, 151.2], [64.13, -21.9],
    [40.0, -105.0], [39.999999, -105.000001], [-45.5, -72.5], [89.9, 179.9],
  ]
  for (const [lat, lon] of coords) {
    assert.equal(cellKeyFor(lat, lon, CELL_SIZE), cellKey(lat, lon, CELL_SIZE),
      `disagreement at ${lat},${lon}`)
  }
})

test('the writer and the loader agree across a random sweep', () => {
  let seed = 12345
  const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648 }
  for (let i = 0; i < 20000; i += 1) {
    const lat = rand() * 180 - 90
    const lon = rand() * 360 - 180
    assert.equal(cellKeyFor(lat, lon, CELL_SIZE), cellKey(lat, lon, CELL_SIZE))
  }
})

test('cellsForBounds covers every cell a viewport touches', () => {
  const bounds = { south: 39.2, north: 40.4, west: -105.6, east: -104.4 }
  const keys = new Set(cellsForBounds(bounds, CELL_SIZE))

  // Anything inside the box must be in a cell the loader asked for. Sampled on
  // a fine grid, because the failure mode is an edge strip rather than a whole
  // missing cell.
  for (let lat = bounds.south; lat <= bounds.north; lat += 0.05) {
    for (let lon = bounds.west; lon <= bounds.east; lon += 0.05) {
      assert.ok(keys.has(cellKey(lat, lon, CELL_SIZE)), `missed ${lat},${lon}`)
    }
  }
  // 39.2..40.4 spans rows 78,79,80 and -105.6..-104.4 spans columns -212 to
  // -209, so twelve cells and no more: an over-wide box means fetching ground
  // nobody is looking at.
  assert.equal(keys.size, 12)
})

test('cellsForBounds works south and west of zero', () => {
  const keys = cellsForBounds({ south: -0.6, north: 0.4, west: -0.6, east: 0.4 }, CELL_SIZE)
  assert.ok(keys.includes(cellKey(-0.3, -0.3, CELL_SIZE)))
  assert.ok(keys.includes(cellKey(0.2, 0.2, CELL_SIZE)))
  assert.equal(new Set(keys).size, keys.length)
})

test('pickCells skips what is loaded and what the index says is empty', () => {
  const index = { cells: { '0_0': { n: 5 }, '0_1': { n: 3 }, '1_0': { n: 0 }, '1_1': { n: 9 } } }
  const loaded = new Set(['0_1'])
  const got = pickCells(['0_0', '0_1', '1_0', '1_1', '9_9'], {
    loaded, index, centre: null, size: CELL_SIZE,
  })
  // '0_1' is already held, '1_0' holds nothing, '9_9' is not in the index at
  // all. Requesting any of the three would be a round trip for nothing.
  assert.deepEqual(got.sort(), ['0_0', '1_1'])
})

test('pickCells asks for the middle of the view first', () => {
  const cells = {}
  for (let y = 0; y < 6; y += 1) for (let x = 0; x < 6; x += 1) cells[`${y}_${x}`] = { n: 1 }
  const keys = Object.keys(cells)
  // Centre on the cell at row 5, column 5 (0.5-degree cells, so 2.75,2.75).
  const got = pickCells(keys, {
    loaded: new Set(), index: { cells }, centre: [2.75, 2.75], size: CELL_SIZE, limit: 3,
  })
  assert.equal(got[0], '5_5')
  assert.equal(got.length, 3)
})

test('pickCells never fires more than one move is worth', () => {
  const cells = {}
  for (let y = 0; y < 20; y += 1) for (let x = 0; x < 20; x += 1) cells[`${y}_${x}`] = { n: 1 }
  const got = pickCells(Object.keys(cells), {
    loaded: new Set(), index: { cells }, centre: [5, 5], size: CELL_SIZE,
  })
  assert.equal(got.length, MAX_CELLS_PER_MOVE)
})

test('splitFeatures puts every feature in the cell the loader would ask for', () => {
  const feats = [
    point(39.7, -105.2, { species: 'a' }),
    point(39.9, -105.4, { species: 'b' }),
    point(-12.3, 44.8, { species: 'c' }),
    { type: 'Feature', geometry: null, properties: {} },        // dropped
    point(Number.NaN, 10, { species: 'd' }),                    // dropped
  ]
  const cells = splitFeatures(feats)
  const placed = [...cells.values()].reduce((n, list) => n + list.length, 0)
  assert.equal(placed, 3)
  for (const [key, list] of cells) {
    for (const f of list) {
      const [lon, lat] = f.geometry.coordinates
      assert.equal(cellKeyFor(lat, lon, CELL_SIZE), key)
    }
  }
})

test('the overview thins by stride, so it stays spread over the whole map', () => {
  const feats = []
  for (let i = 0; i < 1000; i += 1) feats.push(point(i / 100, i / 100, { species: `s${i}` }))
  const { features, stride } = buildOverview(feats, { limit: 100, fields: ['species'] })
  assert.equal(stride, 10)
  assert.equal(features.length, 100)
  // First and last both present: taking the head instead would draw one corner
  // of the map in full and leave the rest blank.
  assert.equal(features[0].properties.species, 's0')
  assert.equal(features.at(-1).properties.species, 's990')
})

test('an unthinned overview keeps everything', () => {
  const feats = [point(1, 1), point(2, 2)]
  const { features, stride } = buildOverview(feats, { limit: 0 })
  assert.equal(stride, 1)
  assert.equal(features.length, 2)
})

test('slim keeps the geometry and drops the fields the map does not draw', () => {
  const f = point(39.7, -105.2, { species: 'Amanita', elevation: 2400, twi: null, notes: '' })
  const out = slim(f, ['species', 'elevation', 'twi', 'notes'])
  assert.deepEqual(out.geometry, f.geometry)
  assert.equal(out.properties.species, 'Amanita')
  assert.equal(out.properties.elevation, 2400)
  // Empty and null are omitted rather than written: at 48,233 features the
  // difference between `null` and absent is megabytes.
  assert.ok(!('twi' in out.properties))
  assert.ok(!('notes' in out.properties))
})

test('the overview carries a column for every field heatmap', () => {
  // The regression this guards: the first version of OVERVIEW_FIELDS held only
  // what a point is drawn from, so every field heatmap except elevation drew a
  // blank map at the zoom the overview is showing. A mode offered in the menu
  // must have its column here — or be derivable from one.
  const HEATMAP_FIELDS = [
    'rain7', 'tavg', 'soil_moisture', 'water_retention', 'slope', 'aspect',
    'solar_exposure', 'wind_exposure', 'ndvi', 'ndmi', 'elevation',
  ]
  for (const field of HEATMAP_FIELDS) {
    assert.ok(OVERVIEW_FIELDS.includes(field), `${field} is missing from the overview`)
  }
})

test('rain7 is summed at build time rather than carried as seven columns', () => {
  const props = { prcp_d0: 1.5, prcp_d1: 2, prcp_d2: 0, prcp_d3: null, prcp_d4: '', prcp_d5: 3, prcp_d6: 0.25 }
  assert.equal(DERIVED.rain7(props), 6.75)
  // Absent throughout is null, not zero: no data is not the same as no rain,
  // and a zero would colour a cell as dry.
  assert.equal(DERIVED.rain7({}), null)
  assert.equal(DERIVED.rain7({ prcp_d0: null }), null)
  // A single present day still totals.
  assert.equal(DERIVED.rain7({ prcp_d3: 4 }), 4)
})

test('tavg is the midpoint of the day s max and min', () => {
  assert.equal(DERIVED.tavg({ tmax_d0: 20, tmin_d0: 10 }), 15)
  // One end missing is better than nothing; both missing is null.
  assert.equal(DERIVED.tavg({ tmax_d0: 20 }), 20)
  assert.equal(DERIVED.tavg({ tmin_d0: 10 }), 10)
  assert.equal(DERIVED.tavg({}), null)
  // A real tavg column wins over the derivation, so a future pipeline run that
  // fills it in is used as-is.
  assert.equal(DERIVED.tavg({ tavg: 12.5, tmax_d0: 20, tmin_d0: 10 }), 12.5)
  // Below freezing has to survive: mushroom season runs into hard frosts.
  assert.equal(DERIVED.tavg({ tmax_d0: -1, tmin_d0: -9 }), -5)
})

test('slim rounds the numbers it copies and keeps the strings', () => {
  const f = point(39.7, -105.2, {
    species: 'Amanita muscaria', slope: 12.3456789, prcp_d0: 1.111, prcp_d1: 2.222,
  })
  const out = slim(f, ['species', 'slope', 'rain7'])
  assert.equal(out.properties.species, 'Amanita muscaria')
  // Four decimals is far finer than a colour ramp over a grid cell can show.
  assert.equal(out.properties.slope, 12.3457)
  assert.equal(out.properties.rain7, 3.33)
})
