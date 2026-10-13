import test from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_DRIVE_MIN, DEFAULT_RADIUS_MI, driveLabel, filterByDrive } from '../composables/forayDistance.ts'
import { BUILT_UP_SHARE, builtUpShare, parseBuiltUp, passesBuiltUp } from '../composables/forayPlanner.ts'
import { isEdible } from '../composables/edibles.ts'
import { placeKey, placeLabel, reverseUrl } from '../composables/placeName.ts'
import { parseDriveQuery, toMultiPolygon, valhallaUrl } from '../netlify/lib/drive-time.mjs'
import { handleDriveTime } from '../netlify/functions/drive-time.mjs'

const square = { type: 'Polygon', coordinates: [[[-106, 39], [-105, 39], [-105, 40], [-106, 40], [-106, 39]]] }

test('distance defaults: 25 mi and a 45 min drive', () => {
  assert.equal(DEFAULT_RADIUS_MI, 25)
  assert.equal(DEFAULT_DRIVE_MIN, 45)
  assert.equal(driveLabel(45), '45 min drive')
  assert.equal(driveLabel(60), '1 hr drive')
  assert.equal(driveLabel(90), '1½ hr drive')
  assert.equal(driveLabel(0), 'Any drive time')
})

test('filterByDrive keeps places inside the drive-time area; no area passes all', () => {
  const geom = toMultiPolygon(square)
  const cells = [{ key: 'in', lat: 39.5, lon: -105.5 }, { key: 'out', lat: 39.5, lon: -107.9 }]
  assert.deepEqual(filterByDrive(cells, geom).map((c) => c.key), ['in'])
  assert.equal(filterByDrive(cells, null).length, 2)
})

test('toMultiPolygon: Valhalla and ORS FeatureCollections, junk', () => {
  const fc = { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: square }, { type: 'Feature', geometry: square }] }
  assert.equal(toMultiPolygon(fc).coordinates.length, 2)
  assert.equal(toMultiPolygon({ type: 'MultiPolygon', coordinates: [square.coordinates] }).type, 'MultiPolygon')
  assert.equal(toMultiPolygon(null), null)
  assert.equal(toMultiPolygon({ type: 'FeatureCollection', features: [{ geometry: { type: 'LineString', coordinates: [] } }] }), null)
})

test('parseDriveQuery validates and rounds', () => {
  const q = parseDriveQuery(new URLSearchParams('lat=40.01234&lon=-105.27891&min=45'))
  assert.deepEqual(q, { lat: 40.012, lon: -105.279, minutes: 45 })
  assert.throws(() => parseDriveQuery(new URLSearchParams('lat=x&lon=1&min=45')))
  assert.throws(() => parseDriveQuery(new URLSearchParams('lat=40&lon=-105&min=500')))
  assert.match(valhallaUrl(q), /valhalla1\.openstreetmap\.de\/isochrone\?json=/)
})

test('drive-time function: ORS with a key, Valhalla without, 502 when neither answers', async () => {
  const fc = { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: square }] }
  const req = () => new Request('http://x/.netlify/functions/drive-time?lat=40&lon=-105&min=45')
  const calls = []
  const ok = async (url) => { calls.push(url); return fc }
  let r = await (await handleDriveTime(req(), { ORS_API_KEY: 'k' }, ok)).json()
  assert.equal(r.source, 'ors'); assert.equal(r.geom.type, 'MultiPolygon')
  r = await (await handleDriveTime(req(), {}, ok)).json()
  assert.equal(r.source, 'valhalla')
  // ORS failing falls through to Valhalla.
  r = await (await handleDriveTime(req(), { ORS_API_KEY: 'k' }, async (url) => (String(url).includes('openrouteservice') ? null : fc))).json()
  assert.equal(r.source, 'valhalla')
  const none = await handleDriveTime(req(), {}, async () => null)
  assert.equal(none.status, 502)
  assert.equal((await handleDriveTime(new Request('http://x/?lat=1'), {}, ok)).status, 400)
})

test('built-up toggle: skip towns, everywhere, towns only', () => {
  const town = builtUpShare(new Map([['Built-up', 6], ['Tree cover', 4]]))
  const forest = builtUpShare(new Map([['Built-up', 1], ['Tree cover', 9]]))
  assert.equal(town, 0.6); assert.ok(forest < BUILT_UP_SHARE)
  assert.equal(builtUpShare(new Map()), null)
  assert.ok(!passesBuiltUp(town, 'exclude') && passesBuiltUp(forest, 'exclude') && passesBuiltUp(null, 'exclude'))
  assert.ok(passesBuiltUp(town, 'all') && passesBuiltUp(forest, 'all'))
  assert.ok(passesBuiltUp(town, 'only') && !passesBuiltUp(forest, 'only') && !passesBuiltUp(null, 'only'))
  assert.equal(parseBuiltUp('only'), 'only'); assert.equal(parseBuiltUp(null), 'exclude')
})

test('edibles: listed species and whole genera only', () => {
  assert.ok(isEdible('Boletus edulis'))
  assert.ok(isEdible('Cantharellus roseocanus'))
  assert.ok(isEdible('Morchella'))
  assert.ok(!isEdible('Boletus luridiformis'))
  assert.ok(!isEdible('Amanita muscaria'))
  assert.ok(!isEdible(''))
})

test('placeLabel: feature near a town, town alone, nothing', () => {
  assert.equal(placeLabel({ name: 'Brainard Lake', address: { village: 'Ward', county: 'Boulder County' } }), 'Brainard Lake, near Ward')
  assert.equal(placeLabel({ name: '', address: { town: 'Nederland' } }), 'near Nederland')
  assert.equal(placeLabel({ name: 'Ward', address: { village: 'Ward' } }), 'Ward')
  assert.equal(placeLabel({ error: 'Unable to geocode' }), null)
  assert.equal(placeKey(40.01234, -105.2789), '40.012,-105.279')
  assert.match(reverseUrl(40, -105), /reverse\?format=jsonv2&zoom=14&lat=40\.00000&lon=-105\.00000/)
})
