/* Saved areas: named places kept for reading with no signal.
 *
 * The arithmetic that matters here is not "which tiles cover a box" — that is
 * tested next door in offline-tiles.test.mjs — but what happens when saved
 * areas interact. Two of them overlap; one is deleted; the other must not go
 * blank where they met. And an Earth Engine layer's URL expires within hours,
 * so a tile filed under that URL is unreachable long before anybody walks into
 * the woods to read it.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  MAX_AREA_TILES, TILE_KEY_PREFIX,
  areaContains, areaKeys, areaTileCount, cacheKeyFor, countTilesInBounds, describeArea,
  eeMapId, estimateAreaBytes, isVolatileTemplate, keysToDrop, makeArea, newAreaId,
  normaliseSource, saveTargets, suggestAreaName, tilesInBounds,
} from '../composables/offlineTiles.js'

const OSM = 'https://tile.example/{z}/{x}/{y}.png'
const EE = 'https://earthengine.googleapis.com/v1/projects/p/maps/MAPID-ABC123/tiles/{z}/{x}/{y}'

const boxAt = (lat, lon, d = 0.05) => ({
  north: lat + d, south: lat - d, east: lon + d, west: lon - d,
})

// ── Counting without allocating ──────────────────────────────────────────────

test('the count matches the list it avoids building', () => {
  // The slider recomputes this on every frame, so it must not build the list —
  // but it must agree with it exactly, or the number shown is not the number
  // downloaded.
  for (const bounds of [
    boxAt(39.74, -104.99),
    boxAt(0, 0, 1),
    boxAt(-33.9, 151.2, 0.2),
    { north: 60, south: -60, east: 170, west: -170 },
  ]) {
    for (const [lo, hi] of [[8, 8], [8, 11], [3, 6]]) {
      assert.equal(countTilesInBounds(bounds, lo, hi), tilesInBounds(bounds, lo, hi).length,
        `${JSON.stringify(bounds)} at ${lo}-${hi}`)
    }
  }
})

test('the per-area ceiling is low enough to be reachable before the data is spent', () => {
  assert.ok(MAX_AREA_TILES > 0)
  // A normal view three levels deep must sit well under it, or the cap is
  // firing on ordinary use.
  const area = makeArea({ bounds: boxAt(39.74, -104.99), minZoom: 11, maxZoom: 14, sources: [OSM] })
  assert.ok(areaTileCount(area) < MAX_AREA_TILES / 10, `${areaTileCount(area)} tiles`)
})

// ── Volatile sources ─────────────────────────────────────────────────────────

test('an Earth Engine template is recognised as expiring, an ordinary one is not', () => {
  assert.equal(isVolatileTemplate(EE), true)
  assert.equal(isVolatileTemplate(OSM), false)
  assert.equal(isVolatileTemplate(''), false)
  assert.equal(isVolatileTemplate(null), false)
})

test('the expiring part of an Earth Engine url can be picked out', () => {
  assert.equal(eeMapId(EE), 'MAPID-ABC123')
  assert.equal(eeMapId(EE.replace('{z}/{x}/{y}', '5/3/2')), 'MAPID-ABC123')
  assert.equal(eeMapId(OSM), '')
})

test('an ordinary tile is filed under its own url', () => {
  // No indirection for the common case: the worker serves it by matching the
  // request unchanged.
  const key = cacheKeyFor(OSM, { x: 1, y: 2, z: 3 })
  assert.equal(key, 'https://tile.example/3/1/2.png')
})

test('an expiring tile is filed under the layer, not the token', () => {
  const tile = { x: 1, y: 2, z: 3 }
  const now = cacheKeyFor({ template: EE, id: 'custom:tree-cover' }, tile, 'https://app.test')
  const later = cacheKeyFor(
    { template: EE.replace('MAPID-ABC123', 'MAPID-ROTATED'), id: 'custom:tree-cover' },
    tile, 'https://app.test',
  )
  // The whole point: an hour later, the same tile of the same layer is the same
  // cache entry. Keyed by URL these would differ, and everything saved would
  // read as missing while the layer drew blank over ground it had tiles for.
  assert.equal(now, later)
  assert.ok(now.includes(TILE_KEY_PREFIX))
  assert.ok(now.includes(encodeURIComponent('custom:tree-cover')))
  assert.ok(now.endsWith('/3/1/2'))
})

test('a source is accepted as a bare string or as an identified pair', () => {
  assert.deepEqual(normaliseSource(OSM), { template: OSM, id: OSM, volatile: false })
  assert.deepEqual(normaliseSource({ template: EE, id: 'ee:fire' }),
    { template: EE, id: 'ee:fire', volatile: true })
  // An explicit flag wins, so a future volatile host needs no regex change.
  assert.equal(normaliseSource({ template: OSM, id: 'x', volatile: true }).volatile, true)
})

test('a volatile source keeps its identity across a re-save, and its url does not matter', () => {
  const first = makeArea({ bounds: boxAt(39.74, -104.99), minZoom: 10, maxZoom: 10,
    sources: [{ template: EE, id: 'custom:tree-cover' }] })
  const again = makeArea({ ...first,
    sources: [{ template: EE.replace('MAPID-ABC123', 'NEW'), id: 'custom:tree-cover' }] })
  assert.deepEqual([...areaKeys(first)], [...areaKeys(again)])
})

// ── What a save actually fetches ─────────────────────────────────────────────

test('a save fetches every tile of every layer, and files each where it belongs', () => {
  const area = makeArea({
    bounds: boxAt(39.74, -104.99), minZoom: 10, maxZoom: 10,
    sources: [OSM, { template: EE, id: 'ee:fire' }],
  })
  const targets = saveTargets(area, 'https://app.test')
  assert.equal(targets.length, areaTileCount(area))

  // Every fetch goes to a real URL; only where it is FILED differs.
  assert.ok(targets.every((t) => t.url.startsWith('https://')))
  assert.ok(targets.some((t) => t.key === t.url), 'ordinary tiles keep their url as the key')
  assert.ok(targets.some((t) => t.key.includes(TILE_KEY_PREFIX)), 'expiring tiles are re-keyed')
  assert.ok(targets.filter((t) => t.key.includes(TILE_KEY_PREFIX))
    .every((t) => t.url.includes('earthengine')))
})

test('a source with no template is dropped rather than fetched as ""', () => {
  const area = makeArea({ bounds: boxAt(1, 1), minZoom: 5, maxZoom: 5, sources: [OSM, { id: 'x' }, ''] })
  assert.equal(area.sources.length, 1)
  assert.ok(saveTargets(area).every((t) => t.url.startsWith('https://')))
})

// ── Deleting one area without blanking another ───────────────────────────────

test('deleting an area leaves the tiles another one still needs', () => {
  // Two boxes that overlap. This is the case that matters: saved "the ridge"
  // and "the saddle" over the same valley, delete one, and the other must not
  // develop a hole where they met.
  const a = makeArea({ id: 'a', bounds: boxAt(39.74, -104.99, 0.1), minZoom: 11, maxZoom: 12, sources: [OSM] })
  const b = makeArea({ id: 'b', bounds: boxAt(39.76, -104.97, 0.1), minZoom: 11, maxZoom: 12, sources: [OSM] })

  const shared = [...areaKeys(a)].filter((k) => areaKeys(b).has(k))
  assert.ok(shared.length > 0, 'these boxes should overlap, or the test proves nothing')

  const dropped = new Set(keysToDrop(a, [a, b]))
  for (const key of shared) assert.ok(!dropped.has(key), `${key} is still needed by b`)
  // And everything of a's that b does not want does go.
  for (const key of areaKeys(a)) {
    if (!areaKeys(b).has(key)) assert.ok(dropped.has(key), `${key} should have been dropped`)
  }
})

test('the last area standing gives up all of its tiles', () => {
  const a = makeArea({ id: 'a', bounds: boxAt(39.74, -104.99), minZoom: 11, maxZoom: 11, sources: [OSM] })
  assert.equal(keysToDrop(a, [a]).length, areaKeys(a).size)
  assert.equal(keysToDrop(a, []).length, areaKeys(a).size)
})

test('an area nested inside a bigger one gives up nothing', () => {
  const big = makeArea({ id: 'big', bounds: boxAt(39.74, -104.99, 0.4), minZoom: 10, maxZoom: 12, sources: [OSM] })
  const small = makeArea({ id: 'small', bounds: boxAt(39.74, -104.99, 0.05), minZoom: 10, maxZoom: 12, sources: [OSM] })
  assert.deepEqual(keysToDrop(small, [small, big]), [])
  // The other way round, the big one keeps only what the small one covers.
  assert.ok(keysToDrop(big, [big, small]).length > 0)
})

test('areas sharing a place but not a layer do not protect each other', () => {
  const a = makeArea({ id: 'a', bounds: boxAt(39.74, -104.99), minZoom: 11, maxZoom: 11, sources: [OSM] })
  const b = makeArea({ id: 'b', bounds: boxAt(39.74, -104.99), minZoom: 11, maxZoom: 11,
    sources: ['https://other.example/{z}/{x}/{y}.png'] })
  assert.equal(keysToDrop(a, [a, b]).length, areaKeys(a).size)
})

// ── Records ──────────────────────────────────────────────────────────────────

test('an area gets an id, a name and a time without being given any', () => {
  const area = makeArea({ bounds: boxAt(39.74, -104.99), minZoom: 10, maxZoom: 12, sources: [OSM] })
  assert.match(area.id, /^a[a-z0-9]+$/)
  assert.ok(area.name.length)
  assert.ok(Date.parse(area.savedAt) > 0)
})

test('ids do not collide across a burst of saves', () => {
  const ids = new Set(Array.from({ length: 500 }, () => newAreaId()))
  assert.equal(ids.size, 500)
})

test('an unnamed area is named for where it is, not what number it is', () => {
  // A list of saved places is read weeks later. "Area 3" says nothing about
  // which one is the ridge you meant.
  assert.equal(suggestAreaName(boxAt(39.74, -104.99)), '39.74°N, 104.99°W')
  assert.equal(suggestAreaName(boxAt(-33.87, 151.21)), '33.87°S, 151.21°E')
  assert.equal(suggestAreaName(null), 'Saved area')
  // A name that is only whitespace is not a name.
  assert.equal(makeArea({ name: '   ', bounds: boxAt(1, 2), minZoom: 1, maxZoom: 1 }).name,
    suggestAreaName(boxAt(1, 2)))
})

test('a reversed zoom range is repaired rather than saving nothing', () => {
  const area = makeArea({ bounds: boxAt(1, 2), minZoom: 12, maxZoom: 9, sources: [OSM] })
  assert.ok(area.maxZoom >= area.minZoom)
  assert.ok(areaTileCount(area) > 0)
})

test('bounds are rounded, so saving the same place twice looks the same', () => {
  const a = makeArea({ bounds: { north: 40.000000001, south: 39.9, east: -104.9, west: -105.1 },
    minZoom: 10, maxZoom: 10, sources: [OSM] })
  assert.equal(a.bounds.north, 40)
})

test('an area knows whether you are standing in it', () => {
  const area = makeArea({ bounds: boxAt(39.74, -104.99), minZoom: 10, maxZoom: 10, sources: [OSM] })
  assert.equal(areaContains(area, 39.74, -104.99), true)
  assert.equal(areaContains(area, 39.74, -104.8), false)
  assert.equal(areaContains(area, 0, 0), false)
  assert.equal(areaContains(null, 0, 0), false)
})

test('a description says what the area covers, in units a person walks in', () => {
  const area = makeArea({ bounds: boxAt(39.74, -104.99), minZoom: 10, maxZoom: 13, sources: [OSM] })
  const d = describeArea(area)
  assert.equal(d.zooms, 'zoom 10–13')
  assert.equal(d.layers, 1)
  assert.equal(d.tiles, areaTileCount(area))
  assert.equal(d.bytes, estimateAreaBytes(area))
  assert.match(d.extent, /km × .*km/)

  // A single level does not read as a range.
  assert.equal(describeArea(makeArea({ bounds: boxAt(1, 2), minZoom: 9, maxZoom: 9 })).zooms, 'zoom 9')
})

test('extent is scaled by latitude, so a polar box does not read as enormous', () => {
  // A degree of longitude is 111 km at the equator and nearly nothing at the
  // pole. Unscaled, every high-latitude area would claim a width it does not
  // have.
  const d = 0.5
  const equator = describeArea(makeArea({ bounds: boxAt(0, 0, d), minZoom: 8, maxZoom: 8 }))
  const arctic = describeArea(makeArea({ bounds: boxAt(70, 0, d), minZoom: 8, maxZoom: 8 }))
  const widthOf = (s) => Number(s.split(' × ')[0].replace(' km', ''))
  assert.ok(widthOf(arctic.extent) < widthOf(equator.extent) / 2,
    `${arctic.extent} should be much narrower than ${equator.extent}`)
})

test('the size of an area scales with its layers', () => {
  const one = makeArea({ bounds: boxAt(1, 2), minZoom: 9, maxZoom: 10, sources: [OSM] })
  const two = makeArea({ ...one, sources: [OSM, 'https://b.test/{z}/{x}/{y}.png'] })
  assert.equal(estimateAreaBytes(two), estimateAreaBytes(one) * 2)
})
