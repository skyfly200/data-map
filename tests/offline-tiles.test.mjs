import test from 'node:test'
import assert from 'node:assert/strict'

import {
  AVG_TILE_BYTES, MAX_EXTRA_ZOOM, estimateSave, formatBytes, tileFor, tileUrl, tilesInBounds,
} from '../composables/offlineTiles.js'

test('zoom 0 is one tile and the origin is the north-west corner', () => {
  assert.deepEqual(tileFor(0, 0, 0), [0, 0])
  assert.deepEqual(tileFor(85, -180, 1), [0, 0])   // north-west
  assert.deepEqual(tileFor(-85, 179.9, 1), [1, 1]) // south-east
})

test('a known coordinate lands on its known tile', () => {
  // Denver at z10. Computed independently from the standard slippy-map
  // formula rather than read off this implementation, so the test can fail.
  assert.deepEqual(tileFor(39.7392, -104.9903, 10), [213, 388])
  // And the same place one level out is that tile halved.
  assert.deepEqual(tileFor(39.7392, -104.9903, 9), [106, 194])
})

test('coordinates past the Mercator limit clamp instead of going infinite', () => {
  // tan(90°) is infinite; a pole must still produce a real tile index.
  for (const lat of [90, -90, 89.99, -89.99]) {
    const [x, y] = tileFor(lat, 10, 5)
    assert.ok(Number.isFinite(x) && Number.isFinite(y), `lat ${lat} gave ${x},${y}`)
    assert.ok(y >= 0 && y < 32, `lat ${lat} gave y=${y}`)
  }
})

test('a bounding box yields every tile covering it, at each zoom', () => {
  const bounds = { north: 40.1, south: 39.9, east: -104.9, west: -105.1 }
  const one = tilesInBounds(bounds, 10, 10)
  assert.ok(one.length >= 1)
  assert.ok(one.every((t) => t.z === 10))

  // Every corner of the box must be inside the returned set, or a saved area
  // has holes at its edges — which is exactly where you notice them.
  const keys = new Set(one.map((t) => `${t.x}/${t.y}`))
  for (const [lat, lon] of [[40.1, -105.1], [40.1, -104.9], [39.9, -105.1], [39.9, -104.9]]) {
    const [x, y] = tileFor(lat, lon, 10)
    assert.ok(keys.has(`${x}/${y}`), `corner ${lat},${lon} not covered`)
  }
})

test('each extra zoom level roughly quadruples the tile count', () => {
  const bounds = { north: 40.5, south: 39.5, east: -104.5, west: -105.5 }
  const counts = [0, 1, 2, 3].map((extra) => tilesInBounds(bounds, 9, 9 + extra).length)
  for (let i = 1; i < counts.length; i += 1) {
    const added = counts[i] - counts[i - 1]
    const prev = counts[i - 1] - (counts[i - 2] ?? 0)
    if (i > 1) assert.ok(added >= prev * 3, `level ${i}: added ${added} vs previous ${prev}`)
  }
  // Which is the whole reason the count is shown before anything downloads.
  assert.ok(counts.at(-1) > counts[0] * 10)
})

test('no duplicate tiles within a save', () => {
  const tiles = tilesInBounds({ north: 41, south: 39, east: -104, west: -106 }, 8, 11)
  const keys = new Set(tiles.map((t) => `${t.z}/${t.x}/${t.y}`))
  assert.equal(keys.size, tiles.length)
})

test('a url template is filled with the tile, and {s} is pinned', () => {
  const url = tileUrl('https://{s}.tile.example/{z}/{x}/{y}.png', { x: 1, y: 2, z: 3 }, 'abc')
  assert.equal(url, 'https://a.tile.example/3/1/2.png')
  // Pinned, not spread: Leaflet picks a subdomain per tile from the same list,
  // so a tile saved under one host and requested from another is a cache miss
  // on everything that was saved.
  assert.equal(tileUrl('https://{s}.t/{z}/{x}/{y}.png', { x: 1, y: 2, z: 3 }, 'abc'),
    tileUrl('https://{s}.t/{z}/{x}/{y}.png', { x: 1, y: 2, z: 3 }, 'abc'))
})

test('y-before-x templates are filled correctly too', () => {
  // GIBS and ArcGIS put the row before the column; swapping them silently
  // draws the wrong part of the world.
  assert.equal(tileUrl('https://e.test/{z}/{y}/{x}', { x: 7, y: 9, z: 4 }), 'https://e.test/4/9/7')
})

test('the estimate scales with layers as well as tiles', () => {
  assert.deepEqual(estimateSave(100, 1), { tiles: 100, bytes: 100 * AVG_TILE_BYTES })
  assert.deepEqual(estimateSave(100, 3), { tiles: 300, bytes: 300 * AVG_TILE_BYTES })
  assert.deepEqual(estimateSave(0, 4), { tiles: 0, bytes: 0 })
})

test('byte counts read as sizes a person can judge', () => {
  assert.equal(formatBytes(0), '0 MB')
  assert.equal(formatBytes(-5), '0 MB')
  assert.equal(formatBytes(NaN), '0 MB')
  assert.equal(formatBytes(500), '1 KB')
  assert.equal(formatBytes(1024 * 400), '400 KB')
  assert.equal(formatBytes(1024 * 1024 * 2.5), '2.5 MB')
  assert.equal(formatBytes(1024 * 1024 * 48), '48 MB')
  assert.equal(formatBytes(1024 * 1024 * 1024 * 2), '2.0 GB')
})

test('the extra-zoom cap keeps a save from becoming a download', () => {
  // Three levels past a city view is a few hundred tiles; six is tens of
  // thousands, which is not something anyone means to start.
  assert.equal(MAX_EXTRA_ZOOM, 3)
  const view = { north: 40.05, south: 39.95, east: -104.95, west: -105.05 }
  assert.ok(tilesInBounds(view, 11, 11 + MAX_EXTRA_ZOOM).length < 1000)
})
