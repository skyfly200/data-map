import test from 'node:test'
import assert from 'node:assert/strict'

import {
  CELL_SHAPES, cellAt, cellKeyAt, hexCentre, hexIndex, hexPolygon, hexRadius,
} from '../composables/gridCells.js'

const SIZE = 0.05

test('both shapes are offered', () => {
  assert.deepEqual(CELL_SHAPES.map((s) => s.value).sort(), ['hex', 'square'])
})

test('square cells tile the plane on multiples of the size', () => {
  const c = cellAt(-105.73, 39.52, 0.1, 'square')
  assert.equal(c.key, '395:-1058')
  assert.ok(Math.abs(c.lat0 - 39.5) < 1e-9)
  assert.ok(Math.abs(c.lon1 - -105.7) < 1e-9)
  assert.equal(c.polygon.length, 4)
})

test('a hex is sized to the same area as the square it replaces', () => {
  const r = hexRadius(SIZE)
  const area = ((3 * Math.sqrt(3)) / 2) * r * r
  // Within a rounding error of size², so switching shape re-bins at the same
  // resolution rather than silently coarsening or sharpening the map.
  assert.ok(Math.abs(area - SIZE * SIZE) < 1e-12, `hex area ${area} vs ${SIZE * SIZE}`)
})

test('every point is assigned to its nearest hex centre', () => {
  // The rounding shortcut alone lands in the wrong hex near the slanted edges,
  // which is the whole reason for the neighbour comparison. Check it against
  // brute force: the assigned centre must be the closest of all candidates.
  const r = hexRadius(SIZE)
  let checked = 0
  for (let i = 0; i < 400; i += 1) {
    // Deterministic scatter across a couple of cells' worth of ground.
    const lon = -105.7 + ((i * 37) % 211) / 211 * SIZE * 3
    const lat = 39.5 + ((i * 91) % 173) / 173 * SIZE * 3

    const [pi, pj] = hexIndex(lon, lat, SIZE)
    const [cx, cy] = hexCentre(pi, pj, SIZE)
    const d = Math.hypot(lon - cx, lat - cy)

    // Every centre within two rings of the assignment.
    for (let dj = -2; dj <= 2; dj += 1) {
      for (let di = -2; di <= 2; di += 1) {
        const [ox, oy] = hexCentre(pi + di, pj + dj, SIZE)
        assert.ok(Math.hypot(lon - ox, lat - oy) >= d - 1e-12,
          `a nearer centre exists for ${lon},${lat}`)
      }
    }
    // And it is inside the hex: no point is further from its centre than the
    // circumradius.
    assert.ok(d <= r + 1e-12, `point ${d} from centre, circumradius ${r}`)
    checked += 1
  }
  assert.equal(checked, 400)
})

test('the hex key is stable across the cell and changes between cells', () => {
  const [cx, cy] = hexCentre(3, 4, SIZE)
  const centreKey = cellKeyAt(cx, cy, SIZE, 'hex')
  // A nudge well inside the cell keeps the key…
  const r = hexRadius(SIZE)
  assert.equal(cellKeyAt(cx + r * 0.2, cy + r * 0.2, SIZE, 'hex'), centreKey)
  // …and a step to the next row does not.
  assert.notEqual(cellKeyAt(cx, cy + r * 1.5, SIZE, 'hex'), centreKey)
})

test('hex and square keys cannot collide', () => {
  // They share one index Map on the map component, so a hex key that happened
  // to look like a square key would silently mix two grids.
  assert.ok(cellKeyAt(-105.7, 39.5, SIZE, 'hex').startsWith('h'))
  assert.ok(!cellKeyAt(-105.7, 39.5, SIZE, 'square').startsWith('h'))
})

test('a hex polygon closes and is regular', () => {
  const pts = hexPolygon(-105.7, 39.5, SIZE)
  assert.equal(pts.length, 6)
  const r = hexRadius(SIZE)
  for (const [lat, lon] of pts) {
    assert.ok(Math.abs(Math.hypot(lon + 105.7, lat - 39.5) - r) < 1e-12)
  }
  // Pointy-top: the first vertex is directly above the centre.
  assert.ok(Math.abs(pts[0][1] - -105.7) < 1e-12)
  assert.ok(pts[0][0] > 39.5)
})

test('every cell carries a bounding box around its outline', () => {
  // The arrow overlay sizes itself from lat0/lat1, so a shape that skipped them
  // would draw zero-length arrows rather than fail visibly.
  for (const shape of ['hex', 'square']) {
    const c = cellAt(-105.73, 39.52, SIZE, shape)
    for (const [lat, lon] of c.polygon) {
      assert.ok(lat >= c.lat0 - 1e-12 && lat <= c.lat1 + 1e-12, `${shape} lat outside bbox`)
      assert.ok(lon >= c.lon0 - 1e-12 && lon <= c.lon1 + 1e-12, `${shape} lon outside bbox`)
    }
    assert.ok(c.lat > c.lat0 && c.lat < c.lat1)
    assert.ok(c.lon > c.lon0 && c.lon < c.lon1)
  }
})

test('hexes partition the plane — no point lands in two cells, none in none', () => {
  const seen = new Map()
  for (let i = 0; i < 2000; i += 1) {
    const lon = -106 + ((i * 7919) % 10007) / 10007
    const lat = 39 + ((i * 6271) % 9973) / 9973
    const key = cellKeyAt(lon, lat, SIZE, 'hex')
    assert.equal(typeof key, 'string')
    seen.set(key, (seen.get(key) || 0) + 1)
  }
  // A 1°×1° patch at this cell size must produce many distinct cells, not one
  // bucket swallowing everything.
  assert.ok(seen.size > 100, `only ${seen.size} cells`)
})
