import test from 'node:test'
import assert from 'node:assert/strict'
import {
  MAX_EXPORT_PIXELS, defaultExportScale, exportPixels,
} from '../composables/modelExport.ts'

// Roughly Colorado.
const colorado = { west: -109.05, south: 36.99, east: -102.04, north: 41.0 }

test('pixel count follows the region and the scale', () => {
  const at1km = exportPixels(colorado, 1000)
  assert.ok(at1km > 200_000 && at1km < 300_000, String(at1km))
  assert.ok(exportPixels(colorado, 90) > MAX_EXPORT_PIXELS)
  assert.equal(exportPixels(null, 1000), null)
})

test('the default is the finest scale that fits', () => {
  assert.equal(defaultExportScale(colorado), 250)
  assert.equal(defaultExportScale({ west: -105.3, south: 39.9, east: -105.1, north: 40.1 }), 90)
  assert.equal(defaultExportScale(null), 1000)
})
