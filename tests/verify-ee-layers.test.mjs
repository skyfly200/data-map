/* The layer-verification harness picks the right layers to check.
 *
 * The rendering itself needs Earth Engine credentials and is the manual pass the
 * script exists to run; what is testable here is that it enumerates the whole
 * catalogue and that the name filter narrows it.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { layersToCheck } from '../scripts/verify_ee_layers.mjs'
import { EE_LAYER_KEYS } from '../netlify/lib/ee-tile-layers.mjs'

test('no filter checks every catalogue layer', () => {
  assert.deepEqual(layersToCheck(), EE_LAYER_KEYS)
  assert.deepEqual(layersToCheck(''), EE_LAYER_KEYS)
})

test('a filter narrows to matching keys, case-insensitively', () => {
  const soil = layersToCheck('soil')
  assert.ok(soil.length)
  assert.ok(soil.every((k) => k.includes('soil')))
  assert.deepEqual(layersToCheck('SOIL'), soil)
})

test('a filter that matches nothing returns nothing', () => {
  assert.deepEqual(layersToCheck('no-such-layer-xyz'), [])
})
