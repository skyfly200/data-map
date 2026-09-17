/* Blend modes, solo, and moving a layer through the stack.
 *
 * All three are about what the map draws, and all three are easy to get subtly
 * wrong in ways that look like a rendering bug: a default that fires when it
 * should not, a solo that loses the stack, a "send to top" that silently does
 * nothing at the top.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  BLEND_MODES, NORMAL, blendLabel, drawnKeys, effectiveBlend, isBlendMode, reorderStack,
} from '../composables/blendModes.js'

// ── The catalogue ────────────────────────────────────────────────────────────

test('every mode is a real CSS value, named and described', () => {
  const css = new Set(['normal', 'multiply', 'screen', 'overlay', 'darken', 'lighten',
    'color-dodge', 'color-burn', 'hard-light', 'soft-light', 'difference', 'exclusion',
    'hue', 'saturation', 'color', 'luminosity'])
  for (const m of BLEND_MODES) {
    assert.ok(css.has(m.key), `${m.key} is not a mix-blend-mode value`)
    assert.ok(m.label && m.note, `${m.key} has no label or note`)
  }
  assert.equal(BLEND_MODES[0].key, NORMAL, 'normal is not first, so it is not the obvious default')
  const keys = BLEND_MODES.map((m) => m.key)
  assert.equal(new Set(keys).size, keys.length)
})

test('an unknown mode is not one of ours', () => {
  assert.ok(isBlendMode('multiply'))
  assert.ok(!isBlendMode('hue'), 'colour-wheel modes were left out on purpose')
  assert.ok(!isBlendMode('url(#x)'))
  assert.ok(!isBlendMode(''))
  assert.ok(!isBlendMode(null))
  assert.equal(blendLabel('multiply'), 'Multiply')
  assert.equal(blendLabel('nonsense'), 'Normal')
})

// ── Which mode a layer draws with ────────────────────────────────────────────

test('a layer set by hand keeps its mode, stack or no stack', () => {
  const overrides = { hillshade: 'multiply' }
  assert.equal(effectiveBlend('hillshade', { overrides, drawn: 1 }), 'multiply')
  assert.equal(effectiveBlend('hillshade', { overrides, drawn: 4, fallback: 'screen' }), 'multiply')
})

test('the stack default applies only when there is a stack', () => {
  // One layer over the basemap is not "several layers drawn together". A
  // default that fired there would change every layer anyone switched on.
  assert.equal(effectiveBlend('slope', { fallback: 'multiply', drawn: 1 }), NORMAL)
  assert.equal(effectiveBlend('slope', { fallback: 'multiply', drawn: 0 }), NORMAL)
  assert.equal(effectiveBlend('slope', { fallback: 'multiply', drawn: 2 }), 'multiply')
})

test('a stored mode that is no longer offered behaves as if it were absent', () => {
  // Preferences outlive the list they were chosen from, and they reach the
  // browser as a style value, so an unknown one must not be passed through.
  assert.equal(effectiveBlend('a', { overrides: { a: 'plaid' }, drawn: 3, fallback: 'screen' }),
    'screen')
  assert.equal(effectiveBlend('a', { overrides: { a: 'plaid' }, drawn: 1 }), NORMAL)
  assert.equal(effectiveBlend('a', { fallback: 'plaid', drawn: 3 }), NORMAL)
})

test('no arguments at all is still normal rather than undefined', () => {
  assert.equal(effectiveBlend('a'), NORMAL)
  assert.equal(effectiveBlend(), NORMAL)
})

// ── Solo ─────────────────────────────────────────────────────────────────────

test('solo draws one layer and leaves the rest switched on', () => {
  const active = ['fire', 'soil', 'slope']
  assert.deepEqual(drawnKeys(active, 'soil'), ['soil'])
  // The caller's list is not touched, because it is the stack the viewer built
  // and solo is a way of looking at it, not an edit to it.
  assert.deepEqual(active, ['fire', 'soil', 'slope'])
})

test('no solo draws everything', () => {
  assert.deepEqual(drawnKeys(['a', 'b'], ''), ['a', 'b'])
  assert.deepEqual(drawnKeys(['a', 'b']), ['a', 'b'])
  assert.deepEqual(drawnKeys([], 'a'), [])
})

test('a solo on a layer that is no longer on draws the rest, not nothing', () => {
  // Switching the soloed layer off from the checkbox list would otherwise
  // leave an empty map with three layers ticked.
  assert.deepEqual(drawnKeys(['a', 'b'], 'gone'), ['a', 'b'])
})

// ── Order ────────────────────────────────────────────────────────────────────

test('a layer moves one place at a time', () => {
  assert.deepEqual(reorderStack(['a', 'b', 'c'], 'b', -1), ['b', 'a', 'c'])
  assert.deepEqual(reorderStack(['a', 'b', 'c'], 'b', 1), ['a', 'c', 'b'])
})

test('a layer goes to the top or the bottom in one step', () => {
  // With eight layers on, eight presses of "up" is a counting exercise.
  assert.deepEqual(reorderStack(['a', 'b', 'c', 'd'], 'd', 'top'), ['d', 'a', 'b', 'c'])
  assert.deepEqual(reorderStack(['a', 'b', 'c', 'd'], 'a', 'bottom'), ['b', 'c', 'd', 'a'])
})

test('moving past either end changes nothing', () => {
  const order = ['a', 'b', 'c']
  assert.deepEqual(reorderStack(order, 'a', -1), order)
  assert.deepEqual(reorderStack(order, 'c', 1), order)
  assert.deepEqual(reorderStack(order, 'a', 'top'), order)
  assert.deepEqual(reorderStack(order, 'c', 'bottom'), order)
})

test('a key that is not in the stack changes nothing', () => {
  assert.deepEqual(reorderStack(['a', 'b'], 'z', 'top'), ['a', 'b'])
  assert.deepEqual(reorderStack([], 'a', 1), [])
})

test('reordering never mutates the order it was given', () => {
  const order = ['a', 'b', 'c']
  reorderStack(order, 'a', 'bottom'); reorderStack(order, 'c', -1)
  assert.deepEqual(order, ['a', 'b', 'c'])
})

test('a nonsense delta leaves the stack alone rather than dropping a layer', () => {
  const order = ['a', 'b', 'c']
  assert.deepEqual(reorderStack(order, 'b', NaN), order)
  assert.deepEqual(reorderStack(order, 'b', undefined), order)
  assert.deepEqual(reorderStack(order, 'b', 'sideways'), order)
})
