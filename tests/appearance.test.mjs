import test from 'node:test'
import assert from 'node:assert/strict'

import {
  PALETTES, SHAPE_SETS, ALL_SHAPES, UNCLUSTERED,
  categoryColor, categoryShape, colorFor, stableColor, overrideKey,
} from '../composables/useAppearance.js'

// These run against the module's default state. useAppearance()'s setters need
// a Nuxt runtime (localStorage, useState), so what is covered here is the pure
// mapping every view depends on: the same value must always resolve to the same
// mark, and the palettes themselves must be well-formed.

test('the same value always gets the same colour', () => {
  // This is the whole point of stableColor: a species keeps its colour across
  // the map, the charts and the table.
  assert.equal(stableColor('Amanita muscaria'), stableColor('Amanita muscaria'))
  assert.equal(categoryColor('species', 'Morchella'), categoryColor('species', 'Morchella'))
})

test('missing values fall to the neutral colour rather than a palette slot', () => {
  for (const empty of [null, undefined, '']) {
    assert.equal(stableColor(empty), UNCLUSTERED)
    assert.equal(categoryColor('species', empty), UNCLUSTERED)
  }
  assert.equal(colorFor(null), UNCLUSTERED)
  assert.equal(colorFor(undefined), UNCLUSTERED)
  assert.equal(colorFor(NaN), UNCLUSTERED)
})

test('cluster ids index the palette and wrap past its end', () => {
  const first = colorFor(0)
  const wrapped = colorFor(PALETTES[0].colors.length)
  assert.equal(first, wrapped)
})

test('cluster labels are read through their prefix', () => {
  // Live clusters are labelled "K3"/"C3"; they must land on the same colour as
  // the pipeline cluster 3 so the two legends agree.
  assert.equal(categoryColor('cluster', 3), colorFor(3))
  assert.equal(categoryColor('live_cluster', 'K3'), colorFor(3))
  assert.equal(categoryColor('cluster', 'C3'), colorFor(3))
})

test('an unparseable cluster label is neutral, not a wrong colour', () => {
  assert.equal(categoryColor('cluster', 'not-a-number'), UNCLUSTERED)
})

test('shapes rotate through the active set by index', () => {
  const set = SHAPE_SETS.find((s) => s.key === 'all').shapes
  assert.equal(categoryShape('species', 'a', 0), set[0])
  assert.equal(categoryShape('species', 'b', 1), set[1])
  // Wraps rather than running off the end.
  assert.equal(categoryShape('species', 'c', set.length), set[0])
})

test('override keys are unique per field and value', () => {
  assert.equal(overrideKey('species', 'Morchella'), 'species:Morchella')
  assert.notEqual(overrideKey('species', 'x'), overrideKey('genus', 'x'))
})

test('every palette is well-formed and distinct', () => {
  assert.ok(PALETTES.length >= 2)
  for (const p of PALETTES) {
    assert.ok(p.colors.length >= 8, `${p.key} needs at least 8 colours`)
    for (const c of p.colors) {
      assert.match(c, /^#[0-9a-f]{6}$/i, `${p.key} has a non-hex colour: ${c}`)
    }
    // Repeats inside one palette would make two categories indistinguishable.
    assert.equal(new Set(p.colors).size, p.colors.length, `${p.key} repeats a colour`)
    // The neutral "no value" colour must never be a category colour.
    assert.ok(!p.colors.includes(UNCLUSTERED), `${p.key} collides with UNCLUSTERED`)
  }
})

test('every shape set draws from the known shapes', () => {
  for (const s of SHAPE_SETS) {
    assert.ok(s.shapes.length >= 1)
    for (const shape of s.shapes) {
      assert.ok(ALL_SHAPES.includes(shape), `${s.key} names an unknown shape: ${shape}`)
    }
  }
})
