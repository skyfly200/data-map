import { test } from 'node:test'
import assert from 'node:assert/strict'

import { PALETTES, shade } from '../composables/useAppearance.js'

// The map's categorical legend shows up to this many rows.
const LEGEND_CAP = 12

test('every palette can colour a full legend without repeating', () => {
  // An eight-colour palette made repeated swatches unavoidable in a key whose
  // whole job is telling categories apart. No hashing scheme can fix a palette
  // smaller than the legend.
  for (const p of PALETTES) {
    assert.ok(p.colors.length >= LEGEND_CAP,
      `${p.key} has ${p.colors.length} colours for a ${LEGEND_CAP}-row legend`)
    assert.equal(new Set(p.colors).size, p.colors.length, `${p.key} repeats a colour`)
  }
})

test('no palette contains a colour that disappears into the map', () => {
  // Okabe–Ito's published set ends in black, which is right for lines on a white
  // page and wrong here: on the dark theme it is invisible, and on any basemap
  // it reads as a hole rather than a category.
  for (const p of PALETTES) {
    for (const c of p.colors) {
      assert.match(c, /^#[0-9a-f]{6}$/i, `${p.key}: ${c} is not a hex colour`)
      const [r, g, b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
      assert.ok(lum > 24, `${p.key}: ${c} is effectively black (luminance ${lum.toFixed(0)})`)
      assert.ok(lum < 244, `${p.key}: ${c} is effectively white (luminance ${lum.toFixed(0)})`)
    }
  }
})

test('shading mixes toward white and black without reaching either', () => {
  assert.equal(shade('#2a78d6', 0), '#2a78d6')
  const lighter = shade('#2a78d6', 0.3)
  const darker = shade('#2a78d6', -0.28)
  assert.notEqual(lighter, '#2a78d6')
  assert.notEqual(darker, '#2a78d6')
  assert.notEqual(lighter, darker)

  const lum = (c) => {
    const [r, g, b] = [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
  }
  assert.ok(lum(lighter) > lum('#2a78d6'))
  assert.ok(lum(darker) < lum('#2a78d6'))
  // The point of bounding the shades: a category must never become invisible.
  assert.ok(lum(darker) > 20, `darkest shade luminance ${lum(darker).toFixed(0)}`)
})

test('shading leaves anything that is not a plain hex colour alone', () => {
  assert.equal(shade('rebeccapurple', 0.3), 'rebeccapurple')
  assert.equal(shade('#abc', 0.3), '#abc')
  assert.equal(shade(null, 0.3), null)
  assert.equal(shade('#2a78d6', undefined), '#2a78d6')
})

test('shading widens the palette instead of repeating it', () => {
  // Three shades of twelve bases is what lets hundreds of species be told apart
  // while a species still keeps one colour everywhere.
  for (const p of PALETTES) {
    const all = new Set()
    for (const t of [0, 0.3, -0.28]) for (const c of p.colors) all.add(shade(c, t))
    assert.equal(all.size, p.colors.length * 3,
      `${p.key}: ${all.size} distinct colours from ${p.colors.length} bases`)
  }
})
