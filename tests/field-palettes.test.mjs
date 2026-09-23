/* Colouring a point the way the layer under it is coloured.
 *
 * The failures here are all quiet ones. A palette that drifts from its layer
 * still draws; a domain that no longer matches the layer's stretch still draws;
 * a class label the pipeline spells differently just falls back to a hash
 * colour. In every case the map looks fine and the two views of one number
 * disagree, which is the thing reading a point against its layer is for.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  FIELD_LAYERS, MATCHED_FIELDS,
  classColorFor, fraction, matchNote, paletteFor,
} from '../composables/fieldPalettes.js'
import { mix, rampColor } from '../composables/ramps.js'
import { EE_TILE_LAYERS, WORLDCOVER_CLASSES, WORLDCOVER_FROM, resolveLayer }
  from '../netlify/lib/ee-tile-layers.mjs'

// The same stub the layer tests use: what matters is the visualisation each
// build() asks for, not that Earth Engine agrees.
function stubEe() {
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => (prop === 'then' ? undefined : chain),
    apply: () => chain,
  })
  return new Proxy({}, {
    get: (t, prop) => {
      if (prop === 'Filter') return { lt: () => chain, eq: () => chain }
      if (prop === 'Image') {
        const img = () => chain
        img.constant = () => chain
        img.cat = () => chain
        return img
      }
      return chain
    },
  })
}

// ── The table points at real layers ──────────────────────────────────────────

test('every matched field names a layer that exists and has a key', () => {
  assert.ok(MATCHED_FIELDS.length >= 6)
  for (const field of MATCHED_FIELDS) {
    const { layer: key } = FIELD_LAYERS[field]
    const layer = EE_TILE_LAYERS[key]
    assert.ok(layer, `${field} points at “${key}”, which is not a layer`)
    assert.ok(layer.legend, `${field} points at ${key}, which has no legend to borrow`)
  }
})

test('a declared domain is the one the layer actually renders', () => {
  // The check that keeps this table honest. A layer's stretch is a judgement
  // that gets revisited; if it moves and this does not, points and pixels drift
  // apart by exactly that much and nothing reports it.
  for (const field of MATCHED_FIELDS) {
    const match = FIELD_LAYERS[field]
    if (!match.domain) continue
    const { layer, params } = resolveLayer(match.layer)
    const { vis } = layer.build(stubEe(), params)
    assert.deepEqual(
      match.domain, [vis.min, vis.max],
      `${field} claims the ${match.layer} layer renders `
      + `${JSON.stringify(match.domain)}, but it renders [${vis.min}, ${vis.max}]`,
    )
  }
})

test('a field whose units differ from its layer declares no domain', () => {
  // The pipeline rank-normalises these to 0..1 while the layer draws raw heat
  // load, raw TPI in metres and raw TWI. Borrowing the layer's stretch would
  // put a confident number on a coincidence.
  for (const field of ['solar_exposure', 'wind_exposure', 'water_retention']) {
    assert.equal(FIELD_LAYERS[field].comparable, false, `${field} should not claim comparable units`)
    assert.equal(paletteFor(field).domain, null, `${field} must not borrow a domain`)
  }
})

test('an unmatched field gets nothing rather than a wrong palette', () => {
  assert.equal(paletteFor('elevation'), null)
  assert.equal(paletteFor('species'), null)
  assert.equal(paletteFor(undefined), null)
})

// ── The palettes themselves ──────────────────────────────────────────────────

test('a matched field borrows its layer\'s palette rather than copying it', () => {
  const p = paletteFor('ndvi')
  assert.equal(p.kind, 'ramp')
  // Identity, not equality: a copy is a second definition that can drift.
  assert.equal(p.stops, EE_TILE_LAYERS['ndvi-recent'].legend.stops)
  assert.deepEqual(p.domain, [-0.2, 0.9])
})

test('land cover borrows the class list', () => {
  const p = paletteFor('land_cover_label')
  assert.equal(p.kind, 'classes')
  assert.equal(p.items, EE_TILE_LAYERS['land-cover'].legend.items)
})

// ── Ramps ────────────────────────────────────────────────────────────────────

test('a ramp hits its own ends exactly', () => {
  const stops = ['#000000', '#888888', '#ffffff']
  assert.equal(rampColor(stops, 0), '#000000')
  assert.equal(rampColor(stops, 1), '#ffffff')
  assert.equal(rampColor(stops, 0.5), '#888888')
})

test('a ramp uses every stop, not just the two ends', () => {
  // The layers declare six to ten stops, and that shape is the whole reason
  // their middles are readable. A two-colour lerp between the extremes would
  // keep the ends and flatten everything between them.
  const stops = ['#000000', '#ff0000', '#ffffff']
  const middle = rampColor(stops, 0.5)
  assert.equal(middle, '#ff0000')
  assert.notEqual(middle, mix('#000000', '#ffffff', 0.5))
})

test('a ramp clamps rather than running off either end', () => {
  const stops = ['#000000', '#ffffff']
  assert.equal(rampColor(stops, -5), '#000000')
  assert.equal(rampColor(stops, 5), '#ffffff')
  // NaN is what an unparseable value produces, and it must not become a colour
  // outside the ramp.
  assert.equal(rampColor(stops, NaN), '#000000')
})

test('a degenerate ramp still returns a colour', () => {
  assert.equal(rampColor(['#123456'], 0.4), '#123456')
  assert.match(rampColor([], 0.4), /^#[0-9a-f]{6}$/)
  assert.match(rampColor(null, 0.4), /^#[0-9a-f]{6}$/)
})

test('every layer ramp this table borrows produces valid hex throughout', () => {
  for (const field of MATCHED_FIELDS) {
    const p = paletteFor(field)
    if (p.kind !== 'ramp') continue
    for (const t of [0, 0.17, 0.33, 0.5, 0.66, 0.83, 1]) {
      assert.match(rampColor(p.stops, t), /^#[0-9a-f]{6}$/, `${field} at ${t}`)
    }
  }
})

// ── Domains ──────────────────────────────────────────────────────────────────

test('a value maps to where it sits in the domain', () => {
  assert.equal(fraction(0, [0, 45]), 0)
  assert.equal(fraction(45, [0, 45]), 1)
  assert.equal(fraction(22.5, [0, 45]), 0.5)
  // A domain that does not start at zero is the common case here.
  assert.ok(Math.abs(fraction(0.35, [-0.2, 0.9]) - 0.5) < 1e-9)
})

test('a zero-width domain does not divide by zero', () => {
  assert.equal(fraction(5, [3, 3]), 0)
  assert.equal(fraction(5, [NaN, 3]), 0)
})

// ── Class labels ─────────────────────────────────────────────────────────────

test('a class label finds its colour', () => {
  const items = paletteFor('land_cover_label').items
  assert.equal(classColorFor(items, 'Tree cover'), '#006400')
  assert.equal(classColorFor(items, 'Grassland'), '#ffff4c')
})

test('the spellings the pipeline writes find the same colours', () => {
  // scripts/enrich_with_rasters.py names code 60 "Bare / sparse vegetation" and
  // code 90 "Wetland", where WorldCover's own table says "Bare / sparse" and
  // "Herbaceous wetland". Both appear in the shipped dataset, and without the
  // aliases they fall through to a hash colour that matches nothing.
  const items = paletteFor('land_cover_label').items
  assert.equal(classColorFor(items, 'Bare / sparse vegetation'), '#b4b4b4')
  assert.equal(classColorFor(items, 'Wetland'), '#0096a0')
  assert.equal(classColorFor(items, 'Water'), '#0064c8')
})

test('matching ignores case and surrounding space', () => {
  const items = paletteFor('land_cover_label').items
  assert.equal(classColorFor(items, '  tree cover '), '#006400')
})

test('an unknown class gets null so the caller can fall back', () => {
  const items = paletteFor('land_cover_label').items
  assert.equal(classColorFor(items, 'Lava field'), null)
  assert.equal(classColorFor(items, null), null)
  assert.equal(classColorFor(items, undefined), null)
})

// ── The WorldCover table itself ──────────────────────────────────────────────

test('the remap codes come from the class table rather than beside it', () => {
  // As two parallel lists they could fall out of order and nothing would say
  // so — the map would paint the wrong class the wrong colour and still look
  // like a map.
  assert.deepEqual(WORLDCOVER_FROM, WORLDCOVER_CLASSES.map((c) => c.code))
  assert.deepEqual(WORLDCOVER_FROM, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100])
})

test('no class colour or code is used twice', () => {
  assert.equal(new Set(WORLDCOVER_CLASSES.map((c) => c.color)).size, WORLDCOVER_CLASSES.length)
  assert.equal(new Set(WORLDCOVER_CLASSES.map((c) => c.code)).size, WORLDCOVER_CLASSES.length)
})

test('no alias collides with another class\'s label', () => {
  const labels = new Set(WORLDCOVER_CLASSES.map((c) => c.label.toLowerCase()))
  for (const c of WORLDCOVER_CLASSES) {
    for (const alias of c.aliases || []) {
      assert.ok(!labels.has(alias.toLowerCase()),
        `“${alias}” is an alias of ${c.label} and also another class's own name`)
    }
  }
})

// ── What the legend says ─────────────────────────────────────────────────────

test('the legend says which kind of match it is', () => {
  assert.match(matchNote(paletteFor('slope')), /match the Slope/i)
  // The normalised case has to say so, or a viewer reads a rank as a value.
  assert.match(matchNote(paletteFor('solar_exposure')), /normalised/i)
  assert.equal(matchNote(null), '')
})
