/* The two soil taxonomy map layers.
 *
 * Both are a remap: four hundred class codes in, a small number of painted
 * values out. A remap is the easiest thing here to get quietly wrong, because
 * every way of getting it wrong renders. A short `to` list paints the wrong
 * order; a missing selfMask paints nodata as a real class; an empty class table
 * paints nothing at all and looks exactly like ground with no soil on it.
 *
 * So Earth Engine is stubbed down to the four calls these layers make, and the
 * assertions are about the arguments that reach it.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { EE_TILE_LAYERS, describeLayer, tierFor } from '../netlify/lib/ee-tile-layers.mjs'
import { MATSUTAKE_GREAT_GROUPS, SOIL_ORDERS, orderForGreatGroup } from '../netlify/lib/soil-taxonomy.mjs'

/** Just enough Earth Engine to record what a build asked for. */
function stubEe() {
  const calls = { remap: null, selfMask: 0, selected: null, asset: null }
  const image = {
    select(band) { calls.selected = band; return image },
    remap(from, to, dflt) { calls.remap = { from, to, dflt }; return image },
    selfMask() { calls.selfMask += 1; return image },
    get(prop) { return { prop } },
  }
  const ee = {
    Image(asset) { calls.asset = asset; return image },
    Dictionary(obj) { return { dict: obj, evaluate() {} } },
  }
  return { ee, calls, image }
}

const TABLE = {
  values: [18, 61, 120, 213, 357, 400, 999],
  names: ['Hapludalfs', 'Haplocryands', 'Cryopsamments', 'Dystrocryepts', 'Haplorthods',
    'Argiudolls', 'Water'],
}

const taxonomy = EE_TILE_LAYERS['soil-taxonomy']
const matsutake = EE_TILE_LAYERS['soil-taxonomy-matsutake']

// ── Shape ────────────────────────────────────────────────────────────────────

test('both layers are in the catalogue, in the Soil group, gated like the rest of it', () => {
  for (const [key, layer] of [['soil-taxonomy', taxonomy], ['soil-taxonomy-matsutake', matsutake]]) {
    assert.ok(layer, `${key} is missing`)
    assert.equal(layer.group, 'Soil')
    assert.equal(tierFor(key), 'member')
    assert.ok(layer.note, `${key} has no caveat`)
    assert.equal(typeof layer.prepare, 'function', `${key} cannot read its class table`)
    assert.equal(describeLayer(key).classes, 'great-groups',
      `${key} does not tell the map a class table exists`)
  }
})

test('only the layer whose key IS the orders hands its key to the browser', () => {
  // The browser draws the twelve orders as chips you can filter by. For the
  // orders layer those chips are the key, so drawing the flat swatch list too
  // is the same twelve entries twice, one set of them not clickable. The
  // matsutake layer's key is one swatch the browser does not show, so it keeps
  // its own.
  assert.equal(describeLayer('soil-taxonomy').legendInBrowser, true)
  assert.equal(describeLayer('soil-taxonomy-matsutake').legendInBrowser, undefined)
})

test('the catalogue entry does not carry four hundred classes to every viewer', () => {
  const described = describeLayer('soil-taxonomy')
  assert.equal(typeof described.classes, 'string')
  assert.ok(JSON.stringify(described).length < 4000)
})

test('the orders legend has all twelve, in the palette order the remap produces', () => {
  const items = taxonomy.legend.items
  assert.equal(taxonomy.legend.type, 'classes')
  assert.equal(items.length, 12)
  items.forEach((item, i) => {
    assert.equal(item.label, SOIL_ORDERS[i].name)
    assert.equal(item.color, SOIL_ORDERS[i].color)
  })
})

// ── The orders layer ─────────────────────────────────────────────────────────

test('each class is remapped to the index of its own order', () => {
  const { ee, calls } = stubEe()
  const { vis } = taxonomy.build(ee, {}, TABLE)

  assert.equal(calls.selected, 'grtgroup')
  assert.deepEqual(calls.remap.from, TABLE.values, 'the source codes were not passed through')
  // Hapludalfs → Alfisols (1), Haplocryands → Andisols (2), Cryopsamments →
  // Entisols (4), Dystrocryepts → Inceptisols (7), Haplorthods → Spodosols
  // (10), Argiudolls → Mollisols (8), Water → nothing (0).
  assert.deepEqual(calls.remap.to, [1, 2, 4, 7, 10, 8, 0])
  assert.equal(calls.remap.dflt, 0)

  assert.equal(vis.min, 1)
  assert.equal(vis.max, 12)
  assert.equal(vis.palette.length, 12)
})

test('the from and to lists are always the same length', () => {
  // A short `to` is the remap bug that renders: Earth Engine pairs what it has
  // and every class past the end silently takes the default.
  const { ee, calls } = stubEe()
  taxonomy.build(ee, {}, TABLE)
  assert.equal(calls.remap.from.length, calls.remap.to.length)
})

test('a class that is not a great group is masked, not painted', () => {
  const { ee, calls } = stubEe()
  taxonomy.build(ee, {}, TABLE)
  const water = TABLE.names.indexOf('Water')
  assert.equal(calls.remap.to[water], 0, 'Water was given an order')
  assert.equal(calls.selfMask, 1, 'zero is painted, so nodata reads as the first order')
})

test('the palette index and the legend agree for every order', () => {
  // The two are built from the same list, and this is what says so: a class
  // remapped to n must be the order the nth legend entry names.
  const table = {
    values: SOIL_ORDERS.map((_, i) => 100 + i),
    names: SOIL_ORDERS.map((o) => `hapl${o.suffix}`),
  }
  const { ee, calls } = stubEe()
  taxonomy.build(ee, {}, table)
  calls.remap.to.forEach((n, i) => {
    assert.equal(taxonomy.legend.items[n - 1].label, orderForGreatGroup(table.names[i]).name)
  })
})

// ── The matsutake layer ──────────────────────────────────────────────────────

test('only the flagged great groups are painted', () => {
  const { ee, calls } = stubEe()
  const { vis } = matsutake.build(ee, {}, TABLE)

  // Argiudolls and Water are not on the list; the other five are.
  assert.deepEqual(calls.remap.from, [18, 61, 120, 213, 357])
  assert.deepEqual(calls.remap.to, [1, 1, 1, 1, 1])
  assert.equal(calls.remap.dflt, 0)
  assert.equal(calls.selfMask, 1, 'unflagged ground is painted "no" rather than left blank')
  assert.equal(vis.palette.length, 1)
})

test('the match is on the name and ignores case and padding', () => {
  const { ee, calls } = stubEe()
  matsutake.build(ee, {}, {
    values: [1, 2, 3],
    names: ['  hapludalfs ', 'HAPLORTHODS', 'Argiudolls'],
  })
  assert.deepEqual(calls.remap.from, [1, 2])
})

test('its legend says what the one colour means', () => {
  assert.equal(matsutake.legend.items.length, 1)
  assert.ok(matsutake.legend.items[0].label)
  assert.equal(matsutake.legend.items[0].color, '#2ca25f')
})

test('the note does not promise mushrooms', () => {
  // The layer says the soil is the right kind. It knows nothing about hosts,
  // and a habitat layer read as a prediction is how somebody ends up walking
  // a long way to a place with no trees on it.
  assert.match(matsutake.note, /not a prediction|soil filter/i)
})

// ── The class table ──────────────────────────────────────────────────────────

test('the table is read off the asset rather than written down here', () => {
  const { ee } = stubEe()
  const prepared = taxonomy.prepare(ee)
  assert.deepEqual(Object.keys(prepared.dict).sort(), ['names', 'values'])
  assert.equal(prepared.dict.values.prop, 'grtgroup_class_values')
  assert.equal(prepared.dict.names.prop, 'grtgroup_class_names')
})

test('both layers read the same table, so they cannot disagree about a code', () => {
  const a = stubEe()
  const b = stubEe()
  taxonomy.prepare(a.ee)
  matsutake.prepare(b.ee)
  assert.equal(a.calls.asset, b.calls.asset)
  assert.match(a.calls.asset, /SOL_GRTGROUP_USDA-SOILTAX_C/)
})

test('an empty table yields an empty remap rather than a painted map', () => {
  // The tile function refuses to render on an empty table; this is the second
  // line, so that if it ever got through, the layer is blank rather than
  // uniformly one colour.
  for (const layer of [taxonomy, matsutake]) {
    const { ee, calls } = stubEe()
    layer.build(ee, {}, { values: [], names: [] })
    assert.deepEqual(calls.remap.from, [])
    assert.deepEqual(calls.remap.to, [])
    assert.equal(calls.selfMask, 1)
  }
})

test('a missing table does not throw before the error can be reported', () => {
  for (const layer of [taxonomy, matsutake]) {
    const { ee } = stubEe()
    assert.doesNotThrow(() => layer.build(ee, {}, null))
    assert.doesNotThrow(() => layer.build(ee, {}, undefined))
  }
})

test('every flagged great group would be found in a full table', () => {
  // The list is by name, and a typo in it is invisible: the layer just paints
  // seventeen soils instead of eighteen.
  const { ee, calls } = stubEe()
  matsutake.build(ee, {}, {
    values: MATSUTAKE_GREAT_GROUPS.map((_, i) => i + 1),
    names: [...MATSUTAKE_GREAT_GROUPS],
  })
  assert.equal(calls.remap.from.length, MATSUTAKE_GREAT_GROUPS.length)
})
