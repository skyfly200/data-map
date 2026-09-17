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

import {
  CODE_LIMIT, EE_TILE_LAYERS, LayerError, cacheKey, codeList, describeLayer, normaliseCodes,
  resolveLayer, tierFor,
} from '../netlify/lib/ee-tile-layers.mjs'
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
const chosen = EE_TILE_LAYERS['soil-taxonomy-select']

// ── Shape ────────────────────────────────────────────────────────────────────

test('both layers are in the catalogue, in the Soil group, gated like the rest of it', () => {
  for (const [key, layer] of [['soil-taxonomy', taxonomy], ['soil-taxonomy-select', chosen]]) {
    assert.ok(layer, `${key} is missing`)
    assert.equal(layer.group, 'Soil')
    assert.equal(tierFor(key), 'member')
    assert.ok(layer.note, `${key} has no caveat`)
    assert.equal(typeof layer.prepare, 'function', `${key} cannot read its class table`)
    assert.equal(describeLayer(key).classes, 'great-groups',
      `${key} does not tell the map a class table exists`)
  }
})

test('a layer whose key IS the order chips hands its key to the browser', () => {
  // The browser draws the twelve orders as chips you can filter by. For the
  // orders layer those chips are the key, so drawing the flat swatch list too
  // is the same twelve entries twice, one set of them not clickable. The
  // Both layers paint by order, so both hand their key to the browser.
  assert.equal(describeLayer('soil-taxonomy').legendInBrowser, true)
  assert.equal(describeLayer('soil-taxonomy-select').legendInBrowser, true)
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

// ── The chosen-classes layer ─────────────────────────────────────────────────

test('only the chosen classes are painted, each in its own order colour', () => {
  const { ee, calls } = stubEe()
  const { vis } = chosen.build(ee, { codes: '18,213,999' }, TABLE)

  // Hapludalfs (Alfisols, 1) and Dystrocryepts (Inceptisols, 7) are chosen and
  // placeable; 999 is Water, which has no order and so is masked.
  assert.deepEqual(calls.remap.from, [18, 213, 999])
  assert.deepEqual(calls.remap.to, [1, 7, 0])
  assert.equal(calls.remap.dflt, 0)
  assert.equal(calls.selfMask, 1, 'unchosen ground is painted rather than left blank')
  assert.equal(vis.palette.length, 12, 'a selection across orders cannot be told apart')
})

test('choosing nothing paints nothing, rather than everything', () => {
  // The opposite default is the dangerous one: switch the layer on, see the
  // whole world painted, and read it as "all of this matches".
  for (const codes of ['', undefined, null]) {
    const { ee, calls } = stubEe()
    chosen.build(ee, { codes }, TABLE)
    assert.deepEqual(calls.remap.from, [], `codes=${codes} painted something`)
    assert.deepEqual(calls.remap.to, [])
  }
  const { ee, calls } = stubEe()
  chosen.build(ee, {}, TABLE)
  assert.deepEqual(calls.remap.from, [])
})

test('a code the raster does not have is dropped rather than shifting the list', () => {
  // from and to are built together from the table, so an unknown code cannot
  // pair a real class with the wrong colour.
  const { ee, calls } = stubEe()
  chosen.build(ee, { codes: '18,77777' }, TABLE)
  assert.deepEqual(calls.remap.from, [18])
  assert.deepEqual(calls.remap.to, [1])
})

test('the chosen list is always as long as the colours it is paired with', () => {
  const { ee, calls } = stubEe()
  chosen.build(ee, { codes: TABLE.values.join(',') }, TABLE)
  assert.equal(calls.remap.from.length, calls.remap.to.length)
  assert.equal(calls.remap.from.length, TABLE.values.length)
})

test('the note says it is a filter and not a prediction', () => {
  // The layer says the soil is the kind you asked for. It knows nothing about
  // hosts, and a habitat layer read as a prediction is how somebody ends up
  // walking a long way to a place with no trees on it.
  assert.match(chosen.note, /not a prediction|soil filter/i)
})

// ── The selection parameter ──────────────────────────────────────────────────

test('a selection is deduplicated and sorted, so one selection is one cache entry', () => {
  assert.equal(normaliseCodes('213,18,18,213'), '18,213')
  assert.equal(normaliseCodes([213, 18, 18]), '18,213')
  assert.equal(normaliseCodes(' 18 , 213 '), '18,213')
  assert.equal(normaliseCodes('18,,213'), '18,213')
})

test('an empty selection is an empty string, not a list with nothing in it', () => {
  for (const empty of ['', ' ', ',', [], null, undefined]) {
    assert.equal(normaliseCodes(empty), '', `${JSON.stringify(empty)} did not come back empty`)
  }
  assert.deepEqual(codeList(''), [])
  assert.deepEqual(codeList(null), [])
  assert.deepEqual(codeList('18,213'), [18, 213])
})

test('a selection that is not numbers is refused rather than silently dropped', () => {
  // It reaches Earth Engine as a remap list, and a selection quietly missing
  // what could not be parsed draws a map that is wrong invisibly.
  assert.throws(() => normaliseCodes('18,nonsense'), LayerError)
  assert.throws(() => normaliseCodes('DROP TABLE'), LayerError)
})

test('every class in the raster can be chosen at once', () => {
  // The limit used to be the URL's, which put a ceiling on a selection at a few
  // hundred classes. The selection travels in a request body now, so "all of
  // them" is a selection like any other.
  const all = Array.from({ length: 500 }, (_, i) => i + 1)
  assert.equal(codeList(normaliseCodes(all)).length, 500)
})

test('an absurd selection is still refused rather than trimmed', () => {
  // What is left of the limit is a bound on how much work one request may ask
  // for, not a bound on how many soils there are.
  const many = Array.from({ length: CODE_LIMIT + 1 }, (_, i) => i + 1)
  assert.throws(() => normaliseCodes(many), (err) => {
    assert.ok(err instanceof LayerError)
    assert.match(err.message, new RegExp(String(CODE_LIMIT)))
    return true
  })
  assert.equal(codeList(normaliseCodes(many.slice(0, CODE_LIMIT))).length, CODE_LIMIT)
})

test('a selection of every class is bigger than a URL, which is why it is a body', () => {
  // The number this test is really pinning: if a four-hundred-class selection
  // fitted comfortably in a query string, the POST path would be dead weight.
  const all = Array.from({ length: 430 }, (_, i) => i + 1)
  const query = new URLSearchParams({ layer: 'soil-taxonomy-select', codes: normaliseCodes(all) })
  assert.ok(query.toString().length > 1800,
    `a full selection is only ${query.toString().length} characters, so it would fit`)
})

test('the layer reads its selection through the same checks', () => {
  assert.equal(resolveLayer('soil-taxonomy-select', { codes: '213,18' }).params.codes, '18,213')
  assert.equal(resolveLayer('soil-taxonomy-select').params.codes, '')
  assert.throws(() => resolveLayer('soil-taxonomy-select', { codes: 'x' }), LayerError)
})

test('the same selection in a different order is one tile, not two', () => {
  const a = resolveLayer('soil-taxonomy-select', { codes: '18,213' })
  const b = resolveLayer('soil-taxonomy-select', { codes: '213,18,18' })
  assert.equal(cacheKey('soil-taxonomy-select', a.params), cacheKey('soil-taxonomy-select', b.params))
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
  chosen.prepare(b.ee)
  assert.equal(a.calls.asset, b.calls.asset)
  assert.match(a.calls.asset, /SOL_GRTGROUP_USDA-SOILTAX_C/)
})

test('an empty table yields an empty remap rather than a painted map', () => {
  // The tile function refuses to render on an empty table; this is the second
  // line, so that if it ever got through, the layer is blank rather than
  // uniformly one colour.
  for (const layer of [taxonomy, chosen]) {
    const { ee, calls } = stubEe()
    layer.build(ee, { codes: '18' }, { values: [], names: [] })
    assert.deepEqual(calls.remap.from, [])
    assert.deepEqual(calls.remap.to, [])
    assert.equal(calls.selfMask, 1)
  }
})

test('a missing table does not throw before the error can be reported', () => {
  for (const layer of [taxonomy, chosen]) {
    const { ee } = stubEe()
    assert.doesNotThrow(() => layer.build(ee, { codes: '18' }, null))
    assert.doesNotThrow(() => layer.build(ee, { codes: '18' }, undefined))
  }
})

test('the FRMS starting selection is still every one of its great groups', () => {
  // The list is offered in the picker as one button, and a typo in it is
  // invisible: the button just picks seventeen soils instead of eighteen.
  const table = {
    values: MATSUTAKE_GREAT_GROUPS.map((_, i) => i + 1),
    names: [...MATSUTAKE_GREAT_GROUPS],
  }
  const byName = new Map(table.names.map((n, i) => [n, table.values[i]]))
  const picked = MATSUTAKE_GREAT_GROUPS.map((n) => byName.get(n)).filter((c) => c !== undefined)
  assert.equal(picked.length, MATSUTAKE_GREAT_GROUPS.length)

  const { ee, calls } = stubEe()
  chosen.build(ee, { codes: normaliseCodes(picked) }, table)
  assert.equal(calls.remap.from.length, MATSUTAKE_GREAT_GROUPS.length)
})
