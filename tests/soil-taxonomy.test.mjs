/* Reading a USDA great group name.
 *
 * The map paints twelve orders and the panel explains four hundred great
 * groups, and both come from the same place: the name. So the risk is all in
 * the parsing — a suffix tested in the wrong order putting Mollisols under
 * Gelisols, a short formative element swallowing a long one, a label that is
 * not a great group at all getting a confident description anyway.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  FORMATIVE_ELEMENTS, MATSUTAKE_GREAT_GROUPS, SOIL_ORDERS, SOIL_TAXONOMY_WIKI,
  describeGreatGroup, elementsOf, orderForGreatGroup, orderIndex, searchGreatGroups,
} from '../netlify/lib/soil-taxonomy.mjs'

// ── The orders ───────────────────────────────────────────────────────────────

test('there are twelve orders, each complete', () => {
  assert.equal(SOIL_ORDERS.length, 12)
  for (const o of SOIL_ORDERS) {
    assert.ok(o.key && o.name && o.suffix, `${o.name} is missing a field`)
    assert.ok(o.summary && o.where, `${o.name} has no description`)
    assert.match(o.color, /^#[0-9a-f]{6}$/, `${o.name} has no colour`)
    assert.match(o.wiki, /^https:\/\/en\.wikipedia\.org\/wiki\//, `${o.name} has no article`)
  }
  const keys = SOIL_ORDERS.map((o) => o.key)
  assert.equal(new Set(keys).size, 12)
  const colors = SOIL_ORDERS.map((o) => o.color)
  assert.equal(new Set(colors).size, 12, 'two orders share a colour, so the key cannot be read')
})

test('the orders are listed alphabetically, which is how every source lists them', () => {
  const names = SOIL_ORDERS.map((o) => o.name)
  assert.deepEqual(names, [...names].sort())
})

// ── Finding the order ────────────────────────────────────────────────────────

test('one great group from each order lands in that order', () => {
  const cases = {
    Hapludalfs: 'Alfisols',
    Hapludands: 'Andisols',
    Haplocalcids: 'Aridisols',
    Udipsamments: 'Entisols',
    Haplorthels: 'Gelisols',
    Haplosaprists: 'Histosols',
    Dystrudepts: 'Inceptisols',
    Argiudolls: 'Mollisols',
    Hapludox: 'Oxisols',
    Haplorthods: 'Spodosols',
    Hapludults: 'Ultisols',
    Haplotorrerts: 'Vertisols',
  }
  for (const [group, order] of Object.entries(cases)) {
    assert.equal(orderForGreatGroup(group)?.name, order, `${group} went to the wrong order`)
  }
})

test('the suffixes that could be confused for each other are not', () => {
  // 'olls' ends in 'lls' and 'els' ends in 'els'; a careless endsWith on the
  // last three characters puts every Mollisol in the permafrost.
  assert.equal(orderForGreatGroup('Argiudolls').name, 'Mollisols')
  assert.equal(orderForGreatGroup('Haplorthels').name, 'Gelisols')
  // 'ids' and 'ods' differ by one letter, and one is a desert.
  assert.equal(orderForGreatGroup('Natrargids').name, 'Aridisols')
  assert.equal(orderForGreatGroup('Haplohumods').name, 'Spodosols')
  // 'ents' and 'epts' likewise.
  assert.equal(orderForGreatGroup('Cryopsamments').name, 'Entisols')
  assert.equal(orderForGreatGroup('Cryumbrepts').name, 'Inceptisols')
  // Oxisol great groups are the one family that does not end in s.
  assert.equal(orderForGreatGroup('Acrudox').name, 'Oxisols')
})

test('case and whitespace do not change the answer', () => {
  assert.equal(orderForGreatGroup('  HAPLUDALFS ').name, 'Alfisols')
  assert.equal(orderForGreatGroup('hapludalfs').name, 'Alfisols')
})

test('a label that is not a great group has no order rather than the nearest one', () => {
  for (const junk of ['', '  ', null, undefined, 'No data', 'Water', 'Urban land', 42]) {
    assert.equal(orderForGreatGroup(junk), null, `${junk} was given an order`)
  }
})

test('the palette index is 1-based and matches the order list', () => {
  assert.equal(orderIndex('Hapludalfs'), 1)
  assert.equal(orderIndex('Haplotorrerts'), 12)
  // Zero is nodata, so an unplaceable class is not painted as an Alfisol.
  assert.equal(orderIndex('Water'), 0)
  assert.equal(orderIndex(''), 0)
  for (const o of SOIL_ORDERS) {
    const sample = `hapl${o.suffix}`
    assert.equal(SOIL_ORDERS[orderIndex(sample) - 1].key, o.key)
  }
})

// ── Reading the modifiers ────────────────────────────────────────────────────

test('formative elements are listed longest first, so none hides another', () => {
  // 'quartzi' starts with 'qu' territory, 'dystro' contains 'dystr', 'xero'
  // contains 'xer', 'udi' contains 'ud'. The long one has to be tried first or
  // the short one eats it.
  for (let i = 0; i < FORMATIVE_ELEMENTS.length; i += 1) {
    for (let j = i + 1; j < FORMATIVE_ELEMENTS.length; j += 1) {
      const a = FORMATIVE_ELEMENTS[i].element
      const b = FORMATIVE_ELEMENTS[j].element
      assert.ok(!b.startsWith(a) || a === b,
        `"${b}" comes after "${a}", which it starts with, so it can never match`)
    }
  }
})

test('a compound name decodes into its parts, in the order they are written', () => {
  const parts = elementsOf('Dystrocryepts').map((e) => e.element)
  assert.deepEqual(parts, ['dystro', 'cry'])
  assert.deepEqual(elementsOf('Vitricryands').map((e) => e.element), ['vitr', 'cry'])
  assert.deepEqual(elementsOf('Quartzipsamments').map((e) => e.element), ['quartzi', 'psamm'])
  assert.deepEqual(elementsOf('Haplohumods').map((e) => e.element), ['hapl', 'hum'])
})

test('a short element does not swallow the long one it is inside', () => {
  // 'ud' would match the front of 'quartzi'-free 'udipsamments' correctly, but
  // 'qu' must not be what Quartzipsamments matches on.
  assert.ok(elementsOf('Quartzipsamments').every((e) => e.element !== 'aqu'),
    'Quartzipsamments was read as a wet soil')
  assert.deepEqual(elementsOf('Xeropsamments').map((e) => e.element), ['xero', 'psamm'])
})

test('a connecting vowel is skipped rather than ending the read', () => {
  // 'haplocryalfs' is hapl + o + cry. Stopping at the o would lose the cold.
  assert.deepEqual(elementsOf('Haplocryalfs').map((e) => e.element), ['hapl', 'cry'])
})

test('an element the table does not know does not stop the ones that follow', () => {
  const parts = elementsOf('Zzzcryepts').map((e) => e.element)
  assert.ok(parts.includes('cry'), `expected cry, got ${parts.join(', ')}`)
})

test('reading a name always ends, whatever the name is', () => {
  // The loop drops a letter when nothing matches, so the only way it could run
  // away is a zero-length element.
  for (const e of FORMATIVE_ELEMENTS) assert.ok(e.element.length > 0)
  for (const junk of ['', 'aaaaaaaaaaaaaaaaaaaaaaaa', 'xxxxxxxxalfs']) {
    assert.ok(Array.isArray(elementsOf(junk)))
  }
})

// ── The description ──────────────────────────────────────────────────────────

test('a known class is described by its order and its modifiers', () => {
  const d = describeGreatGroup('Dystrocryepts')
  assert.equal(d.known, true)
  assert.equal(d.name, 'Dystrocryepts')
  assert.equal(d.order, 'Inceptisols')
  assert.equal(d.orderKey, 'inceptisols')
  assert.match(d.color, /^#[0-9a-f]{6}$/)
  assert.match(d.wiki, /Inceptisol/)
  assert.ok(d.summary && d.where)
  assert.equal(d.elements.length, 2)
  assert.match(d.elements[0].meaning, /acid/)
  assert.match(d.elements[1].meaning, /cold/i)
})

test('an unplaceable class says so instead of inventing a description', () => {
  const d = describeGreatGroup('Water')
  assert.equal(d.known, false)
  assert.equal(d.order, '')
  assert.equal(d.summary, '')
  assert.deepEqual(d.elements, [])
  // Still a link, but to the classification rather than to a wrong order.
  assert.equal(d.wiki, SOIL_TAXONOMY_WIKI)
})

test('describing nothing does not throw', () => {
  for (const junk of [null, undefined, '', 0]) {
    const d = describeGreatGroup(junk)
    assert.equal(d.known, false)
    assert.equal(typeof d.name, 'string')
  }
})

// ── Search ───────────────────────────────────────────────────────────────────

const CLASSES = [
  { code: 18, name: 'Hapludalfs' },
  { code: 213, name: 'Dystrocryepts' },
  { code: 357, name: 'Haplorthods' },
  { code: 900, name: 'Water' },
]

test('search matches the name', () => {
  assert.deepEqual(searchGreatGroups(CLASSES, 'hapl').map((c) => c.name),
    ['Hapludalfs', 'Haplorthods'])
  assert.deepEqual(searchGreatGroups(CLASSES, 'CRYEPTS').map((c) => c.name), ['Dystrocryepts'])
})

test('search matches the order, which the names never contain', () => {
  // Nobody looking for podzols knows to type "orthods".
  assert.deepEqual(searchGreatGroups(CLASSES, 'spodosol').map((c) => c.name), ['Haplorthods'])
  assert.deepEqual(searchGreatGroups(CLASSES, 'inceptisols').map((c) => c.name), ['Dystrocryepts'])
})

test('an empty search is everything, and a copy', () => {
  const all = searchGreatGroups(CLASSES, '')
  assert.equal(all.length, CLASSES.length)
  assert.notEqual(all, CLASSES, 'the caller was handed the array it passed in')
  assert.deepEqual(searchGreatGroups(CLASSES, '   ').length, CLASSES.length)
  assert.deepEqual(searchGreatGroups([], 'hapl'), [])
})

test('a search that matches nothing matches nothing', () => {
  assert.deepEqual(searchGreatGroups(CLASSES, 'zzzz'), [])
})

// ── The matsutake list ───────────────────────────────────────────────────────

test('every flagged great group is a real one this app can place', () => {
  assert.equal(new Set(MATSUTAKE_GREAT_GROUPS).size, MATSUTAKE_GREAT_GROUPS.length)
  for (const name of MATSUTAKE_GREAT_GROUPS) {
    assert.ok(orderForGreatGroup(name), `${name} does not end in an order suffix`)
  }
})

test('the flagged list still names the same soils the codes it came from did', () => {
  // The list is kept by name, because names belong to the taxonomy and codes
  // belong to one version of one raster. This is the cross-check: the pairs as
  // they were read out of OpenLandMap v01. If a future version renumbers, this
  // test does not fail — but the pairing is recorded here, so the next person
  // can see what the list was built against.
  const asRead = {
    18: 'Hapludalfs', 44: 'Cryoboralfs', 61: 'Haplocryands', 64: 'Hapludands',
    77: 'Vitricryands', 80: 'Vitrixerands', 120: 'Cryopsamments', 135: 'Quartzipsamments',
    142: 'Udipsamments', 148: 'Xeropsamments', 213: 'Dystrocryepts', 216: 'Dystrudepts',
    228: 'Haploxerepts', 251: 'Haplocryalfs', 254: 'Cryumbrepts', 351: 'Fragiorthods',
    356: 'Haplohumods', 357: 'Haplorthods',
  }
  assert.deepEqual([...MATSUTAKE_GREAT_GROUPS].sort(), Object.values(asRead).sort())
})

test('the flagged soils are the cool, acid, sandy and volcanic ones', () => {
  // Not a formality: the point of the list is a habitat, and a habitat that
  // spans every order would mean the list had drifted into "most soils".
  const orders = new Set(MATSUTAKE_GREAT_GROUPS.map((n) => orderForGreatGroup(n).key))
  assert.deepEqual([...orders].sort(),
    ['alfisols', 'andisols', 'entisols', 'inceptisols', 'spodosols'])
})
