/* The filters the map can now set for itself: a search box, a taxon and an
 * elevation band.
 *
 * These join a predicate that already decided what 48,000 points do, so the
 * risk is not that they fail to match — it is that they match too much. A
 * filter that quietly keeps a record with no elevation, or one that treats an
 * empty box as a filter, removes points nobody asked to remove or fails to
 * remove the ones they did.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  EMPTY_FILTERS, matchesFilters, searchHaystack,
} from '../composables/useFilters.js'

const at = (props, lon = -105, lat = 40) => ({
  geometry: { type: 'Point', coordinates: [lon, lat] },
  properties: props,
})
const pass = (props, filters) => matchesFilters(at(props), { ...EMPTY_FILTERS, ...filters })

// ── Nothing set ──────────────────────────────────────────────────────────────

test('the empty filter keeps everything, including sparse records', () => {
  assert.ok(pass({}, {}))
  assert.ok(pass({ species: 'Amanita muscaria', elevation: 2400 }, {}))
  // No elevation and no elevation filter is not a reason to drop a point.
  assert.ok(pass({ species: 'Amanita muscaria' }, {}))
})

test('a blank search or taxon is not a filter', () => {
  assert.ok(pass({ species: 'Boletus edulis' }, { search: '' }))
  assert.ok(pass({ species: 'Boletus edulis' }, { search: '   ' }))
  assert.ok(pass({ species: 'Boletus edulis' }, { taxon: '' }))
})

// ── Search ───────────────────────────────────────────────────────────────────

test('search looks at the name, the common name and the place', () => {
  const props = {
    species: 'Amanita muscaria',
    common_name: 'Fly agaric',
    location: 'Ward, Boulder County, Colorado, US',
  }
  for (const q of ['amanita', 'muscaria', 'fly agaric', 'boulder', 'colorado']) {
    assert.ok(pass(props, { search: q }), `"${q}" did not match`)
  }
  assert.ok(!pass(props, { search: 'boletus' }))
})

test('search is a substring, because the useful word is often the second one', () => {
  const props = { species: 'Amanita muscaria' }
  assert.ok(pass(props, { search: 'muscaria' }))
  assert.ok(pass(props, { search: 'scar' }))
})

test('search ignores case and surrounding space', () => {
  const props = { species: 'Amanita muscaria' }
  assert.ok(pass(props, { search: '  AMANITA  ' }))
})

test('a record with nothing to search does not match a search', () => {
  assert.ok(!pass({}, { search: 'amanita' }))
  assert.ok(!pass({ elevation: 2400 }, { search: 'amanita' }))
})

test('the haystack is every field worth searching, lower-cased', () => {
  const hay = searchHaystack({
    species: 'Amanita Muscaria', common_name: 'Fly Agaric', genus: 'Amanita',
    family: 'Amanitaceae', location: 'Ward, CO',
  })
  assert.equal(hay, hay.toLowerCase())
  for (const bit of ['amanita muscaria', 'fly agaric', 'amanitaceae', 'ward']) {
    assert.ok(hay.includes(bit), `${bit} is not searched`)
  }
  assert.equal(searchHaystack({}), '')
  assert.equal(searchHaystack(), '')
})

// ── Taxon ────────────────────────────────────────────────────────────────────

test('a taxon matches at any rank', () => {
  const props = { kingdom: 'Fungi', genus: 'Amanita', species: 'Amanita muscaria' }
  assert.ok(pass(props, { taxon: 'Fungi' }))
  assert.ok(pass(props, { taxon: 'Amanita' }))
  assert.ok(pass(props, { taxon: 'Amanita muscaria' }))
  assert.ok(!pass(props, { taxon: 'Boletus' }))
})

test('a genus matches a binomial when the dataset has no genus column', () => {
  // The shipped dataset is exactly this shape, and the map has to agree with
  // the job runner about it — both call matchesTaxon.
  assert.ok(pass({ species: 'Caloboletus conifericola' }, { taxon: 'Caloboletus' }))
  assert.ok(!pass({ species: 'Caloboletus conifericola' }, { taxon: 'Boletus' }))
})

// ── Elevation ────────────────────────────────────────────────────────────────

test('an elevation band keeps what is inside it, ends included', () => {
  const band = { elevMin: 2000, elevMax: 3000 }
  assert.ok(pass({ elevation: 2000 }, band))
  assert.ok(pass({ elevation: 2500 }, band))
  assert.ok(pass({ elevation: 3000 }, band))
  assert.ok(!pass({ elevation: 1999 }, band))
  assert.ok(!pass({ elevation: 3001 }, band))
})

test('one end of the band is a filter on its own', () => {
  assert.ok(pass({ elevation: 3500 }, { elevMin: 3000 }))
  assert.ok(!pass({ elevation: 2000 }, { elevMin: 3000 }))
  assert.ok(pass({ elevation: 1000 }, { elevMax: 2000 }))
  assert.ok(!pass({ elevation: 2500 }, { elevMax: 2000 }))
})

test('a record with no elevation is outside every band rather than inside all of them', () => {
  // The alternative puts unplaced points in whichever band you are looking at,
  // which reads as data where there is none.
  for (const props of [{}, { elevation: null }, { elevation: '' }, { elevation: 'high' }]) {
    assert.ok(!pass(props, { elevMin: 2000 }), `${JSON.stringify(props)} passed a minimum`)
    assert.ok(!pass(props, { elevMax: 3000 }), `${JSON.stringify(props)} passed a maximum`)
  }
})

test('zero is an elevation, not a missing one', () => {
  assert.ok(pass({ elevation: 0 }, { elevMax: 100 }))
  assert.ok(!pass({ elevation: 0 }, { elevMin: 100 }))
})

test('the band is metres whatever the viewer reads in', () => {
  // Stored canonically on purpose: a filter kept in whichever unit happened to
  // be on when it was set changes meaning when somebody switches to feet.
  assert.ok(pass({ elevation: 2400 }, { elevMin: 2000, elevMax: 3000 }),
    '2400 m is inside 2000–3000 m')
  assert.ok(!pass({ elevation: 2400 }, { elevMin: 7000, elevMax: 9000 }),
    'the band was read as feet')
})

// ── Together ─────────────────────────────────────────────────────────────────

test('the filters are an AND, not an OR', () => {
  const props = { species: 'Amanita muscaria', elevation: 2400 }
  assert.ok(pass(props, { search: 'amanita', elevMin: 2000 }))
  assert.ok(!pass(props, { search: 'amanita', elevMin: 3000 }))
  assert.ok(!pass(props, { search: 'boletus', elevMin: 2000 }))
})

test('the new filters compose with the old ones', () => {
  const props = {
    species: 'Amanita muscaria', elevation: 2400, date: '2026-08-01',
    location: 'Ward, Boulder County, Colorado, US',
  }
  assert.ok(pass(props, { search: 'amanita', year: '2026', county: 'Boulder County' }))
  assert.ok(!pass(props, { search: 'amanita', year: '2025' }))
})
