/* The taxa a loaded dataset actually contains.
 *
 * This backs the job runner's taxon picker, and the reason it exists is a bug:
 * the field was free text, so "Amanita" was typeable against a dataset with no
 * genus column, and the job was refused for a reason that named the area and
 * the dates instead. A picker of what is there cannot be asked that question.
 *
 * Which makes the binomial fallback the part that matters most here — it has to
 * agree with matchesTaxon on the server, or the picker offers names that then
 * select nothing.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  TAXON_RANKS, countForTaxon, datasetHasTaxon, impliedGenus, taxaInFeatures,
} from '../netlify/lib/dataset-taxa.mjs'
import { matchesTaxon } from '../netlify/lib/job-source.mjs'

const at = (props) => ({ geometry: { coordinates: [-105, 40] }, properties: props })

// ── Ranks ────────────────────────────────────────────────────────────────────

test('ranks are listed coarse to fine', () => {
  assert.deepEqual(TAXON_RANKS.map((r) => r.key),
    ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species'])
  for (const r of TAXON_RANKS) assert.ok(r.label, `${r.key} has no label`)
})

// ── The binomial fallback ────────────────────────────────────────────────────

test('a binomial implies its genus when there is no genus column', () => {
  assert.equal(impliedGenus({ species: 'Caloboletus conifericola' }), 'Caloboletus')
  assert.equal(impliedGenus({ species: 'Amanita muscaria' }), 'Amanita')
})

test('a real genus column means no guess is made', () => {
  assert.equal(impliedGenus({ genus: 'Amanita', species: 'Amanita muscaria' }), '')
  assert.equal(impliedGenus({ genus: 'Lepiota', species: 'Amanita muscaria' }), '')
})

test('what is not a genus is not guessed at', () => {
  for (const props of [{}, { species: '' }, { species: '   ' }, { species: 'sp.' },
    { species: 'x y' }, { species: 'amanita muscaria' }, { species: '123 abc' }]) {
    assert.equal(impliedGenus(props), '', `${JSON.stringify(props)} produced a genus`)
  }
})

test('the picker and the server agree about every genus the picker offers', () => {
  // The property that makes this feature honest: a name in the list selects
  // records, and one that would select nothing is never in the list.
  const features = [
    at({ species: 'Caloboletus conifericola' }),
    at({ species: 'Amanita muscaria' }),
    at({ genus: 'Boletus', species: 'Boletus edulis' }),
    at({ kingdom: 'Fungi', order: 'Agaricales', species: 'Mycena pura' }),
  ]
  const ranks = taxaInFeatures(features)
  for (const rank of ranks) {
    for (const t of rank.taxa) {
      const hits = features.filter((f) => matchesTaxon(f.properties, t.name))
      assert.ok(hits.length > 0,
        `the picker offers "${t.name}" (${rank.key}) but the server matches nothing`)
      assert.equal(hits.length, t.count,
        `"${t.name}" is offered as ${t.count} records and matches ${hits.length}`)
    }
  }
})

// ── Collecting them ──────────────────────────────────────────────────────────

test('a legacy dataset of binomials still yields genera to pick from', () => {
  // 48,000 records with species and nothing else is the dataset this app
  // actually ships. A picker with no genera in it would be no use at all.
  const features = [
    at({ species: 'Amanita muscaria' }),
    at({ species: 'Amanita pantherina' }),
    at({ species: 'Boletus edulis' }),
  ]
  const ranks = taxaInFeatures(features)
  const genus = ranks.find((r) => r.key === 'genus')
  assert.deepEqual(genus.taxa, [{ name: 'Amanita', count: 2 }, { name: 'Boletus', count: 1 }])
  const species = ranks.find((r) => r.key === 'species')
  assert.equal(species.taxa.length, 3)
})

test('a rank nothing populates is left out rather than listed empty', () => {
  const ranks = taxaInFeatures([at({ species: 'Amanita muscaria' })])
  assert.deepEqual(ranks.map((r) => r.key), ['genus', 'species'])
})

test('names are ordered by how many records carry them', () => {
  // The well-recorded taxa are the useful ones; four hundred alphabetical
  // names buries them.
  const features = [
    ...Array.from({ length: 3 }, () => at({ genus: 'Rare', species: 'Rare one' })),
    ...Array.from({ length: 9 }, () => at({ genus: 'Common', species: 'Common one' })),
  ]
  const genus = taxaInFeatures(features).find((r) => r.key === 'genus')
  assert.deepEqual(genus.taxa.map((t) => t.name), ['Common', 'Rare'])
})

test('ties break alphabetically, so the list is stable between loads', () => {
  const features = [at({ genus: 'Zeta' }), at({ genus: 'Alpha' })]
  const genus = taxaInFeatures(features).find((r) => r.key === 'genus')
  assert.deepEqual(genus.taxa.map((t) => t.name), ['Alpha', 'Zeta'])
})

test('a minimum count drops the long tail', () => {
  const features = [
    ...Array.from({ length: 5 }, () => at({ genus: 'Enough' })),
    at({ genus: 'Once' }),
  ]
  const genus = taxaInFeatures(features, { min: 3 }).find((r) => r.key === 'genus')
  assert.deepEqual(genus.taxa.map((t) => t.name), ['Enough'])
})

test('an empty or malformed collection yields no ranks rather than throwing', () => {
  assert.deepEqual(taxaInFeatures([]), [])
  assert.deepEqual(taxaInFeatures(), [])
  assert.deepEqual(taxaInFeatures([null, {}, { properties: null }]), [])
})

// ── Looking one up ───────────────────────────────────────────────────────────

test('a name is counted at whatever rank it sits', () => {
  const ranks = taxaInFeatures([
    at({ kingdom: 'Fungi', genus: 'Amanita', species: 'Amanita muscaria' }),
    at({ kingdom: 'Fungi', genus: 'Boletus', species: 'Boletus edulis' }),
  ])
  assert.equal(countForTaxon(ranks, 'Fungi'), 2)
  assert.equal(countForTaxon(ranks, 'Amanita'), 1)
  assert.equal(countForTaxon(ranks, 'amanita'), 1, 'case decided the answer')
  assert.equal(countForTaxon(ranks, ' Amanita '), 1)
})

test('a name the dataset does not carry counts zero, which is the warning', () => {
  const ranks = taxaInFeatures([at({ species: 'Boletus edulis' })])
  assert.equal(countForTaxon(ranks, 'Amanita'), 0)
  assert.equal(datasetHasTaxon(ranks, 'Amanita'), false)
  assert.equal(datasetHasTaxon(ranks, 'Boletus'), true)
  // An empty taxon is "any", not a miss, and the caller checks for it first.
  assert.equal(countForTaxon(ranks, ''), 0)
  assert.equal(countForTaxon([], 'Boletus'), 0)
})
