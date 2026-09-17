/* The list of taxa the app asks iNaturalist for.
 *
 * This list ends up in two places that cannot be allowed to disagree: a query
 * to somebody else's API, and an environment variable pasted into the Python
 * pipeline. So the parsing has to be the same parsing at both ends, and it has
 * to refuse what is not a name rather than pass it along and let the failure
 * happen somewhere with less context.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { readFileSync } from 'node:fs'

import {
  FRMS_GENERA, MAX_TAXA, TAXON_PRESETS,
  asEnvLine, formatTaxa, isTaxonName, matchingPreset, parseTaxa, rejectedTaxa, scaleOf,
} from '../composables/taxonList.js'

// ── What counts as a name ────────────────────────────────────────────────────

test('a taxon name at any rank is accepted', () => {
  for (const name of ['Amanita', 'Amanita muscaria', 'Amanitaceae', 'Fungi',
    'Agaricales', 'Boletus edulis', "Cortinarius o'brienii", 'Salix × fragilis',
    'Amanita sp.', 'Tricholoma murrillianum']) {
    assert.ok(isTaxonName(name), `${name} was refused`)
  }
})

test('what is not a name is refused rather than sent to somebody else\'s API', () => {
  for (const junk of ['', ' ', 'a', 'ab', '123', 'Amanita<script>', 'a&b=c',
    '../../etc/passwd', 'SELECT *', 'Amanita;DROP', '%20', null, undefined,
    'x'.repeat(81)]) {
    assert.ok(!isTaxonName(junk), `${JSON.stringify(junk)} was accepted`)
  }
})

test('accented and non-Latin letters are names too', () => {
  // The regex is on letters, not on ASCII. A list that silently dropped these
  // would be a list that quietly worked for some people.
  assert.ok(isTaxonName('Peziza×badia'))
  assert.ok(isTaxonName('Grüne'))
})

// ── Parsing a list ───────────────────────────────────────────────────────────

test('a comma-separated string becomes a list', () => {
  assert.deepEqual(parseTaxa('Amanita, Boletus, Morchella'), ['Amanita', 'Boletus', 'Morchella'])
  assert.deepEqual(parseTaxa(['Amanita', 'Boletus']), ['Amanita', 'Boletus'])
})

test('blanks and duplicates go, and the first spelling stays', () => {
  // "amanita" and "Amanita" are one genus. The one worth keeping is the one
  // written the way every field guide writes it.
  assert.deepEqual(parseTaxa(' Amanita , , amanita, AMANITA '), ['Amanita'])
  assert.deepEqual(parseTaxa('boletus, Boletus'), ['boletus'])
})

test('runs of whitespace inside a name collapse', () => {
  assert.deepEqual(parseTaxa('Amanita   muscaria'), ['Amanita muscaria'])
  assert.deepEqual(parseTaxa('Amanita\tmuscaria'), ['Amanita muscaria'])
})

test('an entry that is not a name is dropped, and can be listed', () => {
  assert.deepEqual(parseTaxa('Amanita, 42, Boletus'), ['Amanita', 'Boletus'])
  assert.deepEqual(rejectedTaxa('Amanita, 42, <b>, Boletus'), ['42', '<b>'])
  assert.deepEqual(rejectedTaxa('Amanita, Boletus'), [])
})

test('parsing nothing is an empty list rather than a throw', () => {
  for (const empty of ['', '  ', ',,,', [], null, undefined]) {
    assert.deepEqual(parseTaxa(empty), [], `${JSON.stringify(empty)} did not come back empty`)
  }
})

test('a list cannot grow past what a queue can be watched for', () => {
  const many = Array.from({ length: MAX_TAXA + 50 }, (_, i) => `Genus${'abcdefghij'[i % 10]}${i}`)
  // Digits are not names, so build them out of letters only.
  const letters = many.map((_, i) => `Genus${String.fromCharCode(97 + (i % 26))}${String.fromCharCode(97 + Math.floor(i / 26))}`)
  assert.ok(parseTaxa(letters).length <= MAX_TAXA)
})

test('the canonical form round-trips', () => {
  const text = formatTaxa(['Boletus', ' amanita ', 'Boletus'])
  assert.equal(text, 'Boletus, amanita')
  assert.deepEqual(parseTaxa(text), ['Boletus', 'amanita'])
})

// ── The pipeline's environment variable ──────────────────────────────────────

test('the env line is what the Python pipeline parses', () => {
  // Lower-cased, because that is what scripts/iNat.py parse_species_list does
  // with it. A line that round-trips unchanged is one nobody has to wonder
  // about when they compare the two.
  assert.equal(asEnvLine(['Amanita', 'Boletus edulis']),
    'INAT_TAXON_NAME=amanita, boletus edulis')
  assert.equal(asEnvLine([]), 'INAT_TAXON_NAME=')
})

test('the shipped default is still the list the pipeline ships with', () => {
  // The app's default and the .env.example the pipeline reads are two copies of
  // one decision, and this is what stops them drifting.
  const env = readFileSync(new URL('../.env.example', import.meta.url), 'utf8')
  const line = env.split(/\r?\n/).find((l) => l.startsWith('SPECIES='))
  assert.ok(line, 'the example env no longer sets SPECIES')
  const shipped = parseTaxa(line.slice('SPECIES='.length)).map((t) => t.toLowerCase())
  const ours = FRMS_GENERA.map((t) => t.toLowerCase())
  assert.deepEqual(ours, shipped)
})

// ── Presets ──────────────────────────────────────────────────────────────────

test('every preset is a usable list with a scale and a note', () => {
  for (const p of TAXON_PRESETS) {
    assert.ok(p.key && p.label, 'a preset has no name')
    assert.ok(p.taxa.length, `${p.key} is empty`)
    assert.ok(p.scale && p.note, `${p.key} does not say what it costs`)
    assert.deepEqual(parseTaxa(p.taxa), p.taxa, `${p.key} does not survive its own parser`)
  }
  const keys = TAXON_PRESETS.map((p) => p.key)
  assert.equal(new Set(keys).size, keys.length)
})

test('a whole kingdom is one name, not a list of its members', () => {
  // iNaturalist returns everything below a taxon, so asking for Fungi is asking
  // for every fungus. A preset that listed them would be both wrong and huge.
  for (const key of ['fungi', 'plants', 'animals']) {
    assert.equal(TAXON_PRESETS.find((p) => p.key === key).taxa.length, 1)
  }
})

test('a list is recognised as the preset it matches, whatever the order', () => {
  assert.equal(matchingPreset(['Fungi']).key, 'fungi')
  assert.equal(matchingPreset(['Plantae', 'Fungi']).key, 'fungi-plants')
  assert.equal(matchingPreset(FRMS_GENERA).key, 'frms')
  assert.equal(matchingPreset([...FRMS_GENERA].reverse()).key, 'frms')
  assert.equal(matchingPreset(['fungi']).key, 'fungi', 'case decided which preset it was')
})

test('somebody\'s own list matches no preset', () => {
  assert.equal(matchingPreset(['Amanita']), null)
  assert.equal(matchingPreset([...FRMS_GENERA, 'Fungi']), null)
  assert.equal(matchingPreset([]), null)
})

// ── What a list will cost ────────────────────────────────────────────────────

test('a kingdom of plants or animals is called huge', () => {
  assert.equal(scaleOf(['Plantae']), 'huge')
  assert.equal(scaleOf(['Animalia']), 'huge')
  assert.equal(scaleOf(['Amanita', 'plantae']), 'huge', 'case hid the kingdom')
})

test('all fungi is large, and forty genera is ordinary', () => {
  assert.equal(scaleOf(['Fungi']), 'large')
  assert.equal(scaleOf(FRMS_GENERA), 'ordinary')
  assert.equal(scaleOf(['Amanita']), 'ordinary')
  assert.equal(scaleOf([]), 'ordinary')
})

test('a very long list of genera is large even with no kingdom in it', () => {
  const many = Array.from({ length: 80 },
    (_, i) => `Genus${String.fromCharCode(97 + (i % 26))}${String.fromCharCode(97 + Math.floor(i / 26))}`)
  assert.equal(scaleOf(many), 'large')
})
