import test from 'node:test'
import assert from 'node:assert/strict'

import { useImageExport } from '../composables/useImageExport.js'

// The rasterising paths need a DOM and are covered in the browser. What is worth
// pinning here is filename construction: these strings become files on someone's
// disk, so they must not carry characters a filesystem will reject.

const { slugify, stamp } = useImageExport()

test('a chart title becomes a safe filename stem', () => {
  assert.equal(slugify('Observations per environmental cluster'),
    'observations-per-environmental-cluster')
})

test('punctuation and separators are stripped, not escaped', () => {
  // Slashes and colons would create directories or break on Windows.
  assert.equal(slugify('Elevation vs. day of year'), 'elevation-vs-day-of-year')
  assert.equal(slugify('Species × land cover'), 'species-land-cover')
  assert.equal(slugify('7-day rain / total (mm)'), '7-day-rain-total-mm')
  for (const name of ['a/b', 'a\\b', 'a:b', 'a*b', 'a?b', 'a"b', 'a<b>', 'a|b']) {
    assert.ok(!/[/\\:*?"<>|]/.test(slugify(name)), `${name} left an unsafe character`)
  }
})

test('an empty or symbol-only title falls back rather than yielding a dotfile', () => {
  assert.equal(slugify(''), 'export')
  assert.equal(slugify(null), 'export')
  assert.equal(slugify('***'), 'export')
  assert.equal(slugify('   ', 'chart'), 'chart')
})

test('very long titles are truncated to a sane length', () => {
  const stem = slugify('x'.repeat(300))
  assert.ok(stem.length <= 60, `got ${stem.length} characters`)
})

test('leading and trailing separators are trimmed', () => {
  const stem = slugify('  Top species  ')
  assert.equal(stem, 'top-species')
  assert.ok(!stem.startsWith('-') && !stem.endsWith('-'))
})

test('the stamp is a sortable date', () => {
  assert.match(stamp(), /^\d{4}-\d{2}-\d{2}$/)
})

test('a full filename is safe end to end', () => {
  const name = `${slugify('Species × land cover')}-${stamp()}.png`
  assert.match(name, /^[a-z0-9-]+-\d{4}-\d{2}-\d{2}\.png$/)
})
