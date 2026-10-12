import test from 'node:test'
import assert from 'node:assert/strict'
import { zipSync, strToU8, strFromU8 } from 'fflate'
import { GbifUploadError, entryMatches, safeUnzip } from '../netlify/lib/gbif-upload.mjs'

const archive = zipSync({
  'meta.xml': strToU8('<archive/>'),
  'occurrence.txt': strToU8('gbifID\tdecimalLatitude\n1\t40'),
  'verbatim.txt': strToU8('x'.repeat(5000)),
})

test('only the wanted entries are extracted', () => {
  const files = safeUnzip(archive, (n) => entryMatches(n, 'occurrence.txt'))
  assert.deepEqual(Object.keys(files), ['occurrence.txt'])
  assert.match(strFromU8(files['occurrence.txt']), /gbifID/)
})

test('an entry that expands past the limit is refused with 413', () => {
  assert.throws(
    () => safeUnzip(archive, (n) => entryMatches(n, 'verbatim.txt'), { maxEntry: 1000 }),
    (e) => e instanceof GbifUploadError && e.status === 413)
})

test('the total across entries is capped too', () => {
  assert.throws(
    () => safeUnzip(archive, () => true, { maxEntry: 10_000, maxTotal: 5_010 }),
    (e) => e instanceof GbifUploadError && e.status === 413)
})

test('entries match at the root or one folder down', () => {
  assert.equal(entryMatches('occurrence.txt', 'occurrence.txt'), true)
  assert.equal(entryMatches('0001-xyz/occurrence.txt', 'occurrence.txt'), true)
  assert.equal(entryMatches('not-occurrence.txt', 'occurrence.txt'), false)
})
