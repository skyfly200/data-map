/* How a tile request is read.
 *
 * Two things moved when the soil taxonomy selection grew past what a URL will
 * carry: parameters can now arrive in a POST body, and a cache key can now be
 * longer than a blob store will accept as a name. Both are the kind of change
 * that works on every layer but one and then quietly fails on the one — a body
 * that is ignored renders the default selection, and a key that is too long
 * either throws or, worse, gets truncated into a collision.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { blobId, readInput } from '../netlify/functions/ee-tiles.mjs'
import { cacheKey, normaliseCodes, resolveLayer } from '../netlify/lib/ee-tile-layers.mjs'

const BASE = 'https://example.org/.netlify/functions/ee-tiles'

const get = (qs = '') => ({ request: new Request(`${BASE}${qs}`), url: new URL(`${BASE}${qs}`) })
const post = (body, qs = '') => ({
  request: new Request(`${BASE}${qs}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  }),
  url: new URL(`${BASE}${qs}`),
})

// ── Reading the request ──────────────────────────────────────────────────────

test('a GET is read from the query string, as it always was', async () => {
  const { request, url } = get('?layer=slope&through=2024')
  assert.deepEqual(await readInput(request, url), { layer: 'slope', through: '2024' })
})

test('a GET with nothing on it is an empty set of parameters, not an error', async () => {
  const { request, url } = get()
  assert.deepEqual(await readInput(request, url), {})
})

test('a POST is read from its body', async () => {
  const { request, url } = post({ layer: 'soil-taxonomy-select', codes: [18, 213] })
  const input = await readInput(request, url)
  assert.equal(input.layer, 'soil-taxonomy-select')
  assert.deepEqual(input.codes, [18, 213])
})

test('a POST may still carry a query string, and the body wins', async () => {
  const { request, url } = post({ codes: [1, 2] }, '?layer=soil-taxonomy-select&codes=99')
  const input = await readInput(request, url)
  assert.equal(input.layer, 'soil-taxonomy-select')
  assert.deepEqual(input.codes, [1, 2])
})

test('a body that is not an object is ignored rather than trusted', async () => {
  // A bare array or a string would spread into nonsense keys, and "0", "1", "2"
  // are not parameters of anything.
  for (const body of [[1, 2, 3], '"hello"', '42', 'null']) {
    const { request, url } = post(body, '?layer=slope')
    assert.deepEqual(await readInput(request, url), { layer: 'slope' },
      `body ${JSON.stringify(body)} leaked into the parameters`)
  }
})

test('a body that is not JSON is refused by name rather than silently ignored', async () => {
  const { request, url } = post('{not json', '?layer=slope')
  await assert.rejects(() => readInput(request, url), /body could not be read as JSON/)
})

test('what a POST produces validates exactly as what a GET produces', async () => {
  // The whole point: one validation path, whichever way the parameters arrived.
  const codes = [213, 18, 18]
  const fromPost = await readInput(...Object.values(post({ layer: 'soil-taxonomy-select', codes })))
  const fromGet = await readInput(...Object.values(get('?layer=soil-taxonomy-select&codes=213,18,18')))
  assert.equal(
    resolveLayer('soil-taxonomy-select', fromPost).params.codes,
    resolveLayer('soil-taxonomy-select', fromGet).params.codes,
  )
  assert.equal(resolveLayer('soil-taxonomy-select', fromPost).params.codes, '18,213')
})

// ── The cache key ────────────────────────────────────────────────────────────

test('a short key is left alone, so a cache entry still says what it is', () => {
  const id = blobId('slope')
  assert.equal(id, 'slope')
  assert.equal(blobId('years-since-fire|through=2024|window=12'),
    'years-since-fire|through=2024|window=12')
})

test('a key too long to be a blob name is hashed, and still names its layer', () => {
  const all = Array.from({ length: 430 }, (_, i) => i + 1)
  const { params } = resolveLayer('soil-taxonomy-select', { codes: all })
  const long = cacheKey('soil-taxonomy-select', params)
  assert.ok(long.length > 1000, 'the selection did not produce a long key')

  const id = blobId(long)
  assert.ok(id.length <= 120, `the hashed key is still ${id.length} characters`)
  assert.match(id, /^soil-taxonomy-select/, 'the key no longer says which layer it belongs to')
})

test('two different selections cannot share a cache entry', () => {
  // The property the whole cache rests on. The readable prefix is identical
  // between these two, so only the hash separates them.
  const a = Array.from({ length: 430 }, (_, i) => i + 1)
  const b = [...a.slice(0, 429), 999]
  const keyA = cacheKey('soil-taxonomy-select', resolveLayer('soil-taxonomy-select', { codes: a }).params)
  const keyB = cacheKey('soil-taxonomy-select', resolveLayer('soil-taxonomy-select', { codes: b }).params)
  assert.notEqual(keyA, keyB)
  assert.notEqual(blobId(keyA), blobId(keyB))
})

test('the same selection always produces the same cache entry', () => {
  const codes = normaliseCodes([5, 3, 1])
  const again = normaliseCodes([1, 3, 5, 5])
  assert.equal(blobId(cacheKey('soil-taxonomy-select', { codes })),
    blobId(cacheKey('soil-taxonomy-select', { codes: again })))
})
