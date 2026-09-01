import test from 'node:test'
import assert from 'node:assert/strict'
import qrcode from 'qrcode-generator'

// useShareState needs a Nuxt runtime, so the round-trip is exercised through the
// same URLSearchParams handling the composable uses, plus the QR encoder that
// backs the share panel. What matters is that a link survives the trip and that
// a malformed one cannot break the app.

const buildUrl = (params, path = '/map', origin = 'https://example.test') => {
  const qs = new URLSearchParams(params).toString()
  return `${origin}${path}${qs ? `?${qs}` : ''}`
}
const readParams = (url) => Object.fromEntries(new URL(url).searchParams.entries())

test('view params survive a round trip through the URL', () => {
  const params = {
    c: '40.0150,-105.2705', z: '11', color: 'species', ov: 'hotspots', d: '200', w: '14',
  }
  assert.deepEqual(readParams(buildUrl(params)), params)
})

test('a species filter with spaces and pipes round-trips intact', () => {
  const species = ['Amanita muscaria', 'Morchella esculenta', 'Boletus rubriceps']
  const url = buildUrl({ sp: species.join('|') })
  assert.deepEqual(readParams(url).sp.split('|'), species)
  // The separator must be encoded, not left to split the query string.
  assert.ok(!url.includes(' '), 'spaces must be percent-encoded')
})

test('an empty view produces a bare URL', () => {
  assert.equal(buildUrl({}), 'https://example.test/map')
})

test('the embed flag is additive, not a different route', () => {
  const url = new URL(buildUrl({ ov: 'density' }))
  url.searchParams.set('embed', '1')
  assert.equal(url.pathname, '/map')
  assert.equal(url.searchParams.get('ov'), 'density')
  assert.equal(url.searchParams.get('embed'), '1')
})

test('share targets encode the URL rather than splicing it in raw', () => {
  const link = buildUrl({ sp: 'Amanita muscaria', ov: 'season' })
  const text = '48,233 mushroom observations — data-map'
  const targets = [
    `https://twitter.com/intent/tweet?url=${encodeURIComponent(link)}&text=${encodeURIComponent(text)}`,
    `mailto:?subject=${encodeURIComponent(text)}&body=${encodeURIComponent(`${text}\n\n${link}`)}`,
    `sms:?&body=${encodeURIComponent(`${text} ${link}`)}`,
  ]
  for (const t of targets) {
    // An unencoded & or # inside the payload would truncate the shared link.
    const payload = t.slice(t.indexOf('?') + 1)
    assert.ok(!payload.includes('—'), 'the em dash must be encoded')
    assert.ok(!/[?&][a-z]+=[^&]*\s/.test(t), 'no raw spaces in a share target')
  }
})

test('a QR code encodes a realistic share link', () => {
  const link = buildUrl({
    c: '40.0150,-105.2705', z: '11', color: 'species', ov: 'hotspots', d: '200', w: '14',
    pal: 'okabe',
  })
  const qr = qrcode(0, 'M')
  qr.addData(link)
  qr.make()
  assert.ok(qr.getModuleCount() > 0)
  const svg = qr.createSvgTag({ cellSize: 4, margin: 8, scalable: true })
  assert.match(svg, /^<svg/)
})

test('an over-long link fails loudly so the panel can fall back', () => {
  // A QR code has a hard capacity; the panel catches this and tells the reader
  // to use the link instead of silently rendering nothing.
  const huge = buildUrl({ sp: Array.from({ length: 400 }, (_, i) => `Species name ${i}`).join('|') })
  assert.throws(() => {
    const qr = qrcode(0, 'M')
    qr.addData(huge)
    qr.make()
  })
})

test('malformed coordinates are detectable rather than becoming NaN silently', () => {
  for (const bad of ['', 'nonsense', '1', 'a,b']) {
    const [lat, lng] = String(bad).split(',').map(Number)
    assert.ok(!(Number.isFinite(lat) && Number.isFinite(lng)),
      `"${bad}" must not parse as a coordinate pair`)
  }
  const [lat, lng] = '40.015,-105.27'.split(',').map(Number)
  assert.ok(Number.isFinite(lat) && Number.isFinite(lng))
})
