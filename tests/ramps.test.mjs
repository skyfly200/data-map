/* Colour ramps of any length.
 *
 * Two ends was enough for "more is darker" and nothing else. The risk in
 * generalising it is all in the edges: a stored ramp from a version that only
 * knew about pairs, an editor that can empty a ramp until it is not one, a
 * value outside the domain landing off the end of the list.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  MAX_STOPS, MIN_STOPS,
  addStop, gradientCss, mix, normaliseStops, rampColor, removeStop, reverseStops, setStop, toHex,
} from '../composables/ramps.js'

// ── Parsing ──────────────────────────────────────────────────────────────────

test('hex is accepted in both lengths and normalised to one', () => {
  assert.equal(toHex('#A1B2C3'), '#a1b2c3')
  assert.equal(toHex('  #abc '), '#aabbcc')
  assert.equal(toHex('#fff'), '#ffffff')
})

test('anything that is not a colour is null rather than a guess', () => {
  assert.equal(toHex('red'), null)
  assert.equal(toHex('#12345'), null)
  assert.equal(toHex(''), null)
  assert.equal(toHex(null), null)
  assert.equal(toHex(undefined), null)
})

test('a ramp that cannot be read falls back rather than being repaired', () => {
  // These come from stored preferences that sync between devices and from
  // versions that only knew about pairs. Half a ramp is not a ramp.
  assert.equal(normaliseStops(['#000000']), null, 'one stop is not a ramp')
  assert.equal(normaliseStops(['#000000', 'nonsense']), null, 'one survivor is not a ramp')
  assert.equal(normaliseStops([]), null)
  assert.equal(normaliseStops('#000000'), null)
  assert.equal(normaliseStops(null), null)
})

test('a two-stop ramp still reads, because that is every ramp already stored', () => {
  assert.deepEqual(normaliseStops(['#e8f1fb', '#0b3d91']), ['#e8f1fb', '#0b3d91'])
  assert.equal(MIN_STOPS, 2)
})

test('an absurdly long ramp is trimmed rather than refused', () => {
  const many = Array.from({ length: 30 }, () => '#123456')
  assert.equal(normaliseStops(many).length, MAX_STOPS)
})

// ── Reading a colour off one ──────────────────────────────────────────────────

test('a ramp hits each of its own stops', () => {
  const stops = ['#000000', '#ff0000', '#ffffff']
  assert.equal(rampColor(stops, 0), '#000000')
  assert.equal(rampColor(stops, 0.5), '#ff0000')
  assert.equal(rampColor(stops, 1), '#ffffff')
})

test('stops are evenly spaced, so a new one lands where it was added', () => {
  const four = ['#000000', '#404040', '#808080', '#ffffff']
  assert.equal(rampColor(four, 1 / 3), '#404040')
  assert.equal(rampColor(four, 2 / 3), '#808080')
})

test('a value past either end clamps rather than running off the list', () => {
  const stops = ['#000000', '#ffffff']
  assert.equal(rampColor(stops, -1), '#000000')
  assert.equal(rampColor(stops, 2), '#ffffff')
  // NaN is what an unparseable value produces and must not index past the end.
  assert.equal(rampColor(stops, NaN), '#000000')
  assert.equal(rampColor(['#000000', '#888888', '#ffffff'], 1.0000001), '#ffffff')
})

test('a degenerate ramp still yields a colour instead of throwing', () => {
  assert.equal(rampColor(['#123456'], 0.7), '#123456')
  assert.match(rampColor([], 0.5), /^#[0-9a-f]{6}$/)
  assert.match(rampColor(null, 0.5), /^#[0-9a-f]{6}$/)
})

test('the middle of two colours is between them', () => {
  assert.equal(mix('#000000', '#ffffff', 0.5), '#808080')
  assert.equal(mix('#000000', '#ffffff', 0), '#000000')
  assert.equal(mix('#000000', '#ffffff', 1), '#ffffff')
})

// ── Editing ──────────────────────────────────────────────────────────────────

test('adding a stop changes the ramp\'s length and not its appearance', () => {
  // The stop takes the colour the ramp already had at that point, so adding one
  // is a step towards an edit rather than an edit — the ramp on screen when you
  // press + is still the ramp afterwards.
  const before = ['#000000', '#ffffff']
  const after = addStop(before, 0)
  assert.equal(after.length, 3)
  assert.equal(after[1], '#808080')
  // Within a rounding unit, not bit-identical: going via the midpoint rounds
  // each channel twice, so 0.75 of the way along black-to-white comes out 192
  // rather than 191. That is invisible, and demanding equality here would be
  // asserting the arithmetic's rounding rather than the property that matters.
  const channels = (hex) => [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16))
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const a = channels(rampColor(after, t))
    const b = channels(rampColor(before, t))
    a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= 1,
      `t=${t} moved: ${rampColor(after, t)} vs ${rampColor(before, t)}`))
  }
  // The ends are exact, because they are the same stops.
  assert.equal(rampColor(after, 0), rampColor(before, 0))
  assert.equal(rampColor(after, 1), rampColor(before, 1))
})

test('a stop is added in the section it was asked for', () => {
  const stops = ['#000000', '#ff0000', '#ffffff']
  assert.deepEqual(addStop(stops, 1), ['#000000', '#ff0000', '#ff8080', '#ffffff'])
})

test('a ramp cannot grow past the readable limit', () => {
  let stops = ['#000000', '#ffffff']
  for (let i = 0; i < 20; i += 1) stops = addStop(stops, 0)
  assert.equal(stops.length, MAX_STOPS)
})

test('a ramp cannot be emptied below two stops', () => {
  // The editor offers a remove button per stop, and the one thing it must not
  // do is leave something that is no longer a ramp.
  const two = ['#000000', '#ffffff']
  assert.deepEqual(removeStop(two, 0), two)
  assert.deepEqual(removeStop(two, 1), two)
  assert.deepEqual(removeStop(['#000000', '#888888', '#ffffff'], 1), ['#000000', '#ffffff'])
})

test('removing a stop that is not there leaves the ramp alone', () => {
  const three = ['#000000', '#888888', '#ffffff']
  assert.deepEqual(removeStop(three, 9), three)
  assert.deepEqual(removeStop(three, -1), three)
})

test('a stop can be recoloured, and a bad colour is ignored', () => {
  const stops = ['#000000', '#ffffff']
  assert.deepEqual(setStop(stops, 1, '#ff0000'), ['#000000', '#ff0000'])
  assert.deepEqual(setStop(stops, 1, 'chartreuse'), stops, 'an unparseable colour changes nothing')
  assert.deepEqual(setStop(stops, 5, '#ff0000'), stops, 'an index off the end changes nothing')
})

test('reversing a ramp flips it and nothing else', () => {
  assert.deepEqual(reverseStops(['#000000', '#888888', '#ffffff']),
    ['#ffffff', '#888888', '#000000'])
  // And is its own inverse.
  const r = ['#112233', '#445566', '#778899']
  assert.deepEqual(reverseStops(reverseStops(r)), r)
})

test('editing never mutates the ramp it was given', () => {
  const stops = ['#000000', '#ffffff']
  const copy = [...stops]
  addStop(stops, 0); removeStop(stops, 0); setStop(stops, 0, '#ff0000'); reverseStops(stops)
  assert.deepEqual(stops, copy)
})

// ── Rendering ────────────────────────────────────────────────────────────────

test('the css gradient carries every stop, in order', () => {
  assert.equal(gradientCss(['#000000', '#888888', '#ffffff']),
    'linear-gradient(90deg, #000000, #888888, #ffffff)')
  assert.match(gradientCss(['#000000', '#ffffff'], '180deg'), /^linear-gradient\(180deg,/)
})

test('an unusable ramp still renders something rather than breaking the style', () => {
  // This lands in a style attribute; a broken value takes the element's whole
  // background with it, so the swatch would vanish rather than look wrong.
  assert.match(gradientCss(null), /^linear-gradient\(90deg, #[0-9a-f]{6}, #[0-9a-f]{6}\)$/)
  assert.match(gradientCss(['nonsense']), /^linear-gradient\(/)
})
