/* Layers an administrator registers against their own Earth Engine assets.
 *
 * The validation here is a security boundary. An asset ID and a band name are
 * interpolated into Earth Engine calls, so they are checked against a strict
 * shape rather than trusted — an administrator is trusted, a pasted string is
 * not, and the two are not the same thing.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  CUSTOM_PREFIX, CustomLayerError, buildCustomLayer, describeCustomLayer,
  isCustomKey, normaliseCustomLayer, slugFromKey,
  validateAssetId, validateBand, validatePalette, validateSlug,
} from '../netlify/lib/ee-custom-layers.mjs'

const base = {
  name: 'Chanterelle habitat',
  asset_id: 'projects/my-project/assets/chanterelle',
  palette: '#f7fcb9,#31a354',
}

// ── Asset IDs ────────────────────────────────────────────────────────────────

test('the three legal asset id shapes are accepted', () => {
  for (const id of [
    'projects/my-project/assets/chanterelle',
    'users/somebody/my-asset',
    'MODIS/061/MCD64A1',
    'COPERNICUS/S2_SR_HARMONIZED',
    'projects/ee-society/assets/2026/burn_index-v2',
  ]) {
    assert.equal(validateAssetId(id), id)
  }
})

test('anything that is not a name is refused', () => {
  // These are the ones that matter: the string reaches ee.Image(...), so it
  // must not be able to carry an expression, a path climb, or whitespace that
  // could split it.
  for (const bad of [
    'projects/x/assets/../../etc',
    'projects/x/assets/a b',
    "projects/x'); ee.Image(1); //",
    'projects/x/assets/a\nb',
    '/leading/slash',
    'trailing/slash/',
    'double//slash',
    'projects/x/assets/(y)',
    '',
    '   ',
    null,
    undefined,
  ]) {
    assert.throws(() => validateAssetId(bad), CustomLayerError, `accepted ${JSON.stringify(bad)}`)
  }
})

test('a single dot or double dot part is refused on its own', () => {
  assert.throws(() => validateAssetId('projects/./assets/x'), CustomLayerError)
  assert.throws(() => validateAssetId('projects/../assets/x'), CustomLayerError)
  // A dot INSIDE a part is fine — real asset names contain versions.
  assert.equal(validateAssetId('projects/x/assets/v1.2'), 'projects/x/assets/v1.2')
})

test('absurd lengths are refused rather than sent', () => {
  assert.throws(() => validateAssetId('a/'.repeat(200) + 'b'), CustomLayerError)
  assert.throws(() => validateAssetId(`projects/${'x'.repeat(200)}/assets/y`), CustomLayerError)
})

// ── Bands ────────────────────────────────────────────────────────────────────

test('a band name is a name, and optional', () => {
  assert.equal(validateBand('B8'), 'B8')
  assert.equal(validateBand('soil_moisture'), 'soil_moisture')
  assert.equal(validateBand(''), '')
  assert.equal(validateBand(null), '')
  assert.throws(() => validateBand('a b'), CustomLayerError)
  assert.throws(() => validateBand("b'); //"), CustomLayerError)
  assert.throws(() => validateBand('', { required: true }), CustomLayerError)
})

// ── Palettes ─────────────────────────────────────────────────────────────────

test('a palette is parsed from text or an array, with or without hashes', () => {
  assert.deepEqual(validatePalette('#f7fcb9, #31a354'), ['#f7fcb9', '#31a354'])
  assert.deepEqual(validatePalette('f7fcb9 31a354'), ['#f7fcb9', '#31a354'])
  assert.deepEqual(validatePalette(['#F7FCB9']), ['#f7fcb9'])
  assert.deepEqual(validatePalette(''), [])
})

test('a palette entry that is not a colour is refused', () => {
  // Palette entries go into the visualisation, so "red" or a CSS function must
  // not reach it.
  for (const bad of ['red', '#fff', 'rgb(1,2,3)', '#12345g']) {
    assert.throws(() => validatePalette(bad), CustomLayerError, `accepted ${bad}`)
  }
})

test('a one-colour palette is refused, because a ramp needs two ends', () => {
  assert.throws(() => normaliseCustomLayer({ ...base, palette: '#f7fcb9' }), CustomLayerError)
  // None at all is fine: Earth Engine renders the band in greyscale.
  assert.deepEqual(normaliseCustomLayer({ ...base, palette: '' }).palette, [])
})

// ── Slugs ────────────────────────────────────────────────────────────────────

test('a slug is url-safe and derived from the name when absent', () => {
  assert.equal(validateSlug('Burn-Index'), 'burn-index')
  assert.throws(() => validateSlug('a'), CustomLayerError)
  assert.throws(() => validateSlug('has space'), CustomLayerError)
  assert.throws(() => validateSlug('../x'), CustomLayerError)
  assert.equal(normaliseCustomLayer(base).slug, 'chanterelle-habitat')
})

// ── The whole record ─────────────────────────────────────────────────────────

test('a name is required, because it is what the layer is called', () => {
  assert.throws(() => normaliseCustomLayer({ ...base, name: '' }), CustomLayerError)
})

test('the visualisation range must go upwards', () => {
  assert.throws(() => normaliseCustomLayer({ ...base, vis_min: 5, vis_max: 1 }), CustomLayerError)
  assert.throws(() => normaliseCustomLayer({ ...base, vis_min: 1, vis_max: 1 }), CustomLayerError)
  const ok = normaliseCustomLayer({ ...base, vis_min: 0, vis_max: 1 })
  assert.equal(ok.vis_min, 0)
  assert.equal(ok.vis_max, 1)
})

test('opacity is clamped rather than refused', () => {
  // A cosmetic value out of range is a slip, not a mistake worth a red message.
  assert.equal(normaliseCustomLayer({ ...base, opacity: 5 }).opacity, 1)
  assert.equal(normaliseCustomLayer({ ...base, opacity: -1 }).opacity, 0.05)
  assert.equal(normaliseCustomLayer({ ...base, opacity: '' }).opacity, 0.8)
})

test('tier defaults to members and only accepts the three that exist', () => {
  assert.equal(normaliseCustomLayer(base).tier, 'member')
  assert.equal(normaliseCustomLayer({ ...base, tier: 'free' }).tier, 'free')
  assert.equal(normaliseCustomLayer({ ...base, tier: 'admin' }).tier, 'admin')
  // Anything else falls to the safe end rather than throwing, so a stale client
  // cannot accidentally publish a layer.
  assert.equal(normaliseCustomLayer({ ...base, tier: 'everyone' }).tier, 'member')
  assert.equal(normaliseCustomLayer({ ...base, tier: 'superuser' }).tier, 'member')
})

test('collection-only fields are dropped for a single image', () => {
  // Otherwise a layer switched from a collection to an image keeps a reducer
  // and a date range that nothing reads, which is a puzzle on the next edit.
  const img = normaliseCustomLayer({
    ...base, asset_type: 'image', reducer: 'mean', date_from: '2026-01-01',
  })
  assert.equal(img.reducer, null)
  assert.equal(img.date_from, null)
})

test('a collection needs a reducer the builder understands', () => {
  assert.throws(() => normaliseCustomLayer({
    ...base, asset_type: 'image_collection', reducer: 'average',
  }), CustomLayerError)
  const ok = normaliseCustomLayer({ ...base, asset_type: 'image_collection', reducer: 'median' })
  assert.equal(ok.reducer, 'median')
})

test('dates must be dates and in order', () => {
  const c = { ...base, asset_type: 'image_collection' }
  assert.throws(() => normaliseCustomLayer({ ...c, date_from: 'last year' }), CustomLayerError)
  assert.throws(() => normaliseCustomLayer({ ...c, date_from: '2026-06-01', date_to: '2026-01-01' }),
    CustomLayerError)
})

// ── Building the image ───────────────────────────────────────────────────────

function stubEe() {
  const calls = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => {
      if (prop === 'then') return undefined
      calls.push(String(prop))
      return chain
    },
    apply: () => chain,
  })
  const ee = {
    Image: (id) => { calls.push(`Image(${id})`); return chain },
    ImageCollection: (id) => { calls.push(`ImageCollection(${id})`); return chain },
  }
  return { ee, calls, chain }
}

test('a single image is loaded and its band selected', () => {
  const { ee, calls } = stubEe()
  const layer = normaliseCustomLayer({ ...base, band: 'b1' })
  const { vis } = buildCustomLayer(ee, { ...layer, vis_min: 0, vis_max: 1 })
  assert.ok(calls.includes(`Image(${layer.asset_id})`))
  assert.ok(calls.includes('select'))
  // Earth Engine wants bare hex, not CSS hex.
  assert.deepEqual(vis.palette, ['f7fcb9', '31a354'])
})

test('a collection is filtered, selected and reduced with the named reducer', () => {
  const { ee, calls } = stubEe()
  const layer = normaliseCustomLayer({
    ...base, asset_type: 'image_collection', reducer: 'median',
    band: 'NDVI', date_from: '2026-01-01', date_to: '2026-06-30',
  })
  buildCustomLayer(ee, layer)
  assert.ok(calls.includes(`ImageCollection(${layer.asset_id})`))
  assert.ok(calls.includes('filterDate'))
  assert.ok(calls.includes('select'))
  assert.ok(calls.includes('median'), 'the chosen reducer should be the one called')
})

test('mask_below masks, and its absence does not', () => {
  const masked = stubEe()
  buildCustomLayer(masked.ee, normaliseCustomLayer({ ...base, mask_below: 0 }))
  assert.ok(masked.calls.includes('updateMask'))

  const plain = stubEe()
  buildCustomLayer(plain.ee, normaliseCustomLayer(base))
  assert.ok(!plain.calls.includes('updateMask'))
})

test('an absent min or max is simply not sent', () => {
  // Earth Engine picks its own stretch, which is better than a made-up range.
  const { ee } = stubEe()
  const { vis } = buildCustomLayer(ee, normaliseCustomLayer(base))
  assert.ok(!('min' in vis))
  assert.ok(!('max' in vis))
})

// ── Keys and descriptions ────────────────────────────────────────────────────

test('custom keys are namespaced so they cannot collide with a built-in', () => {
  const d = describeCustomLayer(normaliseCustomLayer(base))
  assert.equal(d.key, `${CUSTOM_PREFIX}chanterelle-habitat`)
  assert.ok(isCustomKey(d.key))
  assert.equal(slugFromKey(d.key), 'chanterelle-habitat')
  assert.ok(!isCustomKey('years-since-fire'))
})

test('a description carries the tier, so the picker can mark it', () => {
  const d = describeCustomLayer(normaliseCustomLayer({ ...base, tier: 'free' }))
  assert.equal(d.tier, 'free')
  assert.equal(d.custom, true)
  assert.ok(d.legend, 'a palette should produce a key')
})

test('a layer with no palette has no key rather than an empty one', () => {
  const d = describeCustomLayer(normaliseCustomLayer({ ...base, palette: '' }))
  assert.equal(d.legend, null)
})
