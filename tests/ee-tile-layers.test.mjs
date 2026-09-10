/* The Earth Engine tile catalogue.
 *
 * Two things are being pinned. The parameter checks, because those values are
 * interpolated into Earth Engine calls and a request naming a layer is the only
 * thing a member can send. And the shape of each entry, because a layer missing
 * a legend or a note is one a reader cannot interpret — which on a fire map
 * matters more than usual, since unpainted ground reads as "never burned".
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  EE_LAYER_CATALOGUE, EE_LAYER_KEYS, EE_TILE_LAYERS, LayerError,
  DEFAULT_TIER, MODIS_FIRST_YEAR, cacheKey, describeLayer, resolveLayer, tierFor,
} from '../netlify/lib/ee-tile-layers.mjs'

const YEAR = new Date().getUTCFullYear()

// ── The allowlist ────────────────────────────────────────────────────────────

test('an unknown layer is refused by name', () => {
  assert.throws(() => resolveLayer('../../etc/passwd'), LayerError)
  assert.throws(() => resolveLayer(''), LayerError)
  assert.throws(() => resolveLayer('rm -rf'), (err) => {
    assert.ok(err instanceof LayerError)
    assert.match(err.message, /Unknown layer/)
    return true
  })
})

test('every catalogue entry can be read and drawn', () => {
  for (const key of EE_LAYER_KEYS) {
    const spec = EE_TILE_LAYERS[key]
    assert.ok(spec.name, `${key} has no name`)
    assert.ok(spec.group, `${key} has no group`)
    assert.ok(spec.attribution, `${key} does not credit its source`)
    // A raster with no key is decoration. Every one of these is a measurement.
    assert.ok(spec.legend, `${key} has no legend`)
    assert.ok(spec.note, `${key} has no caveat`)
    assert.equal(typeof spec.build, 'function', `${key} has no build`)
  }
})

test('the described catalogue is safe to send to a browser', () => {
  for (const entry of EE_LAYER_CATALOGUE) {
    // build() closes over the Earth Engine asset ids; serialising it to the
    // client would publish the recipe and, worse, invite the client to send one.
    assert.ok(!('build' in entry), `${entry.key} leaks its build function`)
    assert.equal(typeof entry.key, 'string')
    assert.ok(JSON.stringify(entry).length > 0)
  }
})

// ── Parameters ───────────────────────────────────────────────────────────────

test('missing parameters fall back to their defaults', () => {
  const { params } = resolveLayer('years-since-fire')
  assert.equal(params.through, YEAR)
  assert.equal(params.window, 12)
})

test('a year outside the data is refused rather than clamped', () => {
  // Silently clamping would render a year the viewer did not ask for and label
  // it with the year they did.
  assert.throws(() => resolveLayer('years-since-fire', { through: 1850 }), LayerError)
  assert.throws(() => resolveLayer('years-since-fire', { through: YEAR + 5 }), LayerError)
  assert.throws(() => resolveLayer('burn-severity', { year: 1900 }), LayerError)
  // MODIS burned area starts in 2001; asking earlier would return an empty
  // collection, which paints as "nothing burned" rather than "no data".
  assert.throws(() => resolveLayer('burn-date', { year: MODIS_FIRST_YEAR - 1 }), LayerError)
})

test('a non-numeric parameter is refused', () => {
  for (const bad of ['last year', '2020; DROP', {}, [], 'NaN']) {
    assert.throws(() => resolveLayer('burn-severity', { year: bad }), LayerError,
      `accepted ${JSON.stringify(bad)}`)
  }
})

test('parameters are coerced to whole numbers', () => {
  const { params } = resolveLayer('active-fire', { days: '7.9' })
  assert.equal(params.days, 7)
  assert.equal(typeof params.days, 'number')
})

test('the edges of each range are accepted', () => {
  assert.equal(resolveLayer('active-fire', { days: 1 }).params.days, 1)
  assert.equal(resolveLayer('active-fire', { days: 60 }).params.days, 60)
  assert.throws(() => resolveLayer('active-fire', { days: 0 }), LayerError)
  assert.throws(() => resolveLayer('active-fire', { days: 61 }), LayerError)
})

test('an unknown parameter is ignored, not passed through', () => {
  // Only the schema's own keys reach the build function, so an extra query
  // parameter cannot become part of an Earth Engine call.
  const { params } = resolveLayer('burn-severity', { year: 2024, expression: 'ee.Image(1)' })
  assert.deepEqual(Object.keys(params), ['year'])
})

// ── The cache key ────────────────────────────────────────────────────────────

test('the cache key does not depend on parameter order', () => {
  // Two members asking for the same layer must share one minted tile URL; a
  // key that varied with key order would mint twice and bill twice.
  assert.equal(
    cacheKey('years-since-fire', { through: 2026, window: 12 }),
    cacheKey('years-since-fire', { window: 12, through: 2026 }),
  )
})

test('different parameters are different cache entries', () => {
  assert.notEqual(
    cacheKey('burn-severity', { year: 2024 }),
    cacheKey('burn-severity', { year: 2025 }),
  )
  assert.notEqual(
    cacheKey('burn-date', { year: 2024 }),
    cacheKey('burn-severity', { year: 2024 }),
  )
})

// ── Descriptions ─────────────────────────────────────────────────────────────

test('a description carries the schema the UI builds its controls from', () => {
  const d = describeLayer('years-since-fire')
  assert.equal(d.params.window.type, 'int')
  assert.equal(d.params.window.min, 2)
  assert.equal(d.params.window.max, 25)
  // Resolved to values, not left as the functions that compute them: these
  // cross to the browser as JSON.
  assert.equal(typeof d.params.through.max, 'number')
  assert.equal(typeof d.params.through.default, 'number')
})

test('describing an unknown layer returns nothing rather than throwing', () => {
  assert.equal(describeLayer('nope'), null)
})

// ── The recipes ──────────────────────────────────────────────────────────────

test('every build runs against a stubbed Earth Engine and paints something', () => {
  // A stub rather than a live session: what is being checked is that each
  // recipe is wired up — masks its no-data, names a palette, and produces an
  // image — not that Earth Engine agrees, which needs credentials.
  const calls = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => {
      if (prop === 'then') return undefined            // not a promise
      calls.push(String(prop))
      return chain
    },
    apply: () => chain,
  })
  const ee = new Proxy({}, {
    get: (t, prop) => {
      calls.push(String(prop))
      if (prop === 'Filter') return { lt: () => chain }
      if (prop === 'Image') {
        const img = () => chain
        img.constant = () => chain
        return img
      }
      return chain
    },
  })

  for (const key of EE_LAYER_KEYS) {
    calls.length = 0
    const { layer, params } = resolveLayer(key)
    const out = layer.build(ee, params)
    assert.ok(out.image, `${key} built no image`)
    assert.ok(out.vis, `${key} has no visualisation`)
    assert.ok(Array.isArray(out.vis.palette) && out.vis.palette.length,
      `${key} has no palette`)
    // Every one of these layers is about where something IS. Without a mask,
    // the whole world is painted the bottom of the ramp, which on a fire map
    // says everywhere burned.
    assert.ok(calls.includes('updateMask'), `${key} never masks its no-data`)
  }
})

test('years since fire looks back over the window it was given', () => {
  const dates = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => (prop === 'then' ? undefined : chain),
    apply: () => chain,
  })
  const ee = {
    ImageCollection: () => ({
      filterDate: (from, to) => { dates.push([from, to]); return chain },
      max: () => chain,
    }),
    Image: Object.assign(() => chain, { constant: () => chain }),
  }
  const { layer, params } = resolveLayer('years-since-fire', { through: 2026, window: 5 })
  layer.build(ee, params)
  // One filterDate per year in the window, 2022 through 2026.
  assert.equal(dates.length, 5)
  assert.equal(dates[0][0], '2022-01-01')
  assert.equal(dates.at(-1)[1], '2026-12-31')
})

test('the look-back never reaches before the data starts', () => {
  const dates = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => (prop === 'then' ? undefined : chain),
    apply: () => chain,
  })
  const ee = {
    ImageCollection: () => ({
      filterDate: (from) => { dates.push(from); return chain },
      max: () => chain,
    }),
    Image: Object.assign(() => chain, { constant: () => chain }),
  }
  const { layer, params } = resolveLayer('years-since-fire', { through: 2005, window: 25 })
  layer.build(ee, params)
  assert.equal(dates[0], `${MODIS_FIRST_YEAR}-01-01`)
})

// ── Tiers ────────────────────────────────────────────────────────────────────

test('the two layers that answer where to look are open to everyone', () => {
  // Years since fire and burn severity together answer "which burn scar is
  // worth walking next spring", which is the question the society exists to
  // help with. Gating it would gate the reason to visit at all.
  assert.equal(tierFor('years-since-fire'), 'free')
  assert.equal(tierFor('burn-severity'), 'free')
})

test('the layers that spend real compute per tile need membership', () => {
  // dNBR is calculated from raw Sentinel-2 scenes as you look at it, so it does
  // not share one cached render the way a published asset does.
  assert.equal(tierFor('dnbr'), DEFAULT_TIER)
  assert.equal(DEFAULT_TIER, 'member')
})

test('an unknown layer is gated at the strictest tier, not the loosest', () => {
  // A layer added to the catalogue without a tier, or a key that does not
  // exist, must not fall open.
  assert.equal(tierFor('nope'), DEFAULT_TIER)
  assert.equal(tierFor(''), DEFAULT_TIER)
  assert.equal(tierFor(undefined), DEFAULT_TIER)
})

test('every layer declares a tier the gate understands', () => {
  for (const entry of EE_LAYER_CATALOGUE) {
    assert.ok(['free', 'member', 'admin'].includes(entry.tier),
      `${entry.key} has an unusable tier: ${entry.tier}`)
  }
})
