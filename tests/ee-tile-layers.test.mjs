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
  EE_LAYER_CATALOGUE, EE_LAYER_KEYS, EE_TILE_LAYERS, GAP_REMAP, LayerError,
  DEFAULT_TIER, MODIS_FIRST_YEAR, MODIS_LAG_YEARS, MTBS_LAG_YEARS,
  cacheKey, describeLayer, resolveLayer, tierFor,
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
  // Behind the dataset's publication lag, not the current year — see the
  // publication-lag tests below for why.
  assert.equal(params.through, YEAR - MODIS_LAG_YEARS)
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
      if (prop === 'Filter') return { lt: () => chain, eq: () => chain }
      if (prop === 'Image') {
        const img = () => chain
        img.constant = () => chain
        // cat() is how the multi-band soil composite is assembled.
        img.cat = () => chain
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

    // One band through a palette, or three bands as colour — never both, since
    // Earth Engine refuses a palette on a multi-band image.
    if (layer.rgb) {
      assert.ok(!out.vis.palette, `${key} is rgb but also declares a palette`)
      assert.ok(Number.isFinite(out.vis.min) && Number.isFinite(out.vis.max),
        `${key} is rgb and needs a stretch`)
    } else {
      assert.ok(Array.isArray(out.vis.palette) && out.vis.palette.length,
        `${key} has no palette`)
    }

    // A layer whose no-data would read as a real value has to mask it: unburned
    // ground is zero, and painting zero says the whole world burned. A layer
    // whose source is already masked must not, or it throws the data away.
    // Every layer declares which, so a new one cannot quietly inherit either.
    if (layer.sourceMasked) {
      assert.ok(!calls.includes('updateMask'),
        `${key} declares its source is masked but masks again anyway`)
    } else {
      assert.ok(calls.includes('updateMask'), `${key} never masks its no-data`)
    }
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

// ── Publication lag ──────────────────────────────────────────────────────────

test('no layer defaults to a period its dataset has not published', () => {
  // The bug this guards: reducing an empty collection gives an image with no
  // bands, and Earth Engine refuses a palette on that — so "burn scars this
  // year" failed outright rather than drawing an empty map. Defaults now sit
  // behind each dataset's own lag.
  assert.equal(resolveLayer('burn-severity').params.year, YEAR - MTBS_LAG_YEARS)
  assert.equal(resolveLayer('burn-date').params.year, YEAR - MODIS_LAG_YEARS)
  assert.equal(resolveLayer('years-since-fire').params.through, YEAR - MODIS_LAG_YEARS)
})

test('the current year is still reachable, just not the default', () => {
  // Somebody who knows this year is partly published should be able to ask.
  assert.equal(resolveLayer('burn-date', { year: YEAR }).params.year, YEAR)
})

test('every dated layer can say whether its window holds anything', () => {
  // Without a count, an empty window surfaces as a message about bands, which
  // tells the reader nothing about what to do.
  for (const key of ['burn-severity', 'burn-date', 'active-fire']) {
    assert.equal(typeof EE_TILE_LAYERS[key].count, 'function', `${key} cannot be pre-checked`)
  }
})

test('a single-band image is what reaches the palette', () => {
  // MTBS mosaics carry several bands, and a palette on a multi-band image is
  // refused outright — this failed even in years the data exists for.
  const selected = []
  const chain = new Proxy(function stub() {}, {
    get: (t, prop) => (prop === 'then' ? undefined : chain),
    apply: () => chain,
  })
  const ee = {
    ImageCollection: () => ({
      filterDate: () => ({
        select: (b) => { selected.push(b); return { max: () => chain } },
        max: () => chain,
      }),
    }),
    Image: Object.assign(() => chain, { constant: () => chain }),
  }
  const { layer, params } = resolveLayer('burn-severity')
  layer.build(ee, params)
  assert.deepEqual(selected, ['Severity'])
})

// ── Soil and forest type ─────────────────────────────────────────────────────

test('the GAP remap arrays stay paired', () => {
  // remap() takes two positional lists. If they fall out of step nothing
  // throws: the map simply paints the wrong forest type, in a plausible
  // colour, over ground somebody is about to walk. This is the single
  // most silent failure in the catalogue.
  const { from, to } = GAP_REMAP
  assert.equal(from.length, to.length,
    `${from.length} source codes against ${to.length} classes`)
  assert.ok(from.length > 0)
})

test('every remapped class has a colour and a label, and none is skipped', () => {
  const { to, classes } = GAP_REMAP
  const used = [...new Set(to)].sort((a, b) => a - b)
  // Contiguous from 1: a gap would leave a palette entry painting nothing
  // while every class after it drew in its neighbour's colour.
  assert.deepEqual(used, classes.map((_, i) => i + 1))
  for (const c of classes) {
    assert.match(c.color, /^#[0-9a-f]{6}$/i, `${c.label} has no usable colour`)
    assert.ok(c.label && c.label.length > 2, 'a class needs a readable label')
  }
  assert.equal(new Set(classes.map((c) => c.color)).size, classes.length,
    'two classes share a colour, so they cannot be told apart on the map')
})

test('no GAP source code is listed twice', () => {
  // A duplicate means one of the two mappings silently wins.
  const { from } = GAP_REMAP
  assert.equal(new Set(from).size, from.length)
})

test('the forest layer paints exactly the classes it defines', () => {
  const { layer } = resolveLayer('forest-type')
  assert.equal(layer.legend.type, 'classes')
  assert.equal(layer.legend.items.length, GAP_REMAP.classes.length)
})

test('the soil layers are grouped apart from the fire ones', () => {
  const soil = EE_LAYER_CATALOGUE.filter((l) => l.group === 'Soil')
  assert.ok(soil.length >= 4, 'expected the four soil views')
  for (const l of soil) {
    assert.ok(l.note, `${l.key} has no caveat`)
    assert.ok(l.attribution, `${l.key} is unattributed`)
  }
})

test('every soil layer says it is US-only, because blank ground is ambiguous', () => {
  // SOLUS100 covers the conterminous US. Outside it the layer draws nothing,
  // which is indistinguishable from "no soil here" unless the note says so.
  for (const key of ['soil-depth', 'soil-sand', 'soil-composition']) {
    const note = EE_TILE_LAYERS[key].note
    assert.match(note, /conterminous US|US only/i, `${key} does not state its extent`)
  }
})

test('the layers with no time control still pre-flight their source', () => {
  // These cannot be empty because of publication lag, so a zero count means
  // the property name is wrong — worth catching before Earth Engine returns
  // something opaque about a null image.
  for (const key of ['soil-depth', 'soil-sand', 'soil-composition']) {
    assert.equal(typeof EE_TILE_LAYERS[key].count, 'function', `${key} has no pre-flight`)
    assert.deepEqual(EE_TILE_LAYERS[key].params ?? {}, {}, `${key} should take no parameters`)
  }
})

test('the composite is the only rgb layer, and it declares no palette', () => {
  const rgb = EE_LAYER_KEYS.filter((k) => EE_TILE_LAYERS[k].rgb)
  assert.deepEqual(rgb, ['soil-composition'])
  assert.equal(EE_TILE_LAYERS['soil-composition'].legend.type, 'classes',
    'an rgb layer still needs a key saying what each channel is')
  assert.equal(EE_TILE_LAYERS['soil-composition'].legend.items.length, 3)
})

test('the new layers are gated at the tier the society sells', () => {
  for (const key of ['forest-type', 'soil-texture', 'soil-depth', 'soil-sand', 'soil-composition']) {
    assert.equal(tierFor(key), DEFAULT_TIER, `${key} is not gated as expected`)
  }
})
