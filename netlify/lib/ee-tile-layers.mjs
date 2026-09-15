// Map layers rendered by Earth Engine.
//
// The reference layers already on the map come from somebody else's tile
// server, which limits them to what someone else has published. These are
// computed on demand instead: an Earth Engine expression is turned into a tile
// pyramid that Leaflet consumes like any other, so the map can show a layer
// nobody hosts — "how many years since this ground burned" being the first one,
// and the reason this exists.
//
// Fire first because of what it does to fruiting. A burn scar is the single
// strongest predictor a forager has: morels flush in the first spring after a
// stand-replacing fire, in numbers that do not occur on unburned ground, and
// the flush fades over the following two or three years. A layer that says how
// long ago each patch burned is a map of where to go next April.
//
// THE RULE, same as the job pipeline: nobody sends Earth Engine code. A request
// names a layer in this catalogue and supplies values that are checked against
// a schema. Accepting an expression would be arbitrary compute on the society's
// billing and a code-injection surface in one move.
//
// A caveat on the asset ids below. They were written without a live Earth
// Engine session to check them against, so a name or a band may be wrong. That
// is why netlify/functions/ee-tiles.mjs reports a failure loudly and by name
// rather than letting a layer come back blank: a silently empty layer looks
// exactly like ground that never burned, which is the most misleading thing
// this particular map could say.

export class LayerError extends Error {}

/**
 * The tier a layer is gated on.
 *
 * Per layer rather than one switch over all of them, because they do not cost
 * the same. A published asset rendered at a fixed palette is cheap and its
 * result is cached and shared by everyone; a composite calculated from raw
 * Sentinel-2 scenes as you look at it is not.
 *
 * The two that answer "where should I look for morels next spring" are open to
 * everyone. That is the question the society exists to help people answer, and
 * gating it would be gating the reason someone would visit at all. What
 * membership buys is the expensive end: running the pipeline, and the computed
 * layers that spend real time per tile.
 */
export const DEFAULT_TIER = 'member'

/**
 * Two declarations every layer has to make, because getting either wrong
 * produces a map that is confidently misleading rather than one that errors.
 *
 * `sourceMasked` — whether the source already masks its own no-data. A fire
 * layer must mask: unburned ground is zero, and painting zero says the whole
 * world burned. A soil surface must NOT: SOLUS100 is masked outside the
 * conterminous US by the publisher and every pixel it leaves is a real
 * measurement, so masking would throw away the layer. Declaring which is true
 * means a new layer has to decide rather than inherit whichever the test
 * happened to assert.
 *
 * `rgb` — whether the layer is three bands rendered as colour rather than one
 * band through a palette. Earth Engine refuses a palette on a multi-band image,
 * so these two are mutually exclusive.
 */

/** Earth Engine asset ids, in one place so a correction is a one-line change. */
export const ASSETS = {
  MODIS_BURN: 'MODIS/061/MCD64A1',
  MTBS_SEVERITY: 'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1',
  FIRMS: 'FIRMS',
  S2_SR: 'COPERNICUS/S2_SR_HARMONIZED',
  // Soil. SOLUS100 is a collection where each IMAGE is one soil property,
  // picked out by system:index rather than by band — which is why these cannot
  // be registered through the custom-layer form, whose whole model is one asset
  // id and one band.
  SOLUS100: 'USDA/SOLUS100/V0',
  OPENLANDMAP_TEXTURE: 'OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02',
  // What is growing on the ground, by type rather than by greenness.
  GAP_LANDCOVER: 'USGS/GAP/CONUS/2011',
}

const THIS_YEAR = () => new Date().getUTCFullYear()

/**
 * Publication lag, in years, for the datasets that have one.
 *
 * Every one of these is published well behind real time, and defaulting to the
 * current year asks for a window that does not exist yet. An empty window is
 * not an empty map: reducing an empty collection gives an image with no bands,
 * and visualising that with a palette fails outright — which is what "could not
 * be loaded" was.
 *
 * MTBS is compiled annually and runs one to two years behind. MODIS burned area
 * is published a couple of months in arrears, so the current year is only
 * partly there and the year before is the last complete one.
 */
export const MTBS_LAG_YEARS = 2
export const MODIS_LAG_YEARS = 1

// MODIS burned area starts in November 2000; asking for earlier returns an
// empty collection, which would render as "never burned" rather than "no data".
export const MODIS_FIRST_YEAR = 2001
export const MTBS_FIRST_YEAR = 1984

/** Years-since-fire ramp: the first year after a burn is the one that matters. */
const SINCE_FIRE_PALETTE = ['#d7191c', '#f07c4a', '#fdae61', '#fee090', '#c7e9b4', '#7fcdbb', '#41b6c4']

const MTBS_CLASSES = [
  { color: '#000000', label: 'Background' },
  { color: '#006400', label: 'Unburned to low' },
  { color: '#7fffd4', label: 'Low severity' },
  { color: '#ffff00', label: 'Moderate severity' },
  { color: '#ff0000', label: 'High severity' },
  { color: '#7fff00', label: 'Increased greenness' },
  { color: '#ffffff', label: 'Non-processing mask' },
]

// ─────────────────────────────────────────────────────────────────────────────
// Soil and land cover
// ─────────────────────────────────────────────────────────────────────────────

/** The twelve USDA texture classes, in the order OpenLandMap numbers them. */
const TEXTURE_CLASSES = [
  { color: '#d5c36b', label: 'Clay' },
  { color: '#b96947', label: 'Silty clay' },
  { color: '#9d3706', label: 'Sandy clay' },
  { color: '#ae868f', label: 'Clay loam' },
  { color: '#f86714', label: 'Silty clay loam' },
  { color: '#46d143', label: 'Sandy clay loam' },
  { color: '#368f20', label: 'Loam' },
  { color: '#3e5a14', label: 'Silt loam' },
  { color: '#ffd557', label: 'Sandy loam' },
  { color: '#fff72e', label: 'Silt' },
  { color: '#ff5a9d', label: 'Loamy sand' },
  { color: '#ff005b', label: 'Sand' },
]

const DEPTH_PALETTE = ['#feebe2', '#fcc5c0', '#fa9fb5', '#f768a1', '#dd3497', '#ae017e', '#7a0177']
const SAND_PALETTE = ['#081d58', '#253494', '#225ea8', '#1d91c0', '#41b6c4', '#7fcdbb', '#c7e9b4', '#ffffcc']

/**
 * One SOLUS100 soil property.
 *
 * Each property is a separate image in the collection, identified by
 * system:index; the band then carries the depth. `.first()` on a filter that
 * matches nothing is null and `.select` on null throws, which is what the
 * `count` pre-flight on each of these layers exists to catch — an index typo
 * would otherwise surface as an opaque Earth Engine error.
 */
function solusProperty(ee, index, band) {
  return ee.ImageCollection(ASSETS.SOLUS100)
    .filter(ee.Filter.eq('system:index', index))
    .first()
    .select(band)
}

const solusCount = (ee, index) => ee.ImageCollection(ASSETS.SOLUS100)
  .filter(ee.Filter.eq('system:index', index)).size()

/**
 * GAP land cover codes collapsed into the types a forager actually separates.
 *
 * The source has hundreds of ecological system codes, several of which are the
 * same forest under different names — "Rocky Mountain Subalpine Dry-Mesic
 * Spruce-Fir Forest" and its mesic twin are one thing when you are deciding
 * where to walk. These are the Front Range types, grouped.
 *
 * The two arrays are positional: `GAP_FROM[i]` becomes `GAP_TO[i]`. A test
 * asserts they stay the same length and that the classes stay contiguous,
 * because a mismatch does not throw — it silently paints the wrong forest.
 */
const GAP_FROM = [
  149,                          // Lodgepole pine
  151, 155,                     // Subalpine spruce-fir
  152, 156,                     // Montane spruce
  148,                          // Mixed conifer
  153, 158,                     // Ponderosa pine woodland
  171, 172,                     // Aspen and deciduous
  270, 272,                     // Shrubland and dwarf-shrub
  315, 316, 438, 439, 491,      // Grassland and meadow
  529,                          // Wetland and riparian
  502, 503, 549, 574, 575, 581, // Alpine tundra and barren
]

const GAP_TO = [
  1,
  2, 2,
  3, 3,
  4,
  5, 5,
  6, 6,
  7, 7,
  8, 8, 8, 8, 8,
  9,
  10, 10, 10, 10, 10, 10,
]

/**
 * What each class is, and why it is worth separating.
 *
 * Ordered to match GAP_TO: the nth entry is class n+1, which is what lets the
 * palette and the key be generated from one list rather than kept in step by
 * hand.
 */
const GAP_CLASSES = [
  { color: '#238b45', label: 'Lodgepole pine' },
  { color: '#08519c', label: 'Subalpine spruce-fir' },
  { color: '#41ab5d', label: 'Montane spruce' },
  { color: '#3182bd', label: 'Mixed conifer' },
  { color: '#006d2c', label: 'Ponderosa pine woodland' },
  { color: '#fec44f', label: 'Aspen and deciduous' },
  { color: '#e6550d', label: 'Shrubland' },
  { color: '#fdae6b', label: 'Grassland and meadow' },
  { color: '#31a354', label: 'Wetland and riparian' },
  { color: '#756bb1', label: 'Alpine tundra and barren' },
]

export const GAP_REMAP = { from: GAP_FROM, to: GAP_TO, classes: GAP_CLASSES }

/**
 * Validate a parameter against its schema.
 *
 * Small and strict on purpose: these values are interpolated into Earth Engine
 * calls, so "roughly a year" is not good enough.
 */
function readParams(schema, input = {}) {
  const out = {}
  for (const [key, spec] of Object.entries(schema)) {
    const raw = input[key]
    if (raw === undefined || raw === null || raw === '') {
      out[key] = typeof spec.default === 'function' ? spec.default() : spec.default
      continue
    }
    if (spec.type === 'int') {
      const n = Math.floor(Number(raw))
      if (!Number.isFinite(n)) throw new LayerError(`${spec.label} must be a whole number.`)
      const min = typeof spec.min === 'function' ? spec.min() : spec.min
      const max = typeof spec.max === 'function' ? spec.max() : spec.max
      if (n < min || n > max) throw new LayerError(`${spec.label} must be between ${min} and ${max}.`)
      out[key] = n
    } else if (spec.type === 'enum') {
      if (!spec.values.includes(String(raw))) {
        throw new LayerError(`${spec.label} must be one of: ${spec.values.join(', ')}.`)
      }
      out[key] = String(raw)
    } else {
      throw new LayerError(`Unsupported parameter type for ${key}.`)
    }
  }
  return out
}

/**
 * The most recent year each pixel burned, over a window of years.
 *
 * One image per year reduced with max(): a pixel that burned in 2019 and again
 * in 2024 reports 2024, which is what "years since fire" has to mean. Building
 * it as a band per year and reducing server-side keeps it one request rather
 * than one per year.
 */
function lastBurnYear(ee, { from, to }) {
  const years = []
  for (let y = from; y <= to; y += 1) years.push(y)
  const stack = years.map((y) => ee.ImageCollection(ASSETS.MODIS_BURN)
    .filterDate(`${y}-01-01`, `${y}-12-31`)
    .select('BurnDate')
    .max()
    // BurnDate is a day-of-year where it burned and masked where it did not, so
    // any unmasked value is a burn. unmask(0) turns "no fire" into a real zero
    // the max() below can compare rather than a hole that swallows the pixel.
    .gt(0)
    .unmask(0)
    .multiply(y)
    .rename('year'))
  return ee.ImageCollection(stack).max()
}

/**
 * The catalogue.
 *
 * `build(ee, params)` returns the image and how to paint it. `legend` is the
 * key drawn on the map, in the same shape the existing tile layers use, so the
 * map's legend code does not have to know an Earth Engine layer from any other.
 */
export const EE_TILE_LAYERS = {
  'years-since-fire': {
    name: 'Years since fire',
    group: 'Fire',
    // Open to everyone: this is the layer that answers where to look next
    // spring, and it is one cached render shared by every viewer.
    tier: 'free',
    attribution: 'NASA MODIS MCD64A1 via Google Earth Engine',
    opacity: 0.75,
    note: 'Years since the last detected burn, from MODIS burned area at 500 m. '
      + 'Morels flush in the first spring after a stand-replacing fire and fade over the next few years, '
      + 'so the red end is where to look. Unpainted means no burn detected since '
      + `${MODIS_FIRST_YEAR}, not that none happened: at 500 m a small fire can be missed entirely.`,
    params: {
      through: {
        type: 'int', label: 'Through year', default: () => THIS_YEAR() - MODIS_LAG_YEARS,
        min: MODIS_FIRST_YEAR, max: () => THIS_YEAR(),
      },
      window: { type: 'int', label: 'Years to look back', default: 12, min: 2, max: 25 },
    },
    legend: {
      type: 'ramp', unit: 'years', min: '0', max: '12',
      stops: SINCE_FIRE_PALETTE,
    },
    build(ee, { through, window }) {
      const from = Math.max(MODIS_FIRST_YEAR, through - window + 1)
      const year = lastBurnYear(ee, { from, to: through })
      // Pixels that never burned are zero and must not read as "burned in year
      // zero", which would paint the whole unburned world as freshly burned.
      const burned = year.gt(0)
      const since = ee.Image.constant(through).subtract(year).updateMask(burned)
      return {
        image: since,
        vis: { min: 0, max: Math.max(1, window - 1), palette: SINCE_FIRE_PALETTE },
      }
    },
  },

  'burn-severity': {
    name: 'Burn severity (US)',
    group: 'Fire',
    // Also open. Severity is what decides whether a burn scar is worth walking,
    // so it is half of the same question, and MTBS is a published product that
    // costs one cached render per year rather than per viewer.
    tier: 'free',
    attribution: 'USFS / MTBS via Google Earth Engine',
    opacity: 0.7,
    note: 'Monitoring Trends in Burn Severity, 30 m, US only, one year at a time. '
      + 'Severity is what decides the morel response: high-severity stand-replacing patches flush, '
      + 'lightly burned ground largely does not.',
    params: {
      year: {
        type: 'int', label: 'Fire year', default: () => THIS_YEAR() - MTBS_LAG_YEARS,
        min: MTBS_FIRST_YEAR, max: () => THIS_YEAR(),
      },
    },
    legend: { type: 'classes', items: MTBS_CLASSES.slice(1, 5) },
    count: (ee, { year }) => ee.ImageCollection(ASSETS.MTBS_SEVERITY)
      .filterDate(`${year}-01-01`, `${year}-12-31`).size(),
    build(ee, { year }) {
      // select() is not optional here. Without it the mosaic keeps every band
      // it has, and Earth Engine refuses a palette on a multi-band image — so
      // this failed even in years where the data is published.
      const image = ee.ImageCollection(ASSETS.MTBS_SEVERITY)
        .filterDate(`${year}-01-01`, `${year}-12-31`)
        .select('Severity')
        .max()
      // 0 is background and 6 a processing mask; painting either would cover
      // the continent in a color that means nothing.
      const burned = image.gte(1).and(image.lte(5))
      return {
        image: image.updateMask(burned),
        vis: { min: 1, max: 6, palette: MTBS_CLASSES.slice(1).map((c) => c.color) },
      }
    },
  },

  'burn-date': {
    name: 'Burn scars this year',
    group: 'Fire',
    attribution: 'NASA MODIS MCD64A1 via Google Earth Engine',
    opacity: 0.75,
    note: 'What burned during the chosen year, colored by when in the year it burned. '
      + 'A late-summer burn and an early-spring one are different prospects for the following spring.',
    params: {
      year: {
        type: 'int', label: 'Year', default: () => THIS_YEAR() - MODIS_LAG_YEARS,
        min: MODIS_FIRST_YEAR, max: () => THIS_YEAR(),
      },
    },
    legend: {
      type: 'ramp', unit: 'day of year', min: 'Jan', max: 'Dec',
      stops: ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
    },
    count: (ee, { year }) => ee.ImageCollection(ASSETS.MODIS_BURN)
      .filterDate(`${year}-01-01`, `${year}-12-31`).size(),
    build(ee, { year }) {
      const image = ee.ImageCollection(ASSETS.MODIS_BURN)
        .filterDate(`${year}-01-01`, `${year}-12-31`)
        .select('BurnDate')
        .max()
      return {
        image: image.updateMask(image.gt(0)),
        vis: { min: 1, max: 366, palette: ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'] },
      }
    },
  },

  'active-fire': {
    name: 'Active fires (recent)',
    group: 'Fire',
    attribution: 'NASA FIRMS via Google Earth Engine',
    opacity: 0.85,
    note: 'Thermal anomalies detected in the last few days. Near-real-time and coarse: '
      + 'a detection is a hot pixel, which is usually a fire and is sometimes a flare or hot bare rock.',
    params: {
      days: { type: 'int', label: 'Days back', default: 7, min: 1, max: 60 },
    },
    legend: {
      type: 'ramp', unit: 'K (brightness)', min: '300', max: '400',
      stops: ['#ffeda0', '#feb24c', '#f03b20', '#bd0026'],
    },
    count: (ee, { days }) => {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      return ee.ImageCollection(ASSETS.FIRMS)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)).size()
    },
    build(ee, { days }) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      const image = ee.ImageCollection(ASSETS.FIRMS)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .select('T21')
        .max()
      return {
        image: image.updateMask(image.gt(0)),
        vis: { min: 300, max: 400, palette: ['#ffeda0', '#feb24c', '#f03b20', '#bd0026'] },
      }
    },
  },

  'dnbr': {
    name: 'Burn severity, computed (dNBR)',
    group: 'Fire',
    attribution: 'Copernicus Sentinel-2 via Google Earth Engine',
    opacity: 0.7,
    // The layer that justifies computing tiles rather than fetching them: this
    // exists nowhere as a published product, and outside the US MTBS does not
    // reach, so for most of the world this is the only severity map there is.
    note: 'Difference in Normalised Burn Ratio between the summer before and the summer after the '
      + 'chosen fire year, computed from Sentinel-2. Global, 20 m, and slower to draw than the '
      + 'other layers because it is calculated as you look at it. Cloud and snow can leave artefacts; '
      + 'read a patch, not a pixel.',
    slow: true,
    params: {
      year: {
        type: 'int', label: 'Fire year', default: () => THIS_YEAR() - 1,
        // Sentinel-2 surface reflectance begins in 2017.
        min: 2017, max: () => THIS_YEAR(),
      },
    },
    legend: {
      type: 'ramp', unit: 'dNBR', min: 'low', max: 'high',
      stops: ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'],
    },
    build(ee, { year }) {
      // NBR is (NIR - SWIR2) / (NIR + SWIR2); it collapses after a burn, so the
      // pre-minus-post difference is the severity.
      const nbr = (from, to) => ee.ImageCollection(ASSETS.S2_SR)
        .filterDate(from, to)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
        .median()
        .normalizedDifference(['B8', 'B12'])

      // Peak-growing-season windows either side of the fire year, so the two
      // composites are comparable rather than one being winter.
      const pre = nbr(`${year - 1}-06-01`, `${year - 1}-09-30`)
      const post = nbr(`${year + 1}-06-01`, `${year + 1}-09-30`)
      const dnbr = pre.subtract(post)
      // Below ~0.1 is unburned or noise; painting it would turn every seasonal
      // difference in the composite into an apparent fire.
      return {
        image: dnbr.updateMask(dnbr.gt(0.1)),
        vis: { min: 0.1, max: 1.3, palette: ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'] },
      }
    },
  },

  // ── What is growing ────────────────────────────────────────────────────────

  'forest-type': {
    name: 'Forest and land cover type',
    group: 'Vegetation',
    tier: DEFAULT_TIER,
    attribution: 'USGS GAP/LANDFIRE National Terrestrial Ecosystems 2011 via Google Earth Engine',
    opacity: 0.85,
    // The single most useful layer on this map for most fungi, and the reason
    // it is worth carrying a second land cover product alongside ESA
    // WorldCover: WorldCover says "tree cover", this says WHICH trees. Most
    // ectomycorrhizal species are host-specific, so the difference between
    // lodgepole and ponderosa is the difference between two different lists of
    // what you might find.
    note: 'Ecological system classes from USGS GAP, 30 m, grouped into the forest types a forager '
      + 'separates. Mapped once, from 2011 imagery, so a fire, a clear-cut or a beetle kill since '
      + 'then is not in it — check the fire layers alongside. US only, and the boundaries are '
      + 'modelled, so read a stand rather than a pixel.',
    legend: { type: 'classes', items: GAP_CLASSES },
    build(ee) {
      const remapped = ee.Image(ASSETS.GAP_LANDCOVER).select('landcover').remap(GAP_FROM, GAP_TO)
      // remap leaves everything unlisted masked, but an explicit range guard
      // keeps a future code addition from painting past the end of the palette.
      const known = remapped.gte(1).and(remapped.lte(GAP_CLASSES.length))
      return {
        image: remapped.updateMask(known),
        vis: { min: 1, max: GAP_CLASSES.length, palette: GAP_CLASSES.map((c) => c.color) },
      }
    },
  },

  // ── The ground itself ──────────────────────────────────────────────────────
  //
  // Four views of the same soil, because they answer different questions. Depth
  // says whether there is anything to grow in; texture and sand say how it
  // holds water; the composite says all three proportions at once for reading
  // gradients rather than values.

  'soil-texture': {
    name: 'Soil texture class',
    group: 'Soil',
    tier: DEFAULT_TIER,
    attribution: 'OpenLandMap USDA texture class via Google Earth Engine',
    opacity: 0.85,
    note: 'The USDA texture triangle class of the topsoil, predicted globally at 250 m. '
      + 'Texture is what decides how long the ground stays wet after rain, which is the half of '
      + 'fruiting weather the rain layers cannot tell you. A model prediction, not a soil survey: '
      + 'right about a hillside, unreliable about a square metre.',
    legend: { type: 'classes', items: TEXTURE_CLASSES },
    // Nothing to mask: the source already masks open water and everything it
    // leaves is a real class. See the note on sourceMasked above.
    sourceMasked: true,
    build(ee) {
      const image = ee.Image(ASSETS.OPENLANDMAP_TEXTURE).select('b0')
      return {
        image,
        vis: { min: 1, max: TEXTURE_CLASSES.length, palette: TEXTURE_CLASSES.map((c) => c.color) },
      }
    },
  },

  'soil-depth': {
    name: 'Soil depth to bedrock',
    group: 'Soil',
    tier: DEFAULT_TIER,
    attribution: 'USDA SOLUS100 via Google Earth Engine',
    opacity: 0.85,
    note: 'Predicted depth to any lithic contact — bedrock — at 100 m, conterminous US only. '
      + 'Shallow soil over rock dries fast and holds a different community from deep colluvium at '
      + 'the bottom of the same slope. Unpainted is outside the mapped area, not zero depth.',
    legend: {
      type: 'ramp', unit: 'cm', min: '0', max: '150+',
      stops: DEPTH_PALETTE,
    },
    sourceMasked: true,
    count: (ee) => solusCount(ee, 'anylithicdpt'),
    build(ee) {
      return {
        image: solusProperty(ee, 'anylithicdpt', 'r_cm_p'),
        vis: { min: 0, max: 150, palette: DEPTH_PALETTE },
      }
    },
  },

  'soil-sand': {
    name: 'Sand content (drainage proxy)',
    group: 'Soil',
    tier: DEFAULT_TIER,
    attribution: 'USDA SOLUS100 via Google Earth Engine',
    opacity: 0.85,
    note: 'Percent sand in the surface horizon at 100 m, conterminous US only. Read as a proxy for '
      + 'how fast water leaves: the pale end drains freely and dries within a day or two of rain, '
      + 'the dark end holds it. It is a proxy and not a drainage measurement — a sandy flat with a '
      + 'high water table stays wet whatever its texture says.',
    legend: {
      type: 'ramp', unit: '% sand', min: '0', max: '80+',
      stops: SAND_PALETTE,
    },
    sourceMasked: true,
    count: (ee) => solusCount(ee, 'sandtotal'),
    build(ee) {
      return {
        image: solusProperty(ee, 'sandtotal', 'r_0_cm_p'),
        vis: { min: 0, max: 80, palette: SAND_PALETTE },
      }
    },
  },

  'soil-composition': {
    name: 'Soil composition (sand, silt, clay)',
    group: 'Soil',
    tier: DEFAULT_TIER,
    attribution: 'USDA SOLUS100 via Google Earth Engine',
    opacity: 0.85,
    // Three properties at once, as colour rather than as a scale. Useless for
    // reading a value off and very good for seeing where the ground changes,
    // which is the thing a single-property ramp hides.
    note: 'Sand, silt and clay as red, green and blue at 100 m, conterminous US only. There is no '
      + 'scale to read a number off — the point is the boundaries. Where the colour changes, the '
      + 'soil changes, and those edges often run with the ground rather than with anything visible '
      + 'on the surface. Mixtures read as you would expect: yellow is sand and silt, cyan silt and '
      + 'clay, magenta sand and clay, grey an even mix.',
    legend: {
      type: 'classes',
      items: [
        { color: '#ff0000', label: 'Sand (red)' },
        { color: '#00ff00', label: 'Silt (green)' },
        { color: '#0000ff', label: 'Clay (blue)' },
      ],
    },
    sourceMasked: true,
    // Three bands rendered as colour, so there is no palette to declare and
    // nothing for a one-dimensional key to say.
    rgb: true,
    count: (ee) => solusCount(ee, 'claytotal'),
    build(ee) {
      const band = 'r_0_cm_p'
      const image = ee.Image.cat([
        solusProperty(ee, 'sandtotal', band),
        solusProperty(ee, 'silttotal', band),
        solusProperty(ee, 'claytotal', band),
      ])
      // Three bands and deliberately NO palette: Earth Engine renders a
      // three-band image as RGB, and passing a palette alongside is what it
      // refuses outright.
      return { image, vis: { min: 0, max: 70 } }
    },
  },
}

export const EE_LAYER_KEYS = Object.keys(EE_TILE_LAYERS)

/** The tier a layer requires. Unknown layers are treated as the strictest. */
export function tierFor(key) {
  return EE_TILE_LAYERS[key]?.tier || DEFAULT_TIER
}

/** A layer and its validated parameters, or a LayerError saying what is wrong. */
export function resolveLayer(key, input = {}) {
  const layer = EE_TILE_LAYERS[key]
  if (!layer) throw new LayerError(`Unknown layer “${key}”.`)
  const params = readParams(layer.params || {}, input)
  return { key, layer, params }
}

/**
 * A stable cache key for one layer at one set of parameters.
 *
 * Minting a tile URL costs an Earth Engine call and the result is identical for
 * everyone, so two members looking at last year's burn severity should share
 * one. Sorted so parameter order cannot produce two keys for one layer.
 */
export function cacheKey(key, params) {
  const parts = Object.keys(params).sort().map((k) => `${k}=${params[k]}`)
  return [key, ...parts].join('|')
}

/** What the client needs to draw the layer, minus the tile URL itself. */
export function describeLayer(key) {
  const layer = EE_TILE_LAYERS[key]
  if (!layer) return null
  return {
    key,
    name: layer.name,
    group: layer.group,
    attribution: layer.attribution,
    opacity: layer.opacity ?? 1,
    note: layer.note,
    legend: layer.legend,
    slow: !!layer.slow,
    tier: layer.tier || DEFAULT_TIER,
    params: Object.fromEntries(Object.entries(layer.params || {}).map(([k, spec]) => [k, {
      label: spec.label,
      type: spec.type,
      default: typeof spec.default === 'function' ? spec.default() : spec.default,
      min: typeof spec.min === 'function' ? spec.min() : spec.min,
      max: typeof spec.max === 'function' ? spec.max() : spec.max,
      values: spec.values,
    }])),
  }
}

export const EE_LAYER_CATALOGUE = EE_LAYER_KEYS.map(describeLayer)
