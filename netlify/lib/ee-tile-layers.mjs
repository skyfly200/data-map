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

/** Earth Engine asset ids, in one place so a correction is a one-line change. */
export const ASSETS = {
  MODIS_BURN: 'MODIS/061/MCD64A1',
  MTBS_SEVERITY: 'USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1',
  FIRMS: 'FIRMS',
  S2_SR: 'COPERNICUS/S2_SR_HARMONIZED',
}

const THIS_YEAR = () => new Date().getUTCFullYear()

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
    attribution: 'NASA MODIS MCD64A1 via Google Earth Engine',
    opacity: 0.75,
    note: 'Years since the last detected burn, from MODIS burned area at 500 m. '
      + 'Morels flush in the first spring after a stand-replacing fire and fade over the next few years, '
      + 'so the red end is where to look. Unpainted means no burn detected since '
      + `${MODIS_FIRST_YEAR}, not that none happened: at 500 m a small fire can be missed entirely.`,
    params: {
      through: {
        type: 'int', label: 'Through year', default: THIS_YEAR,
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
    attribution: 'USFS / MTBS via Google Earth Engine',
    opacity: 0.7,
    note: 'Monitoring Trends in Burn Severity, 30 m, US only, one year at a time. '
      + 'Severity is what decides the morel response: high-severity stand-replacing patches flush, '
      + 'lightly burned ground largely does not.',
    params: {
      year: {
        type: 'int', label: 'Fire year', default: () => THIS_YEAR() - 1,
        min: MTBS_FIRST_YEAR, max: () => THIS_YEAR(),
      },
    },
    legend: { type: 'classes', items: MTBS_CLASSES.slice(1, 5) },
    build(ee, { year }) {
      const image = ee.ImageCollection(ASSETS.MTBS_SEVERITY)
        .filterDate(`${year}-01-01`, `${year}-12-31`)
        .max()
      // 0 is background and 6 a processing mask; painting either would cover
      // the continent in a colour that means nothing.
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
    note: 'What burned during the chosen year, coloured by when in the year it burned. '
      + 'A late-summer burn and an early-spring one are different prospects for the following spring.',
    params: {
      year: {
        type: 'int', label: 'Year', default: () => THIS_YEAR(),
        min: MODIS_FIRST_YEAR, max: () => THIS_YEAR(),
      },
    },
    legend: {
      type: 'ramp', unit: 'day of year', min: 'Jan', max: 'Dec',
      stops: ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c'],
    },
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
}

export const EE_LAYER_KEYS = Object.keys(EE_TILE_LAYERS)

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
