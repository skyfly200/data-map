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
// a schema. Accepting an expression would be arbitrary compute on the FRMS
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
 * everyone. That is the question FRMS exists to help people answer, and
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
  // Hansen Global Forest Change: year of stand-replacing forest loss, 30 m,
  // global. Loss is any removal of the canopy — harvest, clearing, blowdown or
  // fire — so it is the closest global proxy for cutting there is. The version
  // fixes the last year in the record, which is why HANSEN_LAST_YEAR tracks it.
  HANSEN: 'UMD/hansen/global_forest_change_2023_v1_11',
  // Soil. SOLUS100 is a collection where each IMAGE is one soil property,
  // picked out by system:index rather than by band — which is why these cannot
  // be registered through the custom-layer form, whose whole model is one asset
  // id and one band.
  SOLUS100: 'USDA/SOLUS100/V0',
  OPENLANDMAP_TEXTURE: 'OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02',
  // What is growing on the ground, by type rather than by greenness.
  GAP_LANDCOVER: 'USGS/GAP/CONUS/2011',
  // Terrain. SRTM is one masked image covering 60°N–56°S at 30 m, which is why
  // the slope/aspect/exposure layers can each be one ee.Image() rather than a
  // mosaic — and why they declare sourceMasked: the sea is already masked and
  // every land pixel it leaves is a real elevation.
  SRTM: 'USGS/SRTMGL1_003',
  // Upstream drainage area (band 'upa', km²), the contributing-area term the
  // topographic wetness index needs and that a slope raster cannot supply on
  // its own.
  MERIT_HYDRO: 'MERIT/Hydro/v1_0_1',
  // USFS TreeMap: modelled forest structure on the FIA plot grid, 30 m, US
  // forests only. A collection whose 2016 image is the baseline, so it is
  // filtered by date and .first() picked out like the burn products.
  TREEMAP: 'projects/gtac-data-publish/assets/TreeMap/Product_Version/2026-1',
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

// Hansen Global Forest Change records loss year as 1–N, where N is the last
// year in the version's record. v2023 covers 2001–2023.
export const HANSEN_FIRST_YEAR = 2001
export const HANSEN_LAST_YEAR = 2023

/** Old loss (blue) through recent loss (red): a fresh cut is what to walk. */
const FOREST_LOSS_PALETTE = ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']

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

// ─────────────────────────────────────────────────────────────────────────────
// Terrain and vegetation index palettes
// ─────────────────────────────────────────────────────────────────────────────

/** Flat green through to steep red. */
const SLOPE_PALETTE = ['#1a9850', '#91cf60', '#d9ef8b', '#fee08b', '#fc8d59', '#d73027']
/** Cyclic, because aspect wraps: north reads the same colour at 0° and 360°. */
const ASPECT_PALETTE = ['#e41a1c', '#ff7f00', '#ffff33', '#4daf4a', '#377eb8', '#984ea3', '#e41a1c']
/** Dry (brown) to wet (teal): high TWI is where water collects. */
const TWI_PALETTE = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']
/** Diverging: sheltered hollows blue, exposed ridges red. */
const EXPOSURE_PALETTE = ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b']
/** Cool shaded slopes to hot sun-facing ones. */
const SOLAR_PALETTE = ['#2166ac', '#67a9cf', '#d1e5f0', '#fee090', '#fc8d59', '#d73027']
/** The classic NDVI ramp: bare ground brown through dense canopy green. */
const NDVI_PALETTE = ['#a50026', '#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b',
  '#a6d96a', '#66bd63', '#1a9850', '#006837']
/** Dry canopy brown to moist canopy teal. */
const NDMI_PALETTE = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e']

/** Short (pale) to tall (deep magenta): a proxy for stand maturity. */
const STAND_HEIGHT_PALETTE = ['#feebe2', '#fcc5c0', '#fa9fb5', '#f768a1', '#ae017e']

/** Open (pale) to closed canopy (deep blue). */
const CANOPY_PALETTE = ['#ffffd9', '#c7e9b4', '#41b6c4', '#225ea8', '#081d58']

/** Moderate (yellow) through complete (dark red) canopy moisture loss. */
const MOISTURE_LOSS_PALETTE = ['#ffeda0', '#feb24c', '#f03b20', '#bd0026']

/**
 * FIA stand-size classes (STDSZCD), by the diameter distribution of the stand.
 * Positional, so class n is the nth entry — code 4 is included between small
 * and nonstocked even though the reference script skipped it, because the
 * source uses it and a gap would paint seedling stands in the nonstocked
 * colour.
 */
const STAND_SIZE_CLASSES = [
  { color: '#2b83ba', label: 'Large diameter' },
  { color: '#abdda4', label: 'Medium diameter' },
  { color: '#fdae61', label: 'Small diameter' },
  { color: '#d7191c', label: 'Seedling / sapling' },
  { color: '#bbbbbb', label: 'Nonstocked' },
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

// ─────────────────────────────────────────────────────────────────────────────
// Terrain analysis, all derived from one DEM
// ─────────────────────────────────────────────────────────────────────────────

const DEG = Math.PI / 180

/** Slope in radians, with a floor so a flat pixel does not divide TWI by zero. */
function slopeRadians(ee) {
  return ee.Terrain.slope(ee.Image(ASSETS.SRTM)).multiply(DEG)
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentinel-2 vegetation indices
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The month window each season spans. Northern-hemisphere seasons, because the
 * map's ground is: a "summer" composite over the Rockies wants June–August, and
 * inverting that for the south is a refinement nobody looking at this map needs.
 */
const SEASONS = {
  spring: { label: 'Spring', from: '03-01', to: '05-31' },
  summer: { label: 'Summer', from: '06-01', to: '08-31' },
  fall: { label: 'Autumn', from: '09-01', to: '11-30' },
  winter: { label: 'Winter', from: '12-01', to: '12-31' },
}

/** A cloud-filtered Sentinel-2 collection over a date range. */
const s2Between = (ee, from, to) => ee.ImageCollection(ASSETS.S2_SR)
  .filterDate(from, to)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))

/** The two bands a normalised index is built from, by index name. */
const INDEX_BANDS = { ndvi: ['B8', 'B4'], ndmi: ['B8', 'B11'] }

/** A recent rolling window: [startISO, endISO] for the last `days` days. */
function recentWindow(days) {
  const end = new Date()
  const start = new Date(end.getTime() - days * 86400000)
  return [start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)]
}

/**
 * The recent (rolling) variant of a Sentinel-2 index, open to everyone: one
 * cloud-filtered median over the last few weeks, so it always reflects now
 * rather than a year you have to pick.
 */
function recentIndexLayer({ name, index, group, note, palette }) {
  const [b1, b2] = INDEX_BANDS[index]
  return {
    name,
    group,
    tier: 'free',
    attribution: 'Copernicus Sentinel-2 via Google Earth Engine',
    opacity: 0.8,
    note,
    slow: true,
    params: {
      days: { type: 'int', label: 'Days back', default: 30, min: 5, max: 120 },
    },
    legend: { type: 'ramp', unit: index, min: '−1', max: '+1', stops: palette },
    count: (ee, { days }) => {
      const [from, to] = recentWindow(days)
      return s2Between(ee, from, to).size()
    },
    build(ee, { days }) {
      const [from, to] = recentWindow(days)
      const image = s2Between(ee, from, to).median().normalizedDifference([b1, b2]).rename(index)
      // A valid-range mask, which also drops the water and cloud-shadow pixels
      // that fall outside [−1, 1] as compositing artefacts.
      return {
        image: image.updateMask(image.gte(-1).and(image.lte(1))),
        vis: { min: -0.2, max: 0.9, palette },
      }
    },
  }
}

/**
 * The seasonal variant, for members: a median over one season of one year, so
 * two years can be compared at the same phenological moment. This is real
 * compute per tile, which is why it is gated where the recent one is not.
 */
function seasonalIndexLayer({ name, index, group, note, palette }) {
  const [b1, b2] = INDEX_BANDS[index]
  const seasonKeys = Object.keys(SEASONS)
  const range = (ee, { year, season }) => {
    const s = SEASONS[season]
    return s2Between(ee, `${year}-${s.from}`, `${year}-${s.to}`)
  }
  return {
    name,
    group,
    tier: DEFAULT_TIER,
    attribution: 'Copernicus Sentinel-2 via Google Earth Engine',
    opacity: 0.8,
    note,
    slow: true,
    params: {
      year: {
        type: 'int', label: 'Year', default: () => THIS_YEAR() - 1,
        // Sentinel-2 surface reflectance begins in 2017.
        min: 2017, max: () => THIS_YEAR(),
      },
      season: { type: 'enum', label: 'Season', default: 'summer', values: seasonKeys },
    },
    legend: { type: 'ramp', unit: index, min: '−1', max: '+1', stops: palette },
    count: (ee, params) => range(ee, params).size(),
    build(ee, params) {
      const image = range(ee, params).median().normalizedDifference([b1, b2]).rename(index)
      return {
        image: image.updateMask(image.gte(-1).and(image.lte(1))),
        vis: { min: -0.2, max: 0.9, palette },
      }
    },
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// USFS TreeMap forest structure
// ─────────────────────────────────────────────────────────────────────────────

/**
 * The 2016 TreeMap baseline image, one band selected.
 *
 * TreeMap is a collection whose 2016 image is the modelled baseline; filtering
 * by that year and taking .first() picks it out. .first() on a filter that
 * matched nothing is null and .select on null throws, which is what the `count`
 * pre-flight on each TreeMap layer catches before Earth Engine returns
 * something opaque.
 */
const treeMap2016 = (ee, band) => ee.ImageCollection(ASSETS.TREEMAP)
  .filterDate('2016-01-01', '2016-12-31')
  .first()
  .select(band)

const treeMapCount = (ee) => ee.ImageCollection(ASSETS.TREEMAP)
  .filterDate('2016-01-01', '2016-12-31').size()

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
    group: 'Fire & disturbance',
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
    group: 'Fire & disturbance',
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
    group: 'Fire & disturbance',
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
    group: 'Fire & disturbance',
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
    group: 'Fire & disturbance',
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

  'forest-loss': {
    name: 'Forest loss / cutting',
    group: 'Fire & disturbance',
    // Free, like the burn layers: a published asset, one cached render shared by
    // everyone, and it answers the same "where was the canopy opened recently"
    // question — a fresh cut or blowdown flushes some of the same species a burn
    // does.
    tier: 'free',
    attribution: 'Hansen / UMD / Google / USGS / NASA via Google Earth Engine',
    opacity: 0.8,
    note: 'Year of stand-replacing forest loss from Hansen Global Forest Change, 30 m, global. '
      + 'Loss is any canopy removal — harvest, clearing, blowdown or fire — so it does not tell cut '
      + 'from burn on its own; read it alongside the burn layers, and what is left is largely '
      + 'cutting. Recent losses are red. Unpainted is ground with no detected loss since '
      + `${HANSEN_FIRST_YEAR}.`,
    params: {
      window: { type: 'int', label: 'Years to look back', default: 12, min: 2, max: 23 },
    },
    legend: {
      type: 'ramp', unit: 'year of loss', min: 'older', max: 'recent', stops: FOREST_LOSS_PALETTE,
    },
    build(ee, { window }) {
      const lossyear = ee.Image(ASSETS.HANSEN).select('lossyear')
      // lossyear is 1–23 for 2001–2023 and 0 where nothing was lost. Keep only
      // losses inside the window, then paint by their actual calendar year.
      const firstIndex = Math.max(1, (HANSEN_LAST_YEAR - 2000) - window + 1)
      const recent = lossyear.gte(firstIndex)
      const year = lossyear.add(2000).updateMask(lossyear.gt(0)).updateMask(recent)
      return {
        image: year,
        vis: { min: HANSEN_FIRST_YEAR, max: HANSEN_LAST_YEAR, palette: FOREST_LOSS_PALETTE },
      }
    },
  },

  'canopy-moisture-loss': {
    name: 'Canopy moisture crash (beetle proxy)',
    group: 'Fire & disturbance',
    // Computed from raw Sentinel-2 across two summers as you look, so it is a
    // members' layer like dNBR rather than a shared cached render.
    tier: DEFAULT_TIER,
    attribution: 'Copernicus Sentinel-2 via Google Earth Engine',
    opacity: 0.85,
    slow: true,
    note: 'The drop in canopy moisture (NDMI) between a baseline summer and a recent one, computed '
      + 'from Sentinel-2 over the late-summer window. A large drop is a canopy that dried out or died '
      + '— often bark beetle, sometimes drought or disease — which opens the stand and changes what '
      + 'fruits under it. Only losses over 0.15 NDMI are painted; both summers must be mostly '
      + 'cloud-free to read.',
    params: {
      baseline: {
        type: 'int', label: 'Baseline year', default: 2017,
        min: 2017, max: () => THIS_YEAR(),
      },
      recent: {
        type: 'int', label: 'Recent year', default: () => THIS_YEAR() - 1,
        min: 2017, max: () => THIS_YEAR(),
      },
    },
    legend: {
      type: 'ramp', unit: 'NDMI drop', min: '0.15', max: '0.4+', stops: MOISTURE_LOSS_PALETTE,
    },
    count: (ee, { recent }) => s2Between(ee, `${recent}-07-15`, `${recent}-09-15`).size(),
    build(ee, { baseline, recent }) {
      // Late summer both years, to align the two composites at the same point in
      // the season rather than comparing spring to autumn.
      const ndmi = (year) => s2Between(ee, `${year}-07-15`, `${year}-09-15`)
        .median().normalizedDifference(['B8', 'B11'])
      const loss = ndmi(baseline).subtract(ndmi(recent))
      // Below ~0.15 is seasonal wobble and cloud noise; painting it would turn
      // every wet-year-to-dry-year difference into an apparent die-off.
      return {
        image: loss.updateMask(loss.gt(0.15)),
        vis: { min: 0.15, max: 0.4, palette: MOISTURE_LOSS_PALETTE },
      }
    },
  },

  // ── Terrain analysis ───────────────────────────────────────────────────────
  //
  // Five reads of the same DEM, each answering a question about where fungi
  // fruit that the flat map cannot. They are free: SRTM is a published asset and
  // every one of these is one cached render shared by every viewer, so gating
  // them would buy nothing. Each declares sourceMasked, because SRTM masks the
  // sea and every land pixel it leaves is a real elevation — masking again would
  // only throw away coastline.

  'slope': {
    name: 'Slope',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'NASA SRTM via Google Earth Engine',
    opacity: 0.7,
    note: 'Steepness in degrees from SRTM at 30 m. Slope sets how fast water runs off and how much '
      + 'sun a face catches, so it underlies both the wetness and the solar layers. Gentle benches '
      + 'and toe-slopes hold moisture that steep ground sheds.',
    legend: { type: 'ramp', unit: 'degrees', min: '0', max: '45+', stops: SLOPE_PALETTE },
    sourceMasked: true,
    build(ee) {
      const slope = ee.Terrain.slope(ee.Image(ASSETS.SRTM))
      return { image: slope, vis: { min: 0, max: 45, palette: SLOPE_PALETTE } }
    },
  },

  'aspect': {
    name: 'Aspect (slope direction)',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'NASA SRTM via Google Earth Engine',
    opacity: 0.7,
    note: 'The compass direction each slope faces, 0–360° from SRTM. North-facing ground stays '
      + 'cooler and damper and holds snow later; south-facing dries first. Which matters depends on '
      + 'the species and the season, which is why this is offered raw rather than pre-judged. '
      + 'The palette is cyclic, so north reads the same colour at both ends.',
    legend: { type: 'ramp', unit: '° from north', min: 'N', max: 'N', stops: ASPECT_PALETTE },
    sourceMasked: true,
    build(ee) {
      const aspect = ee.Terrain.aspect(ee.Image(ASSETS.SRTM))
      return { image: aspect, vis: { min: 0, max: 360, palette: ASPECT_PALETTE } }
    },
  },

  'twi': {
    name: 'Topographic wetness (TWI)',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'NASA SRTM and MERIT Hydro via Google Earth Engine',
    opacity: 0.75,
    note: 'Where the terrain gathers water: high (teal) is convergent, low-lying ground that stays '
      + 'wet between rains, low (brown) is fast-draining upland. Computed as ln(upslope area ÷ tan '
      + 'slope) from SRTM slope and MERIT Hydro drainage area. A shape of the ground, not a '
      + 'measurement of the soil — pair it with the soil texture layer.',
    legend: { type: 'ramp', unit: 'TWI', min: 'dry', max: 'wet', stops: TWI_PALETTE },
    sourceMasked: true,
    build(ee) {
      const tan = slopeRadians(ee).tan().max(0.001)
      // upa is upstream drainage area in km²; to m² so the log spans a sensible
      // range rather than sitting near zero.
      const area = ee.Image(ASSETS.MERIT_HYDRO).select('upa').multiply(1e6)
      const twi = area.divide(tan).log()
      return { image: twi, vis: { min: 2, max: 20, palette: TWI_PALETTE } }
    },
  },

  'wind-exposure': {
    name: 'Wind exposure',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'NASA SRTM via Google Earth Engine',
    opacity: 0.75,
    note: 'How exposed or sheltered each spot is, as a topographic position index: elevation minus '
      + 'the average of the ground within a kilometre. Ridges and summits (red) take the wind and '
      + 'dry out; sheltered hollows and lee slopes (blue) stay still and humid. A proxy from shape '
      + 'alone — it does not know the prevailing wind, only what stands above its surroundings.',
    legend: { type: 'ramp', unit: 'sheltered → exposed', min: 'lee', max: 'ridge', stops: EXPOSURE_PALETTE },
    sourceMasked: true,
    build(ee) {
      const dem = ee.Image(ASSETS.SRTM)
      // Topographic position index: height above the local mean. focalMean over
      // a 1 km circle sets the scale at which "exposed" is judged.
      const tpi = dem.subtract(dem.focalMean(1000, 'circle', 'meters'))
      return { image: tpi, vis: { min: -60, max: 60, palette: EXPOSURE_PALETTE } }
    },
  },

  'solar-exposure': {
    name: 'Solar exposure',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'NASA SRTM via Google Earth Engine',
    opacity: 0.75,
    note: 'A heat-load proxy: how much sun a slope catches, from its steepness and the way it faces. '
      + 'Hot (red) is steep and south-west-facing, the last ground to hold moisture; cool (blue) is '
      + 'shaded north-east-facing ground that stays damp. Derived from SRTM slope and aspect, not '
      + 'from measured radiation, so read gradients rather than absolute values.',
    legend: { type: 'ramp', unit: 'cool → hot', min: 'shaded', max: 'sun', stops: SOLAR_PALETTE },
    sourceMasked: true,
    build(ee) {
      const dem = ee.Image(ASSETS.SRTM)
      const aspect = ee.Terrain.aspect(dem)
      const slope = ee.Terrain.slope(dem).multiply(DEG)
      // Folded aspect (McCune & Keon): south-west = 1 is the hottest bearing in
      // the northern hemisphere, north-east = 0 the coolest. Weighted by the
      // sine of slope so flat ground reads as neutral rather than as either
      // extreme.
      const folded = aspect.subtract(225).multiply(DEG).cos().add(1).divide(2)
      const heat = folded.multiply(slope.sin())
      return { image: heat, vis: { min: 0, max: 0.7, palette: SOLAR_PALETTE } }
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

  // ── Greenness and canopy moisture ───────────────────────────────────────────
  //
  // Two Sentinel-2 indices, each offered twice: a recent rolling composite that
  // is free and always current, and a season-of-a-year composite for members,
  // for comparing the same phenological moment across years. NDVI is how much
  // green is there; NDMI is how wet that green is — the second flags a canopy
  // drawing on groundwater or a recent soaking, which is the half of fruiting
  // weather greenness alone misses.

  'ndvi-recent': recentIndexLayer({
    name: 'Greenness, recent (NDVI)',
    index: 'ndvi',
    group: 'Vegetation',
    palette: NDVI_PALETTE,
    note: 'Normalised Difference Vegetation Index from a cloud-filtered Sentinel-2 median over the '
      + 'chosen number of recent days, 10 m, global. Green is dense canopy, brown is bare or dormant '
      + 'ground. A wide window fills cloud gaps but blurs a fast green-up; a narrow one is sharper '
      + 'but gappier.',
  }),

  'ndmi-recent': recentIndexLayer({
    name: 'Canopy moisture, recent (NDMI)',
    index: 'ndmi',
    group: 'Vegetation',
    palette: NDMI_PALETTE,
    note: 'Normalised Difference Moisture Index from a cloud-filtered Sentinel-2 median over the '
      + 'chosen number of recent days, 10 m, global. Teal is moist vegetation, brown is dry or '
      + 'stressed — a canopy that greens up after rain shows here before the ground does. Read '
      + 'change over days, not the absolute value.',
  }),

  'ndvi-seasonal': seasonalIndexLayer({
    name: 'Greenness, by season (NDVI)',
    index: 'ndvi',
    group: 'Vegetation',
    palette: NDVI_PALETTE,
    note: 'The same NDVI, but a median over one season of one year rather than the last few weeks, '
      + 'so a dry summer can be set beside a wet one at the same point in the year. Computed from '
      + 'raw Sentinel-2 as you look, which is why it is a members’ layer.',
  }),

  'ndmi-seasonal': seasonalIndexLayer({
    name: 'Canopy moisture, by season (NDMI)',
    index: 'ndmi',
    group: 'Vegetation',
    palette: NDMI_PALETTE,
    note: 'The same NDMI over one season of one year, for comparing canopy moisture between years at '
      + 'the same phenological moment — a stand that stays teal through a dry August is drawing on '
      + 'water its neighbours have lost. Computed from raw Sentinel-2 as you look.',
  }),

  // ── Forest structure (USFS TreeMap) ─────────────────────────────────────────
  //
  // Three reads of the same modelled forest: how closed the canopy is, how tall
  // the stand, and the diameter class it falls in — together a proxy for a
  // stand's maturity, which is what decides whether a given host is old enough
  // to fruit its associates. TreeMap is a published asset gated at the members'
  // tier alongside the other structure and soil layers. US forests only, modelled
  // on the FIA plot grid from 2016, so a fire or a cut since then is not in it.

  'canopy-density': {
    name: 'Canopy density',
    group: 'Forest structure',
    tier: DEFAULT_TIER,
    attribution: 'USFS TreeMap via Google Earth Engine',
    opacity: 0.85,
    note: 'Percent live canopy cover from USFS TreeMap, 30 m, US forests only, modelled from 2016 '
      + 'FIA data. Closed canopy (deep blue) shades the ground and holds humidity; open stands (pale) '
      + 'dry faster. Unpainted is non-forest or outside the mapped area, not zero cover.',
    legend: { type: 'ramp', unit: '% cover', min: '0', max: '100', stops: CANOPY_PALETTE },
    sourceMasked: true,
    count: treeMapCount,
    build(ee) {
      return {
        image: treeMap2016(ee, 'CANOPYPCT'),
        vis: { min: 0, max: 100, palette: CANOPY_PALETTE },
      }
    },
  },

  'stand-height': {
    name: 'Stand height',
    group: 'Forest structure',
    tier: DEFAULT_TIER,
    attribution: 'USFS TreeMap via Google Earth Engine',
    opacity: 0.85,
    note: 'Dominant tree height in feet from USFS TreeMap, 30 m, US forests only, modelled from 2016 '
      + 'FIA data. A proxy for stand maturity: taller stands (deep magenta) are older forest, and an '
      + 'old host is what many ectomycorrhizal fruitings need. Unpainted is non-forest or outside the '
      + 'mapped area.',
    legend: { type: 'ramp', unit: 'feet', min: '0', max: '100+', stops: STAND_HEIGHT_PALETTE },
    sourceMasked: true,
    count: treeMapCount,
    build(ee) {
      return {
        image: treeMap2016(ee, 'STANDHT'),
        vis: { min: 0, max: 100, palette: STAND_HEIGHT_PALETTE },
      }
    },
  },

  'stand-size': {
    name: 'Stand-size class',
    group: 'Forest structure',
    tier: DEFAULT_TIER,
    attribution: 'USFS TreeMap via Google Earth Engine',
    opacity: 0.85,
    note: 'The diameter class of the stand from USFS TreeMap (STDSZCD), 30 m, US forests only, '
      + 'modelled from 2016 FIA data. Large-diameter stands are the mature forest an old-growth '
      + 'associate wants; seedling and nonstocked ground is recently disturbed. A structural class, '
      + 'not a species — pair it with the forest-type layer.',
    legend: { type: 'classes', items: STAND_SIZE_CLASSES },
    count: treeMapCount,
    build(ee) {
      const image = treeMap2016(ee, 'STDSZCD')
      // Codes run 1–5; a range guard keeps a future code from painting past the
      // end of the palette. Because this build masks, the layer does not declare
      // sourceMasked — the same shape as forest-type.
      const known = image.gte(1).and(image.lte(STAND_SIZE_CLASSES.length))
      return {
        image: image.updateMask(known),
        vis: { min: 1, max: STAND_SIZE_CLASSES.length, palette: STAND_SIZE_CLASSES.map((c) => c.color) },
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
