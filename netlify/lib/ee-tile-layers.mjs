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

import { MATSUTAKE_GREAT_GROUPS, SOIL_ORDERS, orderIndex } from './soil-taxonomy.mjs'

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
  // USDA soil taxonomy great groups, 250 m, global. About four hundred classes,
  // and the code → name table is carried on the image itself as the properties
  // grtgroup_class_values and grtgroup_class_names — which is why the two
  // layers built on it read that table rather than hardcoding one.
  OPENLANDMAP_GRTGROUP: 'OpenLandMap/SOL/SOL_GRTGROUP_USDA-SOILTAX_C/v01',
  // OpenLandMap volumetric water content at field capacity (33 kPa), 250 m,
  // global. One image whose bands b0…b200 are the depths in cm — how much water
  // the soil can hold, which is the standing property of the ground rather than
  // how wet it is on a given day.
  OPENLANDMAP_WATER_33KPA: 'OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01',
  // What is growing on the ground, by type rather than by greenness.
  GAP_LANDCOVER: 'USGS/GAP/CONUS/2011',
  // ESA WorldCover, global, 10 m. Served here rather than from the publisher's
  // own WMTS: that host went down and took the layer with it, and a tile
  // service nobody on this project can restart is a dependency rather than a
  // feature. Earth Engine already carried this asset for the land-cover
  // enrichment stage, so this costs a layer definition rather than a new
  // relationship.
  WORLDCOVER: 'ESA/WorldCover/v200',
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
  // SMAP L4 surface soil moisture, ~9 km, global, roughly three-hourly. Served
  // here rather than from GIBS: GIBS only publishes SMAP in EPSG:4326, and this
  // map is EPSG:3857, so the GIBS tiles 404'd. The `sm_surface` band is
  // volumetric water in the top 5 cm.
  SMAP: 'NASA/SMAP/SPL4SMGP/007',
  // CSP ERGo terrain indices from SRTM, global. Pre-computed at the source and
  // published as ready-to-use images, so no on-demand computation is needed.
  // mTPI is a multi-scale topographic position index (values −500–500 m) that
  // separates valley floors from ridgelines better than a single-radius TPI.
  // CHILI is a continuous heat-insolation load index (0–255) that integrates
  // slope angle, aspect and neighbourhood shading into one solar-loading score.
  CSP_SRTM_MTPI: 'CSP/ERGo/1_0/Global/SRTM_mTPI',
  CSP_SRTM_CHILI: 'CSP/ERGo/1_0/Global/SRTM_CHILI',
  // TNC Global Human Modification index v3, 90 m, global static snapshot. A
  // cumulative measure of how much human infrastructure (roads, agriculture,
  // urban, etc.) has modified each pixel, 0 = unmodified, 1 = fully modified.
  TNC_HM: 'TNC/HM/v3/90m_s',
  // JRC GHSL global population surfaces, 100 m, modelled for every 5-year epoch
  // from 1975 to 2030. Population count per grid cell, so a single pixel in a
  // dense city is hundreds of people.
  GHSL_POP: 'JRC/GHSL/P2023A/GHS_POP',
  // Copernicus Sentinel-1 C-band SAR, ~10 m, global. VV backscatter over land
  // rises with surface wetness (and roughness), so a recent mean is a radar
  // proxy for how wet the ground is — and unlike optical NDMI it sees through
  // cloud, which is most of what makes soil moisture hard to read in fall.
  S1_GRD: 'COPERNICUS/S1_GRD',
  // ECMWF ERA5-Land daily aggregates, ~11 km, global. A reanalysis, not a
  // sensor: it models the whole soil column, so it carries the two things SMAP
  // and the optical indices cannot — water below the top 5 cm, and soil
  // temperature — as `volumetric_soil_water_layer_1` and
  // `soil_temperature_level_1`.
  ERA5_LAND_DAILY: 'ECMWF/ERA5_LAND/DAILY_AGGR',
  // NOAA CPC Global Unified gauge-based daily precipitation, ~0.5° (~55 km),
  // land only, from 1979. Coarse and gauge-derived rather than radar or
  // satellite, so it is the long, consistent record of how much rain actually
  // fell — the `precipitation` band is the daily total in mm.
  NOAA_CPC_PRECIP: 'NOAA/CPC/Precipitation',
}

/**
 * ESA WorldCover's own class colours, so the map matches every other rendering
 * of this product rather than inventing a second palette for the same classes.
 *
 * The order is the product's own, which is what makes the remap below readable:
 * the nth colour is the nth code in WORLDCOVER_FROM.
 */
// `code` is the product's own class number, and it is what joins this table to
// anything else that speaks WorldCover. `aliases` carries the spellings the
// enrichment pipeline writes for the same class — scripts/enrich_with_rasters
// .py names code 60 "Bare / sparse vegetation" and code 90 "Wetland" — so a
// point can be coloured to match the layer under it without either side having
// to be renamed.
export const WORLDCOVER_CLASSES = [
  { code: 10, color: '#006400', label: 'Tree cover' },
  { code: 20, color: '#ffbb22', label: 'Shrubland' },
  { code: 30, color: '#ffff4c', label: 'Grassland' },
  { code: 40, color: '#f096ff', label: 'Cropland' },
  { code: 50, color: '#fa0000', label: 'Built-up' },
  { code: 60, color: '#b4b4b4', label: 'Bare / sparse', aliases: ['Bare / sparse vegetation'] },
  { code: 70, color: '#f0f0f0', label: 'Snow and ice' },
  { code: 80, color: '#0064c8', label: 'Permanent water', aliases: ['Water'] },
  { code: 90, color: '#0096a0', label: 'Herbaceous wetland', aliases: ['Wetland'] },
  { code: 95, color: '#00cf75', label: 'Mangroves' },
  { code: 100, color: '#fae6a0', label: 'Moss and lichen' },
]

// WorldCover codes are decades with one odd one out at 95, so they cannot be
// used as palette indices directly — a linear stretch from 10 to 100 would put
// every class at the wrong colour. Remapped to 1..11 instead, the same way the
// GAP layer is.
//
// Derived from the table above rather than written out again: as two parallel
// lists they could fall out of order, and nothing would say so — the map would
// simply paint the wrong class the wrong colour.
export const WORLDCOVER_FROM = WORLDCOVER_CLASSES.map((c) => c.code)
export const WORLDCOVER_TO = WORLDCOVER_FROM.map((_, i) => i + 1)

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
export const TEXTURE_CLASSES = [
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
/** Dry ground (brown) to wet ground (deep blue). */
const SOIL_MOISTURE_PALETTE = ['#8c6d3f', '#c7a76c', '#e8dfc0', '#96c8c0', '#3d8fb0', '#16407a']
// Range richness: pale to deep violet, so more overlapping species ranges read
// as a denser colour without colliding with the moisture blues or the fire reds.
const INAT_RANGE_PALETTE = ['#f2e6f7', '#dcc2ec', '#c39bdd', '#a86fcb', '#8c3fb5', '#5c1f86']

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
/**
 * The most classes one selection may carry.
 *
 * Not a rendering limit and no longer a URL limit: a selection this size goes
 * in a request body rather than a query string, so every class in the raster
 * can be chosen at once. What is left is a bound on how much work one request
 * may ask for, well above the four hundred classes the raster actually has and
 * low enough that a malformed or hostile request cannot ask for a remap of a
 * million values.
 */
export const CODE_LIMIT = 2000

/**
 * A list of class codes, as a canonical comma-separated string.
 *
 * Deduplicated and sorted, so the same selection made in a different order is
 * one entry in the tile cache rather than two. Accepts an array or a string,
 * because it is called from the browser with what a checkbox list produces and
 * from the server with what a query string or a JSON body produces.
 */
export function normaliseCodes(raw, max = CODE_LIMIT) {
  const parts = Array.isArray(raw) ? raw : String(raw ?? '').split(',')
  const seen = new Set()
  for (const part of parts) {
    const text = String(part).trim()
    if (!text) continue
    const n = Math.floor(Number(text))
    if (!Number.isFinite(n)) throw new LayerError('Classes must be a list of whole numbers.')
    seen.add(n)
  }
  const cap = Number.isFinite(max) ? max : CODE_LIMIT
  if (seen.size > cap) {
    throw new LayerError(`At most ${cap} classes at a time; ${seen.size} were chosen.`)
  }
  return [...seen].sort((a, b) => a - b).join(',')
}

/** The codes in a normalised selection, as numbers. */
export function codeList(value) {
  if (!value) return []
  return String(value).split(',').filter(Boolean).map(Number)
}

function readParams(schema, input = {}) {
  const out = {}
  for (const [key, spec] of Object.entries(schema)) {
    const raw = input[key]
    if (raw === undefined || raw === null || raw === '') {
      out[key] = typeof spec.default === 'function' ? spec.default() : spec.default
      continue
    }
    if (spec.type === 'int' || spec.type === 'yearSelect') {
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
    } else if (spec.type === 'codes') {
      out[key] = normaliseCodes(raw, spec.max)
    } else if (spec.type === 'text') {
      // A free typed value — a taxon name to search for. Constrained to the
      // characters a scientific name uses (letters, spaces, hyphen, period,
      // parentheses, ×) and a short length, so it cannot carry anything that
      // is not a name into the Earth Engine string filter.
      const text = String(raw).trim()
      const max = spec.maxLength || 60
      if (!text) throw new LayerError(`${spec.label} cannot be empty.`)
      if (text.length > max) throw new LayerError(`${spec.label} must be ${max} characters or fewer.`)
      if (!/^[A-Za-z][A-Za-z .()×-]*$/.test(text)) {
        throw new LayerError(`${spec.label} may only contain letters, spaces, hyphens and periods.`)
      }
      out[key] = text
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

// ─────────────────────────────────────────────────────────────────────────────
// Soil taxonomy
// ─────────────────────────────────────────────────────────────────────────────

/** The great-group raster, which every soil taxonomy layer starts from. */
const grtgroup = (ee) => ee.Image(ASSETS.OPENLANDMAP_GRTGROUP).select('grtgroup')

/**
 * The raster's own code → name table.
 *
 * Read off the image rather than written down here. Four hundred classes is a
 * table nobody will maintain, and a table that drifts from the raster paints
 * one soil and names another — which is worse than having no names at all.
 *
 * This is a `prepare`, so the tile function evaluates it once and hands the
 * plain arrays to build(). Everything after that is ordinary JavaScript and can
 * be tested without Earth Engine.
 */
const grtgroupTable = (ee) => ee.Dictionary({
  values: grtgroup(ee).get('grtgroup_class_values'),
  names: grtgroup(ee).get('grtgroup_class_names'),
})

const SOIL_ORDER_PALETTE = SOIL_ORDERS.map((o) => o.color)
const SOIL_ORDER_CLASSES = SOIL_ORDERS.map((o) => ({ color: o.color, label: o.name }))

/**
 * The catalogue.
 *
 * `build(ee, params, prepared)` returns the image and how to paint it. `legend`
 * is the key drawn on the map, in the same shape the existing tile layers use,
 * so the map's legend code does not have to know an Earth Engine layer from any
 * other.
 *
 * `prepare(ee)` is optional: an Earth Engine value the tile function evaluates
 * before build and passes in as `prepared`. It exists for the one thing a
 * synchronous build cannot do — read a table off the asset itself — and its
 * result is memoised, because what it reads does not change.
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
        type: 'yearSelect', label: 'Year', default: () => THIS_YEAR() - MODIS_LAG_YEARS,
        min: MODIS_FIRST_YEAR, max: () => THIS_YEAR() - MODIS_LAG_YEARS,
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
    // Cyclic, so the key is marked all the way round rather than just at its
    // ends: the ticks sit evenly under the gradient at N, E, S, W and back to N,
    // which is what tells east from west at a glance.
    legend: { type: 'ramp', unit: '° from north', ticks: ['N', 'E', 'S', 'W', 'N'], stops: ASPECT_PALETTE },
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

  'land-cover': {
    name: 'Land cover (ESA)',
    group: 'Ground',
    // Free, because it was free before. This layer was served from the
    // publisher's own tile host and open to everyone; moving it to Earth Engine
    // is a fix for that host going down, and a fix should not quietly take a
    // layer away from people who had it. The render is cached and shared, so
    // the marginal cost of the open tier is one render per six hours rather
    // than one per viewer.
    tier: 'free',
    attribution: 'ESA WorldCover 2021 (CC BY 4.0) via Google Earth Engine',
    opacity: 0.55,
    note: 'ESA WorldCover at 10 m, from 2021. High resolution but not current: a burn, a '
      + 'clear-cut or a new development since then is not in it. It says "tree cover" and not '
      + 'which trees — the forest type layer answers that, for the US.',
    legend: { type: 'classes', items: WORLDCOVER_CLASSES },
    // Masks its own, like the forest type layer and for the same reason: after
    // a remap the unlisted codes are gone, and the range guard below is the
    // explicit mask. `sourceMasked` is for layers that leave their no-data to
    // the publisher — declaring it here and then masking anyway would be two
    // contradictory claims about the same image.
    build(ee) {
      // An ImageCollection of one annual mosaic per version. Mosaicking rather
      // than taking first(), so a version published as tiles still resolves to
      // one continuous image.
      const remapped = ee.ImageCollection(ASSETS.WORLDCOVER)
        .select('Map')
        .mosaic()
        .remap(WORLDCOVER_FROM, WORLDCOVER_TO)
      // remap masks anything unlisted, and the range guard keeps a future code
      // addition from painting past the end of the palette.
      const known = remapped.gte(1).and(remapped.lte(WORLDCOVER_CLASSES.length))
      return {
        image: remapped.updateMask(known),
        vis: {
          min: 1,
          max: WORLDCOVER_CLASSES.length,
          palette: WORLDCOVER_CLASSES.map((c) => c.color),
        },
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

  'soil-taxonomy': {
    name: 'Soil taxonomy (USDA orders)',
    group: 'Soil',
    tier: DEFAULT_TIER,
    attribution: 'OpenLandMap USDA soil taxonomy great groups via Google Earth Engine',
    opacity: 0.8,
    note: 'The USDA soil order at each pixel, predicted globally at 250 m. The source is the '
      + 'great-group level — about four hundred classes — which is far too many to read as a '
      + 'key, so the map paints the twelve orders above them and the soil panel names the great '
      + 'group under your cursor. Orders separate soils by how they formed, so the boundaries '
      + 'often follow geology and climate rather than anything visible on the surface. A model '
      + 'prediction, not a soil survey: right about a hillside, unreliable about a square metre.',
    legend: { type: 'classes', items: SOIL_ORDER_CLASSES },
    // The class browser draws these twelve as chips you can filter by, so the
    // flat swatch list would be the same twelve entries again, directly above
    // them, and not clickable.
    legendInBrowser: true,
    // The source masks open water and leaves a real class everywhere else. What
    // this layer masks is its own doing: see selfMask below.
    sourceMasked: true,
    prepare: grtgroupTable,
    build(ee, params, table) {
      const values = table?.values || []
      // The order is the end of the great group's own name. That is what the
      // taxonomy's naming is for, so there is no lookup table to keep in step
      // with the raster — see netlify/lib/soil-taxonomy.mjs.
      const orders = (table?.names || []).map(orderIndex)
      return {
        // selfMask, because orderIndex gives 0 to a class that is not a great
        // group at all. Painted, zero would be a twelfth of the legend claiming
        // ground it knows nothing about; masked, it is honestly blank.
        image: grtgroup(ee).remap(values, orders, 0).selfMask(),
        vis: { min: 1, max: SOIL_ORDERS.length, palette: SOIL_ORDER_PALETTE },
      }
    },
  },

  'soil-taxonomy-select': {
    name: 'Soil taxonomy: chosen classes',
    group: 'Soil',
    tier: DEFAULT_TIER,
    attribution: 'OpenLandMap USDA soil taxonomy great groups via Google Earth Engine',
    opacity: 0.8,
    note: 'Paints only the great groups you choose, and leaves every other pixel blank. Choose '
      + 'them in the class list below: search by name or by order, tick as many as you want, and '
      + 'the map redraws. Each chosen class keeps the colour of its order, so a selection that '
      + 'spans several orders can still be told apart. This is a soil filter and not a '
      + 'prediction — it says the ground is the kind you asked for, not that anything grows '
      + 'there, and it knows nothing about the trees that decide whether anything can.',
    params: {
      codes: { type: 'codes', label: 'Classes', default: '', max: CODE_LIMIT },
    },
    // The browser's order chips are the key: each chosen class draws in its own
    // order's colour, and the chips are that list, clickable.
    legend: { type: 'classes', items: SOIL_ORDER_CLASSES },
    legendInBrowser: true,
    sourceMasked: true,
    prepare: grtgroupTable,
    build(ee, params, table) {
      const chosen = new Set(codeList(params?.codes))
      const values = table?.values || []
      const names = table?.names || []
      const from = []
      const to = []
      for (let i = 0; i < values.length; i += 1) {
        if (!chosen.has(values[i])) continue
        from.push(values[i])
        to.push(orderIndex(names[i]))
      }
      return {
        // remap with a default of 0 and then selfMask: a class nobody chose is
        // unpainted rather than painted as "no". Blank here means "not one of
        // the ones you asked for", which is what it should mean.
        image: grtgroup(ee).remap(from, to, 0).selfMask(),
        vis: { min: 1, max: SOIL_ORDERS.length, palette: SOIL_ORDER_PALETTE },
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

  'soil-moisture': {
    name: 'Soil moisture',
    group: 'Soil',
    // Free: it answers the question a forager actually asks after rain, and it
    // is a cheap mean of a coarse published product. Global, unlike the SOLUS
    // soil layers, so it is the one soil layer that works outside the US.
    tier: 'free',
    attribution: 'NASA SMAP L4 via Google Earth Engine',
    opacity: 0.7,
    note: 'Modelled water in the top 5 cm of soil from NASA SMAP, ~9 km, global, averaged over the '
      + 'chosen number of recent days. A model assimilating satellite retrievals, not a measurement '
      + 'of your patch, and coarse: a cell is larger than most places on this map. Blue is wet, brown '
      + 'is dry. It lags real time by a couple of days.',
    params: {
      days: { type: 'int', label: 'Days to average', default: 5, min: 1, max: 30 },
    },
    legend: { type: 'ramp', unit: 'm³/m³', min: '0.0', max: '0.6', stops: SOIL_MOISTURE_PALETTE },
    // SMAP masks open water and permanently frozen ground itself, and every
    // pixel it leaves is a real retrieval, so masking again would only discard
    // coastline. See the note on sourceMasked above.
    sourceMasked: true,
    count: (ee, { days }) => {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      return ee.ImageCollection(ASSETS.SMAP)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)).size()
    },
    build(ee, { days }) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      const image = ee.ImageCollection(ASSETS.SMAP)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .select('sm_surface')
        .mean()
      return { image, vis: { min: 0, max: 0.6, palette: SOIL_MOISTURE_PALETTE } }
    },
  },

  'cpc-precip': {
    name: 'Rain accumulation, gauge (CPC)',
    group: 'Weather',
    // Free: a cheap sum of a coarse published product, global (land), and the
    // one rainfall layer with a decades-long consistent record behind it.
    tier: 'free',
    attribution: 'NOAA CPC Global Unified gauge-based precipitation via Google Earth Engine',
    opacity: 0.7,
    // Gauge-based land product: ocean is masked in the source and every land
    // pixel is a real total, so summing zero rain is a true zero, not no-data.
    sourceMasked: true,
    note: 'Total gauge-analysed rainfall over the chosen recent days, from NOAA CPC, ~55 km, land only. '
      + 'Gauge-derived and coarse — a cell is far larger than a foraging patch — but it is the long, '
      + 'consistent rain record, good for how wet a region has been rather than where a shower fell. '
      + 'It lags real time by a day or two, so a one-day window near today can come back empty.',
    params: {
      days: { type: 'int', label: 'Days to total', default: 7, min: 1, max: 60 },
    },
    legend: {
      type: 'ramp', unit: 'mm', min: '0', max: '100+',
      stops: ['#f7fbff', '#d0e1f2', '#94c4df', '#4a97c9', '#1764ab', '#08306b'],
    },
    count: (ee, { days }) => {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      return ee.ImageCollection(ASSETS.NOAA_CPC_PRECIP)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)).size()
    },
    build(ee, { days }) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      const image = ee.ImageCollection(ASSETS.NOAA_CPC_PRECIP)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .select('precipitation')
        .sum()
      return { image, vis: { min: 0, max: 100, palette: ['#f7fbff', '#d0e1f2', '#94c4df', '#4a97c9', '#1764ab', '#08306b'] } }
    },
  },

  'field-capacity': {
    name: 'Soil water capacity (field capacity)',
    group: 'Soil',
    // Free and static: one published image, one cached render per depth shared
    // by everyone. It is a property of the soil itself, not a recent condition,
    // so it has no date window — just which depth you read.
    tier: 'free',
    attribution: 'OpenLandMap volumetric water content at 33 kPa via Google Earth Engine',
    opacity: 0.8,
    // Ocean is masked in the source and every land pixel is a real estimate.
    sourceMasked: true,
    note: 'Volumetric water the soil holds at field capacity (33 kPa suction), 250 m, global, at the '
      + 'chosen depth. This is capacity, not today’s moisture: how much water the ground can retain '
      + 'after it drains, which is what keeps a site damp between rains. Deeper blue holds more. A '
      + 'modelled property of the soil, so it does not change with the weather.',
    params: {
      depth: { type: 'enum', label: 'Depth (cm)', default: '0', values: ['0', '10', '30', '60', '100', '200'] },
    },
    legend: {
      type: 'ramp', unit: '% vol', min: '5', max: '45',
      stops: ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'],
    },
    build(ee, { depth }) {
      const image = ee.Image(ASSETS.OPENLANDMAP_WATER_33KPA).select(`b${depth}`)
      return {
        image,
        vis: {
          min: 5,
          max: 45,
          palette: ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'],
        },
      }
    },
  },

  'radar-moisture': {
    name: 'Radar moisture proxy (Sentinel-1)',
    group: 'Soil',
    // Computed from raw SAR as you look — a mean of every pass in the window —
    // so it is a members' layer like dNBR rather than a cheap published product.
    tier: DEFAULT_TIER,
    attribution: 'Copernicus Sentinel-1 GRD via Google Earth Engine',
    opacity: 0.75,
    sourceMasked: true,
    note: 'Mean VV backscatter from Sentinel-1 C-band radar over the chosen recent days, 10 m. '
      + 'Radar sees the ground through cloud, and brighter VV usually means wetter soil — but roughness '
      + 'and vegetation raise it too, so read it as a proxy, not a moisture measurement. Blue is the '
      + 'wetter (brighter) end. Coverage is per-orbit, so a short window can leave gaps.',
    slow: true,
    params: {
      days: { type: 'int', label: 'Days to average', default: 14, min: 1, max: 60 },
    },
    legend: {
      type: 'ramp', unit: 'dB (VV)', min: 'dry', max: 'wet',
      stops: ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'],
    },
    count: (ee, { days }) => {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      return ee.ImageCollection(ASSETS.S1_GRD)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW')).size()
    },
    build(ee, { days }) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      const image = ee.ImageCollection(ASSETS.S1_GRD)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .select('VV')
        .mean()
      return { image, vis: { min: -20, max: -5, palette: ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494'] } }
    },
  },

  'soil-moisture-column': {
    name: 'Soil moisture, root zone (ERA5-Land)',
    group: 'Soil',
    // Free, like SMAP: a cheap mean of a coarse published reanalysis, global,
    // and the same question a forager asks after rain — but of the 0–7 cm layer
    // a reanalysis models rather than the top 5 cm a satellite retrieves.
    tier: 'free',
    attribution: 'Copernicus ECMWF ERA5-Land via Google Earth Engine',
    opacity: 0.7,
    sourceMasked: true,
    note: 'Modelled volumetric water in the top 0–7 cm of soil from ERA5-Land, ~11 km, global, averaged '
      + 'over the chosen recent days. A reanalysis, not a measurement, and coarse: a cell is larger than '
      + 'most places on this map. Blue is wet. It lags real time by about five days, so a short window '
      + 'near today can come back empty.',
    params: {
      days: { type: 'int', label: 'Days to average', default: 14, min: 1, max: 60 },
    },
    legend: {
      type: 'ramp', unit: 'm³/m³', min: '0.1', max: '0.4',
      stops: ['#ffffd9', '#c7e9b4', '#41b6c4', '#225ea8', '#081d58'],
    },
    count: (ee, { days }) => {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      return ee.ImageCollection(ASSETS.ERA5_LAND_DAILY)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)).size()
    },
    build(ee, { days }) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      const image = ee.ImageCollection(ASSETS.ERA5_LAND_DAILY)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .select('volumetric_soil_water_layer_1')
        .mean()
      return { image, vis: { min: 0.1, max: 0.4, palette: ['#ffffd9', '#c7e9b4', '#41b6c4', '#225ea8', '#081d58'] } }
    },
  },

  'soil-temperature': {
    name: 'Soil temperature (ERA5-Land)',
    group: 'Weather',
    // Free: the same cheap ERA5-Land mean, one band over. Soil temperature is
    // the other half of whether the ground is ready to fruit, and no satellite
    // layer here carries it.
    tier: 'free',
    attribution: 'Copernicus ECMWF ERA5-Land via Google Earth Engine',
    opacity: 0.7,
    sourceMasked: true,
    note: 'Modelled temperature of the top 0–7 cm of soil from ERA5-Land, ~11 km, global, averaged over '
      + 'the chosen recent days and shown in °C. A reanalysis, not a probe in your patch, and coarse. '
      + 'Warm is red, cold is blue. It lags real time by about five days.',
    params: {
      days: { type: 'int', label: 'Days to average', default: 14, min: 1, max: 60 },
    },
    legend: {
      type: 'ramp', unit: '°C', min: '0', max: '15',
      stops: ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027'],
    },
    count: (ee, { days }) => {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      return ee.ImageCollection(ASSETS.ERA5_LAND_DAILY)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10)).size()
    },
    build(ee, { days }) {
      const end = new Date()
      const start = new Date(end.getTime() - days * 86400000)
      // ERA5-Land carries soil temperature in kelvin; °C is what a reader can use.
      const image = ee.ImageCollection(ASSETS.ERA5_LAND_DAILY)
        .filterDate(start.toISOString().slice(0, 10), end.toISOString().slice(0, 10))
        .select('soil_temperature_level_1')
        .mean()
        .subtract(273.15)
      return { image, vis: { min: 0, max: 15, palette: ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027'] } }
    },
  },

  // ── Human influence ────────────────────────────────────────────────────────
  //
  // Two global datasets that capture what people have done to the landscape.
  // Human modification tells you where the ground is still ecologically intact;
  // population tells you how many people live nearby — a forager's proxy for
  // access pressure and how hard a spot is likely to be hit.

  'human-modification': {
    name: 'Human modification index',
    group: 'Human influence',
    tier: 'free',
    attribution: 'TNC Global Human Modification v3 via Google Earth Engine',
    opacity: 0.65,
    sourceMasked: true,
    note: 'TNC Global Human Modification index (gHM) at 90 m, a static snapshot. '
      + 'Values range 0 (wilderness) to 1 (fully modified by roads, agriculture, '
      + 'built-up land and similar). Low values (green) are where ecological '
      + 'processes still run largely intact. Read gradients, not boundaries: the '
      + 'transition from modified to intact is rarely a line.',
    legend: {
      type: 'ramp', unit: 'gHM', min: '0', max: '1',
      stops: ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'],
    },
    build(ee) {
      const image = ee.ImageCollection(ASSETS.TNC_HM)
        .mosaic()
        .select('cumulative_human_modification')
      return {
        image: image.updateMask(image.gte(0)),
        vis: { min: 0, max: 1, palette: ['#1a9641', '#a6d96a', '#ffffbf', '#fdae61', '#d7191c'] },
      }
    },
  },

  'population-density': {
    name: 'Population density (GHSL)',
    group: 'Human influence',
    tier: 'free',
    attribution: 'JRC GHSL GHS-POP P2023A via Google Earth Engine',
    opacity: 0.7,
    sourceMasked: true,
    note: 'Modelled population count per 100 m cell from the Global Human '
      + 'Settlement Layer, for the chosen epoch (1975–2030 in five-year steps). '
      + 'Shown on a log scale — one pale pixel in a city holds far more people '
      + 'than one bright pixel in a village. Useful as a proxy for how heavily '
      + 'a spot is visited rather than for reading absolute counts.',
    params: {
      year: {
        type: 'enum', label: 'Epoch',
        default: '2020',
        values: ['1975', '1980', '1985', '1990', '1995', '2000', '2005', '2010', '2015', '2020', '2025', '2030'],
      },
    },
    legend: {
      type: 'ramp', unit: 'people / cell (log)', min: '1', max: '1000+',
      stops: ['#feebe2', '#fcc5c0', '#fa9fb5', '#f768a1', '#ae017e', '#7a0177'],
    },
    count: (ee, { year }) => ee.ImageCollection(ASSETS.GHSL_POP)
      .filterDate(`${year}-01-01`, `${Number(year) + 1}-01-01`).size(),
    build(ee, { year }) {
      const image = ee.ImageCollection(ASSETS.GHSL_POP)
        .filterDate(`${year}-01-01`, `${Number(year) + 1}-01-01`)
        .first()
        .select('population_count')
      // Log1p so zero stays zero and the distribution compresses enough to read
      // without saturating on dense cities. Mask where no one lives.
      const logPop = image.log1p()
      return {
        image: logPop.updateMask(image.gt(0)),
        vis: { min: 0, max: 7, palette: ['#feebe2', '#fcc5c0', '#fa9fb5', '#f768a1', '#ae017e', '#7a0177'] },
      }
    },
  },

  // ── Extended terrain indices ────────────────────────────────────────────────
  //
  // Two pre-computed CSP/ERGo indices that go beyond the SRTM layers above.
  // mTPI integrates position across multiple neighbourhood scales, so a summit
  // plateau reads differently from a narrow ridge even though both sit above
  // their immediate surroundings. CHILI is a heat-insolation score that folds
  // shading into the aspect estimate, so a north-facing slope shaded by a ridge
  // reads colder than a north-facing slope with an open sky.

  'srtm-mtpi': {
    name: 'Topographic position (mTPI)',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'CSP ERGo / NASA SRTM via Google Earth Engine',
    opacity: 0.7,
    sourceMasked: true,
    note: 'Multi-Scale Topographic Position Index from CSP ERGo at 90 m, global. '
      + 'Negative values (blue) are valley floors and basins; positive (red) are '
      + 'ridgelines and summits. Computed by comparing each pixel to its '
      + 'neighbourhood at several radii, so a broad plateau reads as flat even '
      + 'if it sits above a local hollow. A proxy for cold-air pooling, drainage '
      + 'and moisture accumulation at the landscape scale.',
    legend: {
      type: 'ramp', unit: 'mTPI (m)', min: 'valley', max: 'ridge',
      stops: ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'],
    },
    build(ee) {
      const image = ee.Image(ASSETS.CSP_SRTM_MTPI).select('constant')
      return {
        image,
        vis: { min: -300, max: 300, palette: ['#2166ac', '#67a9cf', '#d1e5f0', '#f7f7f7', '#fddbc7', '#ef8a62', '#b2182b'] },
      }
    },
  },

  'srtm-chili': {
    name: 'Heat-insolation load (CHILI)',
    group: 'Terrain analysis',
    tier: 'free',
    attribution: 'CSP ERGo / NASA SRTM via Google Earth Engine',
    opacity: 0.7,
    sourceMasked: true,
    note: 'Continuous Heat-Insolation Load Index from CSP ERGo at 90 m, global. '
      + 'Low values (blue) are cold, shaded or north-facing ground; high values '
      + '(red) are hot, sun-exposed south- and west-facing slopes. Unlike the '
      + 'simpler solar-exposure layer, CHILI accounts for local shading by '
      + 'adjacent terrain, so a south-facing bench in a canyon reads cooler than '
      + 'one on an open hillside.',
    legend: {
      type: 'ramp', unit: 'CHILI', min: 'cool / shaded', max: 'hot / exposed',
      stops: ['#2166ac', '#67a9cf', '#d1e5f0', '#fee090', '#fc8d59', '#d73027'],
    },
    build(ee) {
      const image = ee.Image(ASSETS.CSP_SRTM_CHILI).select('constant')
      return {
        image,
        vis: { min: 0, max: 255, palette: ['#2166ac', '#67a9cf', '#d1e5f0', '#fee090', '#fc8d59', '#d73027'] },
      }
    },
  },

  'inat-range': {
    name: 'Species range richness (iNaturalist)',
    group: 'Species',
    // Computed from a BigQuery-backed range map on every view, so it is a
    // members' layer like dNBR rather than a shared cached render. It also needs
    // the deployment's Earth Engine project to have BigQuery access; without it
    // the render fails with Earth Engine's own message rather than silently.
    tier: DEFAULT_TIER,
    attribution: 'iNaturalist open range maps via Google Earth Engine',
    opacity: 0.6,
    slow: true,
    note: 'Modelled iNaturalist ranges for every species whose name matches the taxon you type, '
      + 'summed so the colour is how many of those species range over each place — a richness '
      + 'heatmap, not observations. Type a genus (Morchella, Cantharellus) or a species. These are '
      + 'coarse expert-and-model range maps, not where anyone found one, and not every taxon has a '
      + 'published range. Deeper colour means more overlapping ranges.',
    params: {
      taxon: { type: 'text', label: 'Taxon name', default: 'Morchella', maxLength: 60 },
    },
    legend: {
      type: 'ramp', unit: 'overlapping ranges', min: '1', max: '8+',
      stops: INAT_RANGE_PALETTE,
    },
    build(ee, { taxon }) {
      const ranges = ee.FeatureCollection
        .loadBigQueryTable('earth-engine-public-data.inaturalist_open_range_map.multispecies_latest')
        .filter(ee.Filter.stringContains('name', taxon))
        // One flat weight per range, so the sum below is a count of how many
        // species' ranges cover a pixel rather than an accident of some property.
        .map((f) => f.set('present', 1))
      // Summed to a raster: overlapping ranges add up, which is the richness the
      // legend reads. Zero (no range here) is masked so it is honestly blank
      // rather than the palette's lightest colour claiming ground it does not
      // cover.
      const image = ranges.reduceToImage(['present'], ee.Reducer.sum())
      return {
        image: image.updateMask(image.gt(0)),
        vis: { min: 1, max: 8, palette: INAT_RANGE_PALETTE },
      }
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
/**
 * A layer's visualisation, in the shape the Earth Engine client will accept.
 *
 * The Node client reads `min`, `max` and `gamma` with a helper that does
 * `csv.split(',')` on whatever it is handed, so a NUMBER throws
 * "csv.split is not a function" — and a numeric zero slips through only because
 * zero is falsy and short-circuits before the split. Every layer in this
 * catalogue declares numeric bounds, the natural way to write them, which is
 * why not one of them could render.
 *
 * The Code Editor accepts numbers here, which is what makes this so easy to
 * write and so hard to spot: the same visualisation object works in the browser
 * and throws in the Node client, and the error names neither the parameter nor
 * the layer.
 *
 * Arrays are joined rather than stringified, so a per-band stretch arrives as
 * the CSV the client is about to parse back out. `palette` is deliberately left
 * alone — that one is accepted as an array and used as-is.
 */
export function visParams(vis = {}) {
  const out = { ...vis }
  for (const key of ['min', 'max', 'gamma']) {
    if (!(key in out)) continue
    const value = out[key]
    if (value === null || value === undefined || value === '') {
      delete out[key]
      continue
    }
    out[key] = Array.isArray(value) ? value.join(',') : String(value)
  }
  return out
}

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
    // Says the layer has a class table worth asking for, without sending four
    // hundred classes to every viewer of the catalogue. The map fetches it when
    // the layer is switched on.
    classes: layer.prepare ? 'great-groups' : undefined,
    legendInBrowser: layer.legendInBrowser || undefined,
    params: Object.fromEntries(Object.entries(layer.params || {}).map(([k, spec]) => {
      const min = typeof spec.min === 'function' ? spec.min() : spec.min
      const max = typeof spec.max === 'function' ? spec.max() : spec.max
      const values = spec.type === 'yearSelect'
        ? Array.from({ length: max - min + 1 }, (_, i) => max - i)
        : spec.values
      return [k, {
        label: spec.label,
        type: spec.type,
        default: typeof spec.default === 'function' ? spec.default() : spec.default,
        min,
        max,
        values,
      }]
    })),
  }
}

export const EE_LAYER_CATALOGUE = EE_LAYER_KEYS.map(describeLayer)
