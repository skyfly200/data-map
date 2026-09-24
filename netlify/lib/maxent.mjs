// The modelling core: MaxEnt habitat suitability from presence points.
//
// Enrichment produces a feature matrix — every environmental layer sampled at
// each observation. A species distribution model reads exactly that: it learns
// the environment where a species was seen and projects a suitability surface
// across a region. This module is the reusable heart of that step.
//
// It is split deliberately. Everything above buildSuitabilityImage is pure — the
// predictor registry, the request validation, the background plan — so the
// arithmetic that decides what a model does can be tested without Earth Engine.
// buildSuitabilityImage is the one function that touches `ee`, and it takes the
// `ee` object as an argument so it too can be exercised against a stub.
//
// What this module does NOT do: load the presence points (the job source path
// already does that) or store the result. It turns "these presences, these
// predictors, this region" into an Earth Engine image and how to paint it.

import { SRTM, S2_SR, ERA5_DAILY, CHIRPS_DAILY, SOLUS100, MODIS_BURN, TREEMAP, SpecError, normaliseBounds } from './ee-pipeline.mjs'
import { ASSETS } from './ee-tile-layers.mjs'
import { CHUNK_SIZE } from './quotas.mjs'

// Additional asset IDs used only by the extended model predictor set.
// Reference ASSETS where possible to stay in sync with the map layer definitions.
const WORLDCLIM_BIO = 'WORLDCLIM/V1/BIO'
const NLCD_TCC = 'USGS/NLCD_RELEASES/2021_REL/TCC/v2021-4'
const HANSEN_GFC = 'UMD/hansen/global_forest_change_2023_v1_11'
const {
  CSP_SRTM_MTPI, CSP_SRTM_CHILI, CSP_HM,
  MERIT_HYDRO, GHSL_POP,
  OPENLANDMAP_WATER_33KPA,
} = ASSETS

/**
 * The predictors a suitability model may use.
 *
 * Static by design: a suitability surface is a claim about the ground, so its
 * predictors describe the ground, not the weather of one day. Each is a
 * continuous Earth Engine image reduced to a single band — MaxEnt reads numbers,
 * and a categorical layer like land cover would need one-hot encoding that the
 * first cut does not do. All free, so a model spends only the compute it runs.
 */
export const MAXENT_PREDICTORS = {
  elevation: {
    label: 'Elevation',
    image: (ee) => ee.Image(SRTM).select('elevation'),
  },
  slope: {
    label: 'Slope',
    image: (ee) => ee.Terrain.slope(ee.Image(SRTM).select('elevation')),
  },
  aspect: {
    label: 'Aspect',
    image: (ee) => ee.Terrain.aspect(ee.Image(SRTM).select('elevation')),
  },
  ndvi: {
    label: 'Vegetation (NDVI)',
    // A multi-year median, so a cloud on one date does not become the predictor.
    image: (ee) => ee.ImageCollection(S2_SR)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
      .select(['B8', 'B4'])
      .median()
      .normalizedDifference(['B8', 'B4'])
      .rename('ndvi'),
  },
  soil_moisture: {
    label: 'Soil moisture (normal)',
    // The long-run mean of the top-layer soil water: the standing wetness of a
    // place rather than today's.
    image: (ee) => ee.ImageCollection(ERA5_DAILY)
      .select('volumetric_soil_water_layer_1')
      .mean()
      .rename('soil_moisture'),
  },

  // ── Climate normals ────────────────────────────────────────────────────────
  // The enrichment samples weather on the day of each record, but a suitability
  // surface is a claim about a place, not a day, so a per-record daily value has
  // nothing to project onto a pixel. The honest way to bring weather into a
  // static model is the climate normal: the long-run average, which IS a
  // property of the place. So the per-date layers the model left out return here
  // as their multi-year means.
  precip_normal: {
    label: 'Rainfall (normal)',
    // Mean daily CHIRPS rainfall over the whole record. The absolute scale does
    // not matter to MaxEnt — only how places compare — so the daily mean stands
    // in for "how wet this place is on average" without an annual multiply.
    image: (ee) => ee.ImageCollection(CHIRPS_DAILY)
      .select('precipitation')
      .mean()
      .rename('precip_normal'),
  },
  temp_normal: {
    label: 'Temperature (normal)',
    // Mean daily 2 m air temperature from ERA5-Land, in °C. The standing warmth
    // of a place, which for most species is the strongest climate predictor of
    // where they can live at all.
    image: (ee) => ee.ImageCollection(ERA5_DAILY)
      .select('temperature_2m')
      .mean()
      .subtract(273.15)
      .rename('temp_normal'),
  },

  // ── WorldClim bioclimatic variables ───────────────────────────────────────
  // Static 1 km² climate normals — more commonly used in MaxEnt literature than
  // ERA5 means, and much cheaper to sample (one image, not a collection mean).
  annual_precip: {
    label: 'Annual precipitation (WorldClim)',
    image: (ee) => ee.Image(WORLDCLIM_BIO).select('bio12').rename('annual_precip'),
  },
  annual_temp: {
    label: 'Mean annual temperature (WorldClim)',
    // WorldClim bio01 is in °C × 10; divide to get the natural scale.
    image: (ee) => ee.Image(WORLDCLIM_BIO).select('bio01').divide(10).rename('annual_temp'),
  },

  // ── Extended terrain derivatives ──────────────────────────────────────────
  northness: {
    label: 'Northness (cos aspect)',
    // cos(aspect in radians): +1 is north-facing, −1 is south-facing. Wraps the
    // circular aspect into a linear value MaxEnt can use directly.
    image: (ee) => ee.Terrain.aspect(ee.Image(SRTM).select('elevation'))
      .multiply(Math.PI / 180)
      .cos()
      .rename('northness'),
  },
  tpi: {
    label: 'Topographic position index (300 m)',
    // Deviation from a 300 m focal mean — positive is a ridge, negative is a
    // valley. Reprojected before the kernel so it does not blow up memory.
    image: (ee) => {
      const dem = ee.Image(SRTM).select('elevation')
      const smooth = dem.resample('bilinear').reproject({ crs: 'EPSG:3857', scale: 40 })
      return dem.subtract(smooth.reduceNeighborhood({
        reducer: ee.Reducer.mean(),
        kernel: ee.Kernel.circle(300, 'meters'),
      })).rename('tpi')
    },
  },
  twi: {
    label: 'Topographic wetness index',
    // Simplified TWI = ln(1 / tan(slope + ε)). Cheap proxy for drainage
    // accumulation at the pixel scale; the full upslope-area form needs a flow
    // model that does not fit a serverless timeout.
    image: (ee) => {
      const slope = ee.Terrain.slope(ee.Image(SRTM).select('elevation'))
      return ee.Image(1).divide(slope.add(0.001).tan()).log().rename('twi')
    },
  },

  // ── Vegetation & forest structure ─────────────────────────────────────────
  ndmi: {
    label: 'Vegetation moisture (NDMI)',
    image: (ee) => ee.ImageCollection(S2_SR)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
      .select(['B8', 'B11'])
      .median()
      .normalizedDifference(['B8', 'B11'])
      .rename('ndmi'),
  },
  canopy: {
    label: 'Tree canopy cover (NLCD)',
    image: (ee) => ee.ImageCollection(NLCD_TCC)
      .mosaic()
      .select('Science_Percent_Tree_Canopy_Cover')
      .rename('canopy'),
  },
  forest_loss: {
    label: 'Historical forest loss (Hansen)',
    // Binary: any Hansen-detected loss since 2000. A one-layer summary of
    // disturbance history without the date axis.
    image: (ee) => ee.Image(HANSEN_GFC).select('loss').rename('forest_loss'),
  },

  // ── Soil properties ───────────────────────────────────────────────────────
  clay_percent: {
    label: 'Clay content (SOLUS, 0 cm)',
    image: (ee) => ee.ImageCollection(SOLUS100)
      .filter(ee.Filter.eq('system:index', 'claytotal'))
      .first()
      .select('r_0_cm_p')
      .rename('clay_percent'),
  },
  soil_depth_cm: {
    label: 'Soil depth to bedrock (SOLUS)',
    image: (ee) => ee.ImageCollection(SOLUS100)
      .filter(ee.Filter.eq('system:index', 'anylithicdpt'))
      .first()
      .select('r_cm_p')
      .rename('soil_depth_cm'),
  },
  sand_percent: {
    label: 'Sand content (SOLUS, 0 cm)',
    image: (ee) => ee.ImageCollection(SOLUS100)
      .filter(ee.Filter.eq('system:index', 'sandtotal'))
      .first()
      .select('r_0_cm_p')
      .rename('sand_percent'),
  },

  // ── Terrain exposure (matches enrichment stage bands) ─────────────────────
  solar_exposure: {
    label: 'Solar exposure (heat load index)',
    // Folded aspect: McCune & Keon heat load index. North-facing slopes get low
    // values, south-facing slopes get high. Rescaled to [0, 1].
    image: (ee) => {
      const aspect = ee.Terrain.aspect(ee.Image(SRTM).select('elevation'))
      const slope = ee.Terrain.slope(ee.Image(SRTM).select('elevation'))
      // HLI = 1 − cos(aspect − 225°) × sin(slope) / 2
      const radAspect = aspect.subtract(225).multiply(Math.PI / 180)
      const radSlope = slope.multiply(Math.PI / 180)
      return ee.Image(1)
        .subtract(radAspect.cos().multiply(radSlope.sin()).divide(2))
        .rename('solar_exposure')
    },
  },
  wind_exposure: {
    label: 'Wind exposure (terrain roughness)',
    // Focal standard deviation of elevation as a simple proxy for wind exposure:
    // smooth ridges are exposed, sheltered draws have low variation. The
    // enrichment stage uses the same proxy.
    image: (ee) => {
      const dem = ee.Image(SRTM).select('elevation')
      return dem.reduceNeighborhood({
        reducer: ee.Reducer.stdDev(),
        kernel: ee.Kernel.circle(500, 'meters'),
      }).divide(200).min(1).rename('wind_exposure')
    },
  },

  // ── Fire history ──────────────────────────────────────────────────────────
  last_burn_year: {
    label: 'Last burn year (MODIS, since 2001)',
    // Most recent calendar year any pixel burned, aggregated from MODIS monthly
    // burn-date bands. Unburned pixels get 0. Useful as a habitat-age proxy.
    image: (ee) => {
      const years = []
      for (let y = 2001; y <= 2023; y++) years.push(y)
      const burnYear = ee.ImageCollection(years.map((y) => {
        const burned = ee.ImageCollection(MODIS_BURN)
          .filterDate(`${y}-01-01`, `${y}-12-31`)
          .select('BurnDate').max().gt(0)
        return burned.multiply(y).rename('last_burn_year')
      })).max()
      return burnYear.rename('last_burn_year')
    },
  },

  // ── Forest structure ──────────────────────────────────────────────────────
  stand_height_ft: {
    label: 'Stand height (USFS TreeMap, ft)',
    image: (ee) => ee.ImageCollection(TREEMAP)
      .filterDate('2016-01-01', '2016-12-31')
      .first()
      .select('STANDHT')
      .rename('stand_height_ft'),
  },
  canopy_pct: {
    label: 'Canopy cover % (USFS TreeMap)',
    image: (ee) => ee.ImageCollection(TREEMAP)
      .filterDate('2016-01-01', '2016-12-31')
      .first()
      .select('CANOPYPCT')
      .rename('canopy_pct'),
  },

  // ── Topographic indices (CSP/ERGo) ────────────────────────────────────────
  mtpi: {
    label: 'Multi-scale topographic position (mTPI)',
    // CSP/ERGo SRTM mTPI: positive = ridge, negative = valley. Better than a
    // single-radius TPI because it integrates multiple neighbourhood scales.
    image: (ee) => ee.Image(CSP_SRTM_MTPI).select('constant').rename('mtpi'),
  },
  chili: {
    label: 'Heat-insolation load (CHILI)',
    // Continuous Heat-Insolation Load Index: south + west-facing slopes score
    // high, north-facing valleys score low. Tighter than the HLI proxy.
    image: (ee) => ee.Image(CSP_SRTM_CHILI).select('constant').rename('chili'),
  },

  // ── Hydrology (MERIT) ─────────────────────────────────────────────────────
  hand: {
    label: 'Height above nearest drainage (HAND)',
    // MERIT Hydro hnd band: metres above the nearest river channel. Low values
    // mean riparian / flood-prone ground; high values mean dry upland.
    image: (ee) => ee.Image(MERIT_HYDRO).select('hnd')
      .updateMask(ee.Image(MERIT_HYDRO).select('hnd').gte(0))
      .rename('hand'),
  },

  // ── Soil hydraulics ───────────────────────────────────────────────────────
  field_capacity: {
    label: 'Soil field capacity (water-holding, 0 cm)',
    // OpenLandMap volumetric water content at 33 kPa tension, surface layer.
    // High = clay-rich soils that hold moisture; low = sandy, freely draining.
    image: (ee) => ee.Image(OPENLANDMAP_WATER_33KPA).select('b0').rename('field_capacity'),
  },

  // ── Climate normals (ERA5) ────────────────────────────────────────────────
  soil_temp_normal: {
    label: 'Soil temperature, normal (ERA5-Land)',
    // Long-run mean of ERA5-Land layer-1 soil temperature, converted to °C.
    image: (ee) => ee.ImageCollection(ERA5_DAILY)
      .select('soil_temperature_level_1')
      .mean()
      .subtract(273.15)
      .rename('soil_temp_normal'),
  },

  // ── Anthropogenic ─────────────────────────────────────────────────────────
  human_modification: {
    label: 'Human modification index',
    // CSP Global Human Modification: 0 = pristine, 1 = heavily modified.
    // Useful for separating habitat quality from raw environmental suitability,
    // and for capturing observation-effort bias correction.
    image: (ee) => {
      const img = ee.Image(CSP_HM).select('gHM')
      return img.updateMask(img.gte(0)).rename('human_modification')
    },
  },
  population_density: {
    label: 'Population density, log (GHSL 2020)',
    // log1p of GHSL population count; zero stays zero, dense cities compress.
    // Proxy for observation effort — high-density areas are over-sampled.
    image: (ee) => {
      const pop = ee.ImageCollection(GHSL_POP)
        .filterDate('2020-01-01', '2021-01-01').first()
        .select('population_count')
      return pop.log1p().updateMask(pop.gt(0)).rename('population_density')
    },
  },
}

export const PREDICTOR_KEYS = Object.keys(MAXENT_PREDICTORS)

/** The set switched on by default: terrain plus the two cheap standing indices. */
export const DEFAULT_PREDICTORS = ['elevation', 'slope', 'aspect', 'ndvi', 'soil_moisture']

/**
 * All predictors offered for auto-optimization: the full covariate suite from
 * which the scout model picks the most useful subset. Ordered so that cheaper,
 * widely-available layers come first — if the scout is short on time, the early
 * ones are more likely to stay in.
 */
export const ALL_PREDICTORS = [
  // Terrain
  'elevation', 'slope', 'aspect', 'northness', 'tpi', 'twi',
  'mtpi', 'chili', 'solar_exposure', 'wind_exposure',
  // Hydrology
  'hand',
  // Vegetation
  'ndvi', 'ndmi', 'canopy', 'canopy_pct', 'stand_height_ft',
  // Climate normals
  'annual_precip', 'annual_temp', 'precip_normal', 'temp_normal', 'soil_temp_normal',
  // Soil
  'soil_moisture', 'field_capacity', 'sand_percent', 'clay_percent', 'soil_depth_cm',
  // Disturbance
  'forest_loss', 'last_burn_year',
  // Anthropogenic
  'human_modification', 'population_density',
]

/** Minimum % contribution a predictor must contribute to survive the scout filter. */
export const DEFAULT_CONTRIBUTION_THRESHOLD = 2.0

// A model needs enough of both to mean anything: too few presences and it fits
// noise, too few background points and it has nothing to contrast them with.
export const MIN_PRESENCES = 20
export const MIN_BACKGROUND = 100
export const MAX_BACKGROUND = 10_000
export const DEFAULT_BACKGROUND = 1000
export const MIN_PREDICTORS = 2

export const SUITABILITY_PALETTE = ['#2c2f6b', '#3f7fb2', '#8fc0a9', '#dbe77a', '#f0a23f', '#c6301f']

/** The legend a suitability layer draws: a probability from 0 to 1. */
export function suitabilityLegend() {
  return {
    type: 'ramp', unit: 'suitability', min: '0', max: '1', stops: SUITABILITY_PALETTE,
  }
}

/**
 * A model request, turned into something safe to run.
 *
 * Mirrors normaliseSpec in ee-pipeline: every field is checked against a fixed
 * set or clamped to a range, and the message on failure is written for the
 * person who submitted it. The source is validated the same way an enrichment
 * job's is — that is where the presence points come from — so this only adds the
 * predictors, the background count, and the region to project onto.
 */
export function normaliseModelSpec(input = {}, { normaliseSource } = {}) {
  const requested = Array.isArray(input.predictors) && input.predictors.length
    ? input.predictors.map(String)
    : DEFAULT_PREDICTORS
  const unknown = requested.filter((p) => !MAXENT_PREDICTORS[p])
  if (unknown.length) {
    throw new SpecError(`Unknown predictor${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}.`)
  }
  // Registry order, deduplicated, so two specs asking for the same predictors
  // are the same spec.
  const predictors = PREDICTOR_KEYS.filter((k) => requested.includes(k))
  if (predictors.length < MIN_PREDICTORS) {
    throw new SpecError(`A model needs at least ${MIN_PREDICTORS} predictors.`)
  }

  const rawBackground = input.background === undefined ? DEFAULT_BACKGROUND : Number(input.background)
  if (!Number.isFinite(rawBackground)) throw new SpecError('The background count must be a number.')
  const background = Math.min(MAX_BACKGROUND, Math.max(MIN_BACKGROUND, Math.floor(rawBackground)))

  // Where the fitted surface is drawn. Defaults to the source's own area when it
  // has one (a bbox job), so a model over a region projects onto that region.
  const region = input.region ? normaliseBounds(input.region) : null

  const source = normaliseSource ? normaliseSource(input.source) : (input.source || null)

  const title = String(input.title || '').trim().slice(0, 120)

  // Two-pass auto-optimization: train a scout model over the full requested
  // predictor set, evaluate variable contributions, then re-train a production
  // model using only predictors above the contribution threshold.
  const autoOptimize = Boolean(input.autoOptimize)
  const rawThreshold = input.contributionThreshold === undefined
    ? DEFAULT_CONTRIBUTION_THRESHOLD : Number(input.contributionThreshold)
  const contributionThreshold = Number.isFinite(rawThreshold) && rawThreshold >= 0
    ? rawThreshold : DEFAULT_CONTRIBUTION_THRESHOLD

  return { kind: 'model', predictors, background, region, source, title, autoOptimize, contributionThreshold }
}

/**
 * How many background points to draw, and whether to weight them.
 *
 * Presence-only modelling inherits the observer-effort bias the app's caveats
 * already name: the background has to represent where recording happened, not
 * just where the species could be, or the model learns the survey and calls it
 * the species. So the honest default weights background sampling toward the
 * presences (target-group background), and the count is never allowed below the
 * presence count — a background smaller than the presences cannot describe the
 * range they sit in.
 */
export function backgroundPlan({ presenceCount = 0, background = DEFAULT_BACKGROUND, effortWeighted = true } = {}) {
  const requested = Math.min(MAX_BACKGROUND, Math.max(MIN_BACKGROUND, Math.floor(background) || DEFAULT_BACKGROUND))
  const n = Math.min(MAX_BACKGROUND, Math.max(requested, presenceCount))
  return {
    n,
    weighted: Boolean(effortWeighted),
    enough: presenceCount >= MIN_PRESENCES,
  }
}

/**
 * A stable cache key for a model's fitted output.
 *
 * Fitting a model — sampling, training, cross-validating, projecting — is the
 * expensive part of a model job, and it is identical for everyone asking for the
 * same predictors over the same region from the same source. So two members
 * modelling the same thing should share one mint rather than spend it twice
 * (V12-PERF-2). The key is everything the fit depends on, rounded and sorted so
 * two equivalent requests produce one key; the presences themselves are pinned
 * by the source, which is part of it.
 */
export function modelCacheKey(spec = {}, region = null) {
  const predictors = [...(spec.predictors || DEFAULT_PREDICTORS)].sort().join(',')
  const background = backgroundPlan({ background: spec.background || DEFAULT_BACKGROUND }).n
  const src = spec.source || {}
  const source = src.type === 'dataset'
    ? `dataset:${src.slug}`
    : `bbox:${src.taxon || ''}:${src.dateFrom || ''}:${src.dateTo || ''}`
  const r = region || spec.region
  const box = r
    ? ['north', 'south', 'east', 'west'].map((k) => Number(r[k]).toFixed(4)).join(',')
    : 'auto'
  return `model|${predictors}|bg=${background}|${source}|region=${box}`
}

/**
 * Roughly what a model job costs, in the same "one Earth Engine request" units a
 * quota is measured in.
 *
 * The work is: sample the predictors at the presences and at the background
 * points (one reduceRegions per chunk of each), then train and classify (a
 * handful of evaluations that do not scale with points). Priced as a slight
 * over-count, like estimateUnits, because refusing a job that would just have fit
 * is a smaller harm than admitting one that blows the month.
 */
export function estimateModelUnits({ points = 0, predictors = DEFAULT_PREDICTORS, background = DEFAULT_BACKGROUND } = {}) {
  const { n } = backgroundPlan({ presenceCount: points, background })
  const sampleChunks = Math.ceil(Math.max(1, points) / CHUNK_SIZE) + Math.ceil(n / CHUNK_SIZE)
  const stackDepth = Math.max(MIN_PREDICTORS, predictors.length || DEFAULT_PREDICTORS.length)
  // Sampling both point sets across the predictor stack, plus a fixed handful
  // for the fit and the region-wide classification.
  let total = sampleChunks * stackDepth + 4
  // Cross-validation samples the points a second time and trains once per fold,
  // so it roughly repeats the sampling cost. Only counted when there are enough
  // presences to run it (see MIN_CV_PRESENCES), which is the same gate runModel
  // applies before spending anything on it.
  if (points >= MIN_CV_PRESENCES) total += sampleChunks * stackDepth + CV_FOLDS
  return total
}

/**
 * How a model job's progress bar is divided.
 *
 * Four visible phases, weighted by where the time actually goes: sampling the
 * background dominates because it is the most points, the fit and the projection
 * are a step each. A bar that names the phase beats one that only moves.
 */
export function modelPlan() {
  const steps = [
    { key: 'presences', label: 'Sampling the observations', weight: 1 },
    { key: 'background', label: 'Sampling the background', weight: 2 },
    { key: 'fit', label: 'Fitting the model', weight: 1 },
    { key: 'validate', label: 'Cross-validating', weight: 2 },
    { key: 'project', label: 'Projecting suitability', weight: 2 },
  ]
  const total = steps.reduce((a, s) => a + s.weight, 0)
  let done = 0
  return steps.map((s) => {
    const from = done / total
    done += s.weight
    return { key: s.key, label: s.label, from, to: done / total, weight: s.weight }
  })
}

// ── Cross-validation ─────────────────────────────────────────────────────────
//
// A score for how much to trust the surface. The one trap here is spatial
// autocorrelation: nearby points are alike, so a random train/test split leaves
// a test point beside a training point and the model looks better than it is.
// Blocking by location — hold out whole squares of ground, not scattered points
// — is what makes the number honest, which is why the folds are assigned by
// block below rather than at random.

export const CV_FOLDS = 4
export const CV_BLOCK_DEGREES = 0.25
// Below this, the folds are too small for the AUC to mean anything, so the score
// is skipped rather than reported with false precision.
export const MIN_CV_PRESENCES = 40

/** A small deterministic PRNG (mulberry32), so a seed reproduces a run. */
function seeded(seed) {
  let a = (seed >>> 0) || 1
  return () => {
    a |= 0
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Background points drawn uniformly across the region, seeded.
 *
 * Generated here rather than by ee.FeatureCollection.randomPoints so their
 * coordinates are known on this side — which is what lets a background point be
 * assigned to a spatial fold alongside the presences. Uniform in lon/lat is not
 * area-correct near the poles, but a job's region is a small box, so the
 * distortion across it is negligible.
 */
export function randomBackground(region, n, seed = 1) {
  const rand = seeded(seed)
  const out = []
  const lonSpan = region.east - region.west
  const latSpan = region.north - region.south
  for (let i = 0; i < n; i += 1) {
    out.push([region.west + rand() * lonSpan, region.south + rand() * latSpan])
  }
  return out
}

/**
 * The block a point falls in, and from it a fold.
 *
 * The block is the square of `blockDegrees` it sits in; the fold is a stable hash
 * of that block's integer coordinates, so every point in one square lands in the
 * same fold and whole squares are held out together. The seed shuffles which
 * square goes to which fold without moving points between squares.
 */
export function assignFold(lon, lat, { blockDegrees = CV_BLOCK_DEGREES, folds = CV_FOLDS, seed = 1 } = {}) {
  const bx = Math.floor(lon / blockDegrees)
  const by = Math.floor(lat / blockDegrees)
  // A cheap integer hash of (bx, by, seed), made positive, then folded.
  let h = (bx * 73856093) ^ (by * 19349663) ^ (seed * 83492791)
  h = (h ^ (h >>> 13)) >>> 0
  return h % folds
}

/**
 * Presence and background points combined and tagged with a spatial fold.
 *
 * `presences` and `background` are `[lon, lat]` arrays; the result is one list of
 * `{ lon, lat, presence, fold }`, ready to become an Earth Engine FeatureCollection.
 */
export function foldPoints(presences, background, { folds = CV_FOLDS, blockDegrees = CV_BLOCK_DEGREES, seed = 1 } = {}) {
  const tag = (presence) => ([lon, lat]) => ({
    lon, lat, presence, fold: assignFold(lon, lat, { blockDegrees, folds, seed }),
  })
  return [...presences.map(tag(1)), ...background.map(tag(0))]
}

/**
 * Area under the ROC curve, from predicted scores and 0/1 labels.
 *
 * The rank form of the Mann–Whitney U statistic: the probability that a random
 * presence scores above a random background point. 0.5 is a coin toss, 1.0 is
 * perfect separation. Ties share the average rank so a flat predictor scores 0.5
 * rather than something spuriously off it. Pure, so it is tested against known
 * cases without a model.
 */
export function rocAuc(scores, labels) {
  const n = scores.length
  const pos = labels.reduce((a, l) => a + (l ? 1 : 0), 0)
  const neg = n - pos
  if (!pos || !neg) return null

  // Rank the scores, averaging ties.
  const order = scores.map((s, i) => ({ s, l: labels[i] })).sort((a, b) => a.s - b.s)
  const ranks = new Array(n)
  let i = 0
  while (i < n) {
    let j = i
    while (j + 1 < n && order[j + 1].s === order[i].s) j += 1
    const avg = (i + j) / 2 + 1 // 1-based average rank across the tie group
    for (let k = i; k <= j; k += 1) ranks[k] = avg
    i = j + 1
  }
  let rankSumPos = 0
  for (let k = 0; k < n; k += 1) if (order[k].l) rankSumPos += ranks[k]
  return (rankSumPos - (pos * (pos + 1)) / 2) / (pos * neg)
}

/**
 * Roll the per-fold held-out predictions up into one score.
 *
 * `rows` is `[{ fold, presence, prob }]` — the held-out predictions from every
 * fold. Each fold's AUC is computed on its own held-out block, and the reported
 * score is their mean with the spread beside it, because a good mean over folds
 * that disagree wildly is not the same as a good mean over folds that agree.
 */
export function crossValidationSummary(rows, { folds = CV_FOLDS } = {}) {
  const perFold = []
  for (let f = 0; f < folds; f += 1) {
    const inFold = rows.filter((r) => r.fold === f)
    if (!inFold.length) continue
    const auc = rocAuc(inFold.map((r) => r.prob), inFold.map((r) => r.presence))
    if (auc !== null) perFold.push(auc)
  }
  if (!perFold.length) return null
  const mean = perFold.reduce((a, b) => a + b, 0) / perFold.length
  const variance = perFold.reduce((a, b) => a + (b - mean) ** 2, 0) / perFold.length
  return {
    auc: Number(mean.toFixed(3)),
    sd: Number(Math.sqrt(variance).toFixed(3)),
    folds: perFold.length,
    blockDegrees: CV_BLOCK_DEGREES,
    // A plain reading of the number, so a member does not have to know what AUC
    // is to know whether to trust the map.
    grade: mean >= 0.9 ? 'excellent' : mean >= 0.8 ? 'good' : mean >= 0.7 ? 'fair' : 'weak',
  }
}

/**
 * Train and score the model across spatial folds, in Earth Engine.
 *
 * `points` is the folded `{ lon, lat, presence, fold }` list. Each fold is held
 * out in turn: the model is trained on the others and used to score the held-out
 * block, and the scored held-out features are merged into one collection carrying
 * `fold`, `presence` and `prob`. The AUC itself is computed in JavaScript from
 * the evaluated result (see crossValidationSummary) — Earth Engine trains and
 * predicts; the arithmetic that judges it stays testable here.
 */
export function crossValidate(ee, { stack, points, predictors, folds = CV_FOLDS }) {
  const features = points.map((p) => ee.Feature(
    ee.Geometry.Point([p.lon, p.lat]), { presence: p.presence, fold: p.fold },
  ))
  const withBands = stack.sampleRegions({
    collection: ee.FeatureCollection(features), properties: ['presence', 'fold'], scale: 100, geometries: false,
  })

  let heldOut = null
  for (let f = 0; f < folds; f += 1) {
    const train = withBands.filter(ee.Filter.neq('fold', f))
    const test = withBands.filter(ee.Filter.eq('fold', f))
    const classifier = ee.Classifier.amnhMaxent().train({
      features: train, classProperty: 'presence', inputProperties: predictors,
    })
    // classifyProbability-style output under the name 'prob'.
    const scored = test.classify(classifier, 'prob')
    heldOut = heldOut ? heldOut.merge(scored) : scored
  }
  // Only the three columns the summary reads, so the evaluated payload is small.
  return heldOut.select(['fold', 'presence', 'prob'], null, false)
}

/**
 * The predictor stack as one multi-band Earth Engine image.
 *
 * Keys are validated by normaliseModelSpec before they reach here, so this trusts
 * them; it is a small, mechanical assembly kept out of buildSuitabilityImage so
 * the band order is one obvious list.
 */
export function predictorStack(ee, predictors) {
  const images = predictors.map((key) => MAXENT_PREDICTORS[key].image(ee).rename(key))
  // cat rather than addBands from an empty image, so the band names are exactly
  // the predictor keys and nothing carries a stray constant band.
  return ee.Image.cat(images)
}

/**
 * Fit a MaxEnt model and project it as a suitability image.
 *
 * `presences` is an Earth Engine FeatureCollection of the observation points
 * (the job's own source). Background points are drawn at random across the
 * region, the predictors are sampled at both, amnhMaxent is trained on the
 * contrast, and the fitted classifier paints the whole predictor stack as a
 * probability from 0 to 1.
 *
 * Takes `ee` as an argument, so it runs against the real client in the worker
 * and against a stub in a test.
 */
export function buildSuitabilityImage(ee, {
  presences, predictors, background = DEFAULT_BACKGROUND, region, seed = 1,
}) {
  const stack = predictorStack(ee, predictors)
  const geometry = ee.Geometry.Rectangle([region.west, region.south, region.east, region.north])

  const presenceSamples = stack
    .sampleRegions({ collection: presences, scale: 100, geometries: false })
    .map((f) => f.set('presence', 1))

  const backgroundPoints = ee.FeatureCollection.randomPoints({
    region: geometry, points: background, seed,
  })
  const backgroundSamples = stack
    .sampleRegions({ collection: backgroundPoints, scale: 100, geometries: false })
    .map((f) => f.set('presence', 0))

  const training = presenceSamples.merge(backgroundSamples)

  // amnhMaxent outputs the probability of presence, which is exactly the
  // suitability the map wants — no post-scaling.
  const classifier = ee.Classifier.amnhMaxent().train({
    features: training,
    classProperty: 'presence',
    inputProperties: predictors,
  })

  const suitability = stack.classify(classifier).clip(geometry).rename('suitability')

  return {
    image: suitability,
    vis: { min: 0, max: 1, palette: SUITABILITY_PALETTE },
  }
}
