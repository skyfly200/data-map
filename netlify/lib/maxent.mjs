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

import { SRTM, S2_SR, ERA5_DAILY, SpecError, normaliseBounds } from './ee-pipeline.mjs'

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
}

export const PREDICTOR_KEYS = Object.keys(MAXENT_PREDICTORS)

/** The set switched on by default: terrain plus the two cheap standing indices. */
export const DEFAULT_PREDICTORS = ['elevation', 'slope', 'aspect', 'ndvi', 'soil_moisture']

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

  return { kind: 'model', predictors, background, region, source, title }
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
