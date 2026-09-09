// The Earth Engine pipeline, as something a member can ask for.
//
// scripts/ee_enrich.py already samples every environmental column from Earth
// Engine, but it runs offline: someone with credentials runs it, and the result
// is committed. This is the same work reached through the app, so a member can
// enrich their own points for a foray without a Python environment or a key.
//
// The single most important rule here is that members never send Earth Engine
// code. A spec is values — an area, some dates, a list of stage names checked
// against the catalogue below — and this module turns those into EE calls.
// Accepting an expression instead would be arbitrary compute on the society's
// billing account and a code-injection surface in the same stroke.
//
// The sampling pattern is the one the Python pipeline proved: group the points
// that share a query, build ONE image carrying every band that group needs, and
// issue ONE reduceRegions per chunk. A seven-day weather history is therefore
// one round trip per observation date, not seven per observation.

import { CHUNK_SIZE } from './quotas.mjs'

// ─── Dataset ids, kept in step with scripts/ee_enrich.py ─────────────────────
export const SRTM = 'USGS/SRTMGL1_003'
export const WORLDCOVER = 'ESA/WorldCover/v200'
export const ERA5_DAILY = 'ECMWF/ERA5_LAND/DAILY_AGGR'
export const CHIRPS_DAILY = 'UCSB-CHG/CHIRPS/DAILY'
export const S2_SR = 'COPERNICUS/S2_SR_HARMONIZED'

/**
 * The allowlist. A stage not named here cannot be run, cannot be billed, and
 * cannot name an Earth Engine asset — which is the point: the set of assets the
 * service account will ever touch is this table, and it is not open to input.
 *
 *   passes   images the stage samples per query group, which is what makes a
 *            seven-day history cost seven times a single-day one.
 *   perDate  whether the image moves with the observation's date. Static layers
 *            like elevation are sampled once for every point at any date.
 */
export const STAGES = {
  terrain: {
    label: 'Terrain',
    asset: SRTM,
    scale: 30,
    passes: 1,
    perDate: false,
    bands: ['elevation', 'slope', 'aspect', 'tpi_150m', 'tpi_500m', 'tpi_1500m',
            'solar_exposure', 'wind_exposure', 'water_retention'],
    description: 'Elevation, slope, aspect and the exposure indices derived from them.',
  },
  landcover: {
    label: 'Land cover',
    asset: WORLDCOVER,
    scale: 10,
    passes: 1,
    perDate: false,
    bands: ['land_cover', 'water_mask'],
    description: 'ESA WorldCover class at each point, at 10m.',
  },
  soil_moisture: {
    label: 'Soil moisture',
    asset: ERA5_DAILY,
    scale: 11132,
    passes: 1,
    perDate: true,
    bands: ['soil_moisture'],
    description: 'ERA5-Land volumetric soil water on the day of the record.',
  },
  precip: {
    label: 'Rainfall lead-up',
    asset: CHIRPS_DAILY,
    scale: 5566,
    passes: 7,
    perDate: true,
    bands: Array.from({ length: 7 }, (_, i) => `prcp_d${i}`),
    description: 'CHIRPS daily rainfall for the seven days up to each record.',
  },
  temperature: {
    label: 'Temperature lead-up',
    asset: ERA5_DAILY,
    scale: 11132,
    passes: 7,
    perDate: true,
    bands: [...Array.from({ length: 7 }, (_, i) => `tmax_d${i}`),
            ...Array.from({ length: 7 }, (_, i) => `tmin_d${i}`)],
    description: 'ERA5-Land daily maximum and minimum for the seven days up to each record.',
  },
  ndvi: {
    label: 'Vegetation',
    asset: S2_SR,
    scale: 10,
    passes: 1,
    perDate: true,
    bands: ['ndvi', 'ndmi'],
    description: 'Sentinel-2 vegetation and moisture indices, cloud-screened.',
  },
}

export const STAGE_KEYS = Object.keys(STAGES)

/** The default set: everything the shipped dataset carries. */
export const DEFAULT_STAGES = ['terrain', 'landcover', 'soil_moisture', 'precip', 'temperature', 'ndvi']

// Bounds on what one job may ask for. These are not the per-member quota (that
// is in quotas.mjs); they are the outer edge of what the pipeline can do at all
// without an export task, and they apply to admins too.
export const MAX_BBOX_DEGREES = 20      // a job is a region, not a hemisphere
export const MAX_POINTS = 50000
export const MAX_DAYS = 366 * 5

export class SpecError extends Error {}

function num(value, name) {
  const n = Number(value)
  if (!Number.isFinite(n)) throw new SpecError(`${name} must be a number.`)
  return n
}

function isoDate(value, name) {
  if (value === null || value === undefined || value === '') return null
  const s = String(value).slice(0, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) throw new SpecError(`${name} must be a date like 2026-09-01.`)
  const d = new Date(`${s}T00:00:00Z`)
  if (!Number.isFinite(d.getTime())) throw new SpecError(`${name} is not a real date.`)
  return s
}

/** A bounding box, validated and ordered. */
export function normaliseBounds(input) {
  if (!input || typeof input !== 'object') throw new SpecError('An area is required.')
  const north = num(input.north, 'north')
  const south = num(input.south, 'south')
  let east = num(input.east, 'east')
  let west = num(input.west, 'west')

  if (north <= south) throw new SpecError('The area’s north edge must be above its south edge.')
  for (const [v, n] of [[north, 'north'], [south, 'south']]) {
    if (v < -90 || v > 90) throw new SpecError(`${n} must be between -90 and 90.`)
  }
  for (const [v, n] of [[east, 'east'], [west, 'west']]) {
    if (v < -180 || v > 180) throw new SpecError(`${n} must be between -180 and 180.`)
  }
  // A box dragged across the antimeridian arrives with west > east. Earth
  // Engine takes it either way round, but the area check below would read it as
  // nearly the whole globe and refuse a perfectly small box.
  if (west > east) east += 360

  if (north - south > MAX_BBOX_DEGREES || east - west > MAX_BBOX_DEGREES) {
    throw new SpecError(`That area is larger than ${MAX_BBOX_DEGREES} degrees across. `
      + 'Earth Engine can do it, but not inside a request; narrow the area.')
  }
  return { north, south, east, west }
}

/**
 * A member's request, turned into something safe to run.
 *
 * Throws SpecError with a message meant for the person who submitted it. Every
 * field is either checked against a fixed set or clamped to a range; nothing
 * from the caller reaches Earth Engine as an identifier.
 */
export function normaliseSpec(input = {}) {
  const kind = String(input.kind || 'enrich')
  if (kind !== 'enrich') throw new SpecError(`Unknown job kind “${kind}”.`)

  const requested = Array.isArray(input.stages) && input.stages.length
    ? input.stages.map(String)
    : DEFAULT_STAGES
  const unknown = requested.filter((s) => !STAGES[s])
  if (unknown.length) throw new SpecError(`Unknown stage${unknown.length > 1 ? 's' : ''}: ${unknown.join(', ')}.`)
  // Catalogue order, deduplicated, so two specs asking for the same work are
  // the same spec and the cost estimate does not depend on typing order.
  const stages = STAGE_KEYS.filter((k) => requested.includes(k))

  const source = input.source && typeof input.source === 'object' ? input.source : {}
  const type = String(source.type || 'bbox')
  let normalisedSource

  if (type === 'dataset') {
    const slug = String(source.slug || '').trim()
    // Slugs address a row the caller must already be allowed to read; the
    // pattern keeps anything path-shaped out of a storage key.
    if (!/^[a-z0-9][a-z0-9_-]{0,80}$/i.test(slug)) throw new SpecError('That dataset name is not valid.')
    normalisedSource = { type: 'dataset', slug }
  } else if (type === 'bbox') {
    const bounds = normaliseBounds(source.bounds)
    const from = isoDate(source.dateFrom, 'dateFrom')
    const to = isoDate(source.dateTo, 'dateTo')
    if (from && to && from > to) throw new SpecError('The start date is after the end date.')
    if (from && to) {
      const days = (new Date(`${to}T00:00:00Z`) - new Date(`${from}T00:00:00Z`)) / 86400000
      if (days > MAX_DAYS) throw new SpecError('That date range is longer than five years.')
    }
    const taxon = source.taxon === undefined || source.taxon === null ? '' : String(source.taxon).trim()
    if (taxon.length > 120) throw new SpecError('That taxon name is too long.')
    normalisedSource = { type: 'bbox', bounds, dateFrom: from, dateTo: to, taxon }
  } else {
    throw new SpecError(`Unknown source type “${type}”.`)
  }

  const title = String(input.title || '').trim().slice(0, 120)

  return { kind, stages, source: normalisedSource, title }
}

/** Bands a spec will produce, in catalogue order. */
export function bandsFor(stages = []) {
  return stages.flatMap((key) => STAGES[key]?.bands || [])
}

/**
 * How the progress bar is divided.
 *
 * By pass count rather than by stage count, so a seven-day rainfall stage takes
 * seven times as much of the bar as elevation does. A bar that sits at 20% for
 * four minutes and then jumps is worse than no bar.
 */
export function progressPlan(stages = [], { points = 0, dates = 1 } = {}) {
  const chunks = Math.max(1, Math.ceil(points / CHUNK_SIZE))
  const weights = stages.map((key) => {
    const stage = STAGES[key]
    if (!stage) return 0
    const groups = stage.perDate ? Math.max(chunks, Math.max(1, dates)) : chunks
    return groups * (stage.passes || 1)
  })
  const total = weights.reduce((a, b) => a + b, 0) || 1
  let done = 0
  return stages.map((key, i) => {
    const from = done / total
    done += weights[i]
    return { key, label: STAGES[key]?.label || key, from, to: done / total, weight: weights[i] }
  })
}
