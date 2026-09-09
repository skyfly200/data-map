// Talking to Earth Engine.
//
// Everything that needs a live Earth Engine session lives here, behind a small
// interface, so the rest of the pipeline — the spec allowlist, the cost model,
// the derived indices, the queue — is pure and testable without credentials.
//
// Authentication is a Google service account, not a person: the society holds
// one Earth Engine project and members reach it through the app. That is also
// why quotas exist (see quotas.mjs) — Earth Engine bills the project, so every
// member's job spends from one shared pool.
//
// Set up:
//   1. Create a service account in the Google Cloud project registered with
//      Earth Engine, and register it at signup.earthengine.google.com.
//   2. Put its JSON key in EARTHENGINE_SERVICE_ACCOUNT_KEY (the whole JSON, or
//      base64 of it — Netlify env vars dislike newlines, so both are accepted).
//   3. Set EARTHENGINE_PROJECT to the Cloud project id.
//
// Without those this module reports itself unconfigured and the queue refuses
// jobs with a message saying so, rather than failing halfway through one.

import ee from '@google/earthengine'

import { CHUNK_SIZE } from './quotas.mjs'
import { CHIRPS_DAILY, ERA5_DAILY, S2_SR, SRTM, STAGES, WORLDCOVER } from './ee-pipeline.mjs'
import { derivedIndices } from './terrain-indices.mjs'

const MERIT_HYDRO = 'MERIT/Hydro/v1_0_1'

export function earthEngineConfigured() {
  return Boolean(process.env.EARTHENGINE_SERVICE_ACCOUNT_KEY && process.env.EARTHENGINE_PROJECT)
}

function serviceAccountKey() {
  const raw = process.env.EARTHENGINE_SERVICE_ACCOUNT_KEY || ''
  const text = raw.trim().startsWith('{')
    ? raw
    // Netlify environment variables mangle embedded newlines, so a base64 copy
    // of the key file is the form that survives being pasted into a dashboard.
    : Buffer.from(raw, 'base64').toString('utf8')
  try {
    return JSON.parse(text)
  } catch {
    throw new Error('EARTHENGINE_SERVICE_ACCOUNT_KEY is not valid JSON (or base64 of it).')
  }
}

let session = null

/** Authenticate and initialise once per process. */
export async function initEarthEngine() {
  if (session) return session
  if (!earthEngineConfigured()) throw new Error('Earth Engine is not configured on this deployment.')
  const key = serviceAccountKey()

  session = new Promise((resolve, reject) => {
    ee.data.authenticateViaPrivateKey(key, () => {
      ee.initialize(
        null, null,
        () => resolve(ee),
        (err) => reject(new Error(`Earth Engine failed to initialise: ${err}`)),
        null,
        process.env.EARTHENGINE_PROJECT,
      )
    }, (err) => reject(new Error(`Earth Engine rejected the service account: ${err}`)))
  }).catch((err) => {
    // Do not cache a failure: a transient network problem should not poison the
    // process for every job that follows.
    session = null
    throw err
  })
  return session
}

/** getInfo as a promise, with a deadline so one call cannot hang a worker. */
export function evaluate(obj, { timeoutMs = 120000 } = {}) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Earth Engine request timed out.')), timeoutMs)
    obj.evaluate((value, err) => {
      clearTimeout(timer)
      if (err) reject(new Error(String(err)))
      else resolve(value)
    })
  })
}

/**
 * Retry the transient failures.
 *
 * Earth Engine answers a request it cannot serve right now with a plain error,
 * and under concurrency that is common enough that not retrying loses jobs. A
 * rejected asset or a bad band is permanent, so only the shapes worth retrying
 * are retried.
 */
async function withRetry(fn, { retries = 3, baseDelay = 2000, label = '' } = {}) {
  let last
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      return await fn()
    } catch (err) {
      last = err
      const message = String(err?.message || err)
      const transient = /too many|rate limit|timed out|backend error|internal error|unavailable|deadline/i
        .test(message)
      if (!transient || attempt === retries) break
      await new Promise((r) => setTimeout(r, baseDelay * 2 ** attempt))
    }
  }
  throw new Error(label ? `${label}: ${last?.message || last}` : String(last?.message || last))
}

/** One day of an image collection, as a single named band. */
function dailyBand(collection, band, isoDay, outName) {
  const start = ee.Date(isoDay)
  // mean() over a one-day window yields an image even when the collection is
  // empty for that day, which comes back as a masked (null) sample rather than
  // an error that would fail the whole date group.
  return ee.Image(collection.filterDate(start, start.advance(1, 'day')).select(band).mean())
    .rename(outName)
}

function shiftDays(isoDay, delta) {
  const d = new Date(`${isoDay}T00:00:00Z`)
  d.setUTCDate(d.getUTCDate() + delta)
  return d.toISOString().slice(0, 10)
}

/** Sample one image at up to CHUNK_SIZE points. Returns rows aligned to input. */
async function sampleChunk(image, points, scale, reducer = ee.Reducer.first()) {
  const fc = ee.FeatureCollection(points.map((p, i) => ee.Feature(
    ee.Geometry.Point([p.lon, p.lat]), { __i: i },
  )))
  const sampled = image.reduceRegions({ collection: fc, reducer, scale })
  const info = await withRetry(() => evaluate(sampled), { label: 'sampling' })
  const out = new Array(points.length).fill(null)
  for (const feature of info?.features || []) {
    const props = feature.properties || {}
    const i = props.__i
    if (Number.isInteger(i) && i >= 0 && i < out.length) out[i] = props
  }
  return out
}

/** Points grouped into chunks of CHUNK_SIZE, keeping their original indices. */
function chunk(items, size = CHUNK_SIZE) {
  const out = []
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size))
  return out
}

// ─── The stages ──────────────────────────────────────────────────────────────
// Each writes its own columns into `columns` (a Map of band name → array) and
// reports fractional progress within its own slice of the bar.

async function runTerrain(points, columns, tick) {
  const dem = ee.Image(SRTM)
  const terrain = ee.Terrain.products(dem)
  // TPI at three radii, each reprojected so the circular kernel spans roughly
  // eight pixels: a 1.5km kernel over 30m SRTM would otherwise cover ~8,000
  // pixels per output cell and dominate the request's cost.
  const tpiBands = [150, 500, 1500].map((radius) => {
    const scale = Math.max(30, radius / 8)
    const d = dem.resample('bilinear').reproject({ crs: 'EPSG:3857', scale })
    return d.subtract(d.reduceNeighborhood({
      reducer: ee.Reducer.mean(),
      kernel: ee.Kernel.circle(radius, 'meters'),
    })).rename(`tpi_${radius}m`)
  })
  const upstream = ee.Image(MERIT_HYDRO).select('upa').rename('upstream_area')
  const image = ee.Image.cat([
    terrain.select(['elevation', 'slope', 'aspect']), ...tpiBands, upstream,
  ])

  const groups = chunk(points)
  for (let g = 0; g < groups.length; g += 1) {
    const rows = await sampleChunk(image, groups[g], STAGES.terrain.scale)
    rows.forEach((props, i) => {
      const at = groups[g][i].index
      for (const band of ['elevation', 'slope', 'aspect', 'tpi_150m', 'tpi_500m', 'tpi_1500m', 'upstream_area']) {
        columns.get(band)[at] = props?.[band] ?? null
      }
    })
    tick((g + 1) / groups.length)
  }

  // The three exposure indices are derived from the sampled columns rather than
  // sampled themselves; see terrain-indices.mjs for why they are point-relative.
  const derived = derivedIndices({
    slope: columns.get('slope'),
    aspect: columns.get('aspect'),
    lat: points.reduce((acc, p) => { acc[p.index] = p.lat; return acc }, []),
    tpi: columns.get('tpi_500m'),
    upstream: columns.get('upstream_area'),
  })
  for (const [band, values] of Object.entries(derived)) columns.set(band, values)
}

async function runLandcover(points, columns, tick) {
  const image = ee.Image(WORLDCOVER).select('Map').rename('land_cover')
  const groups = chunk(points)
  for (let g = 0; g < groups.length; g += 1) {
    const rows = await sampleChunk(image, groups[g], STAGES.landcover.scale, ee.Reducer.mode())
    rows.forEach((props, i) => {
      const at = groups[g][i].index
      const cls = props?.land_cover ?? null
      columns.get('land_cover')[at] = cls
      // Class 80 is permanent water. The map hides these unless asked, so the
      // flag is worth carrying rather than re-deriving everywhere.
      columns.get('water_mask')[at] = cls === null ? null : cls === 80
    })
    tick((g + 1) / groups.length)
  }
}

/** Shared shape for the stages that sample an image chosen by the record's date. */
async function runDated(points, columns, tick, { scale, reducer, bands, imageFor }) {
  const byDate = new Map()
  for (const p of points) {
    if (!p.date) continue
    if (!byDate.has(p.date)) byDate.set(p.date, [])
    byDate.get(p.date).push(p)
  }
  const dates = [...byDate.keys()].sort()
  let done = 0
  for (const date of dates) {
    const image = imageFor(date)
    for (const group of chunk(byDate.get(date))) {
      const rows = await sampleChunk(image, group, scale, reducer)
      rows.forEach((props, i) => {
        const at = group[i].index
        for (const band of bands) columns.get(band)[at] = props?.[band] ?? null
      })
    }
    done += 1
    tick(done / dates.length)
  }
}

const DAYS = 7

async function runSoilMoisture(points, columns, tick) {
  const era5 = ee.ImageCollection(ERA5_DAILY)
  await runDated(points, columns, tick, {
    scale: STAGES.soil_moisture.scale,
    reducer: ee.Reducer.mean(),
    bands: ['soil_moisture'],
    imageFor: (date) => dailyBand(era5, 'volumetric_soil_water_layer_1', date, 'soil_moisture'),
  })
}

async function runPrecip(points, columns, tick) {
  const chirps = ee.ImageCollection(CHIRPS_DAILY)
  await runDated(points, columns, tick, {
    scale: STAGES.precip.scale,
    reducer: ee.Reducer.mean(),
    bands: STAGES.precip.bands,
    // All seven days become bands of one image, so a date costs one request
    // rather than seven.
    imageFor: (date) => ee.Image.cat(
      Array.from({ length: DAYS }, (_, d) => dailyBand(chirps, 'precipitation', shiftDays(date, -d), `prcp_d${d}`)),
    ),
  })
}

async function runTemperature(points, columns, tick) {
  const era5 = ee.ImageCollection(ERA5_DAILY)
  await runDated(points, columns, tick, {
    scale: STAGES.temperature.scale,
    reducer: ee.Reducer.mean(),
    bands: STAGES.temperature.bands,
    imageFor: (date) => ee.Image.cat([
      ...Array.from({ length: DAYS }, (_, d) => dailyBand(era5, 'temperature_2m_max', shiftDays(date, -d), `tmax_d${d}`)),
      ...Array.from({ length: DAYS }, (_, d) => dailyBand(era5, 'temperature_2m_min', shiftDays(date, -d), `tmin_d${d}`)),
    ]),
  })
}

async function runNdvi(points, columns, tick) {
  const s2 = ee.ImageCollection(S2_SR)
  await runDated(points, columns, tick, {
    scale: STAGES.ndvi.scale,
    reducer: ee.Reducer.mean(),
    bands: ['ndvi', 'ndmi'],
    imageFor: (date) => {
      // A window either side of the date, because a single day is often cloudy
      // or simply has no overpass.
      const centre = ee.Date(date)
      const scene = s2
        .filterDate(centre.advance(-15, 'day'), centre.advance(15, 'day'))
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
        .median()
      return ee.Image.cat([
        scene.normalizedDifference(['B8', 'B4']).rename('ndvi'),
        scene.normalizedDifference(['B8', 'B11']).rename('ndmi'),
      ])
    },
  })
}

const RUNNERS = {
  terrain: runTerrain,
  landcover: runLandcover,
  soil_moisture: runSoilMoisture,
  precip: runPrecip,
  temperature: runTemperature,
  ndvi: runNdvi,
}

/**
 * Run a normalised spec over a set of features.
 *
 * `onProgress({ fraction, stage, message })` is called as each stage advances,
 * so the queue can write it to the job row and the member watching sees it move.
 * Returns the features with the sampled properties merged in.
 */
export async function runPipeline({ spec, features, plan, onProgress = () => {} }) {
  await initEarthEngine()

  const points = features.map((f, index) => {
    const co = f?.geometry?.coordinates || []
    return {
      index,
      lon: Number(co[0]),
      lat: Number(co[1]),
      date: (f?.properties?.date || '').slice(0, 10) || null,
    }
  }).filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lon))

  if (!points.length) throw new Error('That source has no usable coordinates.')

  const columns = new Map()
  for (const key of spec.stages) {
    for (const band of STAGES[key].bands) columns.set(band, new Array(features.length).fill(null))
  }
  // Sampled but not published: an input to the derived indices, not a column
  // the map offers.
  if (spec.stages.includes('terrain')) columns.set('upstream_area', new Array(features.length).fill(null))

  for (const step of plan) {
    const runner = RUNNERS[step.key]
    if (!runner) continue
    onProgress({ fraction: step.from, stage: step.key, message: `${step.label}…` })
    await runner(points, columns, (within) => {
      onProgress({
        fraction: step.from + (step.to - step.from) * Math.min(1, Math.max(0, within)),
        stage: step.key,
        message: `${step.label}…`,
      })
    })
  }

  columns.delete('upstream_area')
  const out = features.map((f, i) => {
    const properties = { ...(f.properties || {}) }
    for (const [band, values] of columns) properties[band] = values[i]
    return { ...f, properties }
  })

  onProgress({ fraction: 1, stage: 'done', message: 'Finished.' })
  return { features: out, bands: [...columns.keys()], sampled: points.length }
}
