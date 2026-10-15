// Daily weather at one place for the season outlook (WANT-19).
//
// Two kinds of read, each small enough for one function call:
//
//   year      A whole year of observed daily weather: ERA5-Land max/min
//             temperature and CHIRPS rain, the same sources the enrichment
//             pipeline samples (ee_enrich.py, ee-runner.mjs), so a trigger fitted
//             on past years reads the weather the same way it is forecast.
//   forecast  The latest WeatherNext 2 run: 64 ensemble members, 6-hourly to 15
//             days, turned into daily rain / max / min per member. Plus the
//             recent short-range forecasts (first 24 hours of each day's 00z
//             run), which fill the days ERA5 has not caught up on yet and give
//             the client something to measure WeatherNext's local bias against.
//
// The parsing below is pure and tested; the Earth Engine calls are thin.

import ee from '@google/earthengine'
import { CHIRPS_DAILY, ERA5_DAILY } from './ee-pipeline.mjs'
import { evaluate, initEarthEngine } from './ee-runner.mjs'

export const WEATHERNEXT = 'projects/gcp-public-data-weathernext/assets/weathernext_2_0_0'
/** Forecast days kept from a run. WeatherNext 2 runs to 15. */
export const FORECAST_DAYS = 15
/** Days of recent short-range forecasts returned for gap fill and bias. */
export const RECENT_DAYS = 30
/** Members used for the recent short-range read: its ensemble mean is all that is kept. */
export const RECENT_MEMBERS = 8
/** Cache grid, degrees. ERA5-Land is 0.1 degree, so finer buys nothing. */
export const CELL = 0.1

const DAY_MS = 86400000
const DAYS = 366
const K = 273.15

export class SeasonWeatherError extends Error {
  constructor(message, status = 400) { super(message); this.status = status }
}

/** Snap a point to the cache grid's cell centre. */
export function snapCell(lat, lon) {
  const la = Number(lat)
  const lo = Number(lon)
  if (!Number.isFinite(la) || !Number.isFinite(lo) || Math.abs(la) > 90 || Math.abs(lo) > 180) {
    throw new SeasonWeatherError('lat and lon must be numbers on the globe.')
  }
  const snap = (v) => Math.round((Math.floor(v / CELL) + 0.5) * CELL * 1000) / 1000
  return { lat: snap(la), lon: snap(lo) }
}

const round1 = (v) => (Number.isFinite(v) ? Math.round(v * 10) / 10 : null)

const emptyYear = () => ({ rain: Array(DAYS).fill(null), tmax: Array(DAYS).fill(null), tmin: Array(DAYS).fill(null) })

/** Day of year (1-based, UTC) of a millisecond time, and its year. */
function doyOf(ms) {
  const d = new Date(ms)
  const y = d.getUTCFullYear()
  return { year: y, doy: Math.floor((ms - Date.UTC(y, 0, 1)) / DAY_MS) + 1 }
}

/**
 * getRegion rows into one year's series.
 *
 * getRegion answers `[[id, longitude, latitude, time, ...bands], ...rows]`;
 * `map` names which band fills which field and how to convert it.
 */
export function seriesFromRegion(rows, year, map, into = emptyYear()) {
  if (!Array.isArray(rows) || rows.length < 2) return into
  const head = rows[0]
  const at = head.indexOf('time')
  for (const [band, field, convert] of map) {
    const col = head.indexOf(band)
    if (col < 0) continue
    for (const row of rows.slice(1)) {
      const { year: y, doy } = doyOf(Number(row[at]))
      const v = row[col]
      if (y !== year || v === null || v === undefined) continue
      into[field][doy - 1] = round1(convert(Number(v)))
    }
  }
  return into
}

/**
 * WeatherNext records into daily weather per member.
 *
 * Each record is `[member, forecastHour, initMs, precipMetres, tempKelvin]`.
 * Precipitation is the total over the six hours ending at the valid time, so it
 * belongs to the day that period ends in (a period ending at midnight belongs to
 * the day before). Temperature is instantaneous; four samples a day understate
 * the true max and min, which the client's bias correction absorbs.
 *
 * Returns `{ [member]: { [dayMs]: { rain, tmax, tmin, periods } } }`.
 */
export function dailyByMember(records) {
  const out = {}
  for (const [member, hour, init, tp, t2m] of records || []) {
    const valid = Number(init) + Number(hour) * 3600000
    if (!Number.isFinite(valid)) continue
    const m = String(member)
    out[m] ||= {}
    if (Number.isFinite(Number(tp)) && tp !== null) {
      const day = Math.floor((valid - 1) / DAY_MS) * DAY_MS
      const e = (out[m][day] ||= { rain: 0, tmax: -Infinity, tmin: Infinity, periods: 0 })
      e.rain += Math.max(0, Number(tp)) * 1000
      e.periods += 1
    }
    if (Number.isFinite(Number(t2m)) && t2m !== null) {
      const day = Math.floor(valid / DAY_MS) * DAY_MS
      const e = (out[m][day] ||= { rain: 0, tmax: -Infinity, tmin: Infinity, periods: 0 })
      e.tmax = Math.max(e.tmax, Number(t2m) - K)
      e.tmin = Math.min(e.tmin, Number(t2m) - K)
    }
  }
  return out
}

/** One member's days as arrays from `startMs` for `days` days; incomplete days are null. */
export function memberArrays(byDay, startMs, days) {
  const out = { rain: [], tmax: [], tmin: [] }
  for (let i = 0; i < days; i += 1) {
    const e = byDay?.[startMs + i * DAY_MS]
    out.rain.push(e && e.periods >= 4 ? round1(e.rain) : null)
    out.tmax.push(e && Number.isFinite(e.tmax) ? round1(e.tmax) : null)
    out.tmin.push(e && Number.isFinite(e.tmin) ? round1(e.tmin) : null)
  }
  return out
}

/** The ensemble mean of several members' arrays, day by day. */
export function ensembleMean(members) {
  const len = Math.max(0, ...members.map((m) => m.rain.length))
  const out = { rain: [], tmax: [], tmin: [] }
  for (const k of ['rain', 'tmax', 'tmin']) {
    for (let i = 0; i < len; i += 1) {
      const vs = members.map((m) => m[k][i]).filter(Number.isFinite)
      out[k].push(vs.length ? round1(vs.reduce((a, b) => a + b, 0) / vs.length) : null)
    }
  }
  return out
}

const utcMidnight = (ms) => Math.floor(ms / DAY_MS) * DAY_MS

/** Records for a WeatherNext collection at a point: [member, hour, init, tp, t2m]. */
async function weatherNextRecords(collection, point) {
  const feats = collection.map((img) => {
    const v = img.select(['total_precipitation_6hr', '2m_temperature'])
      .reduceRegion({ reducer: ee.Reducer.first(), geometry: point, scale: 27830 })
    return ee.Feature(null, {
      m: img.get('ensemble_member'),
      h: img.get('forecast_hour'),
      t: img.get('system:time_start'),
      tp: v.get('total_precipitation_6hr'),
      t2m: v.get('2m_temperature'),
    })
  })
  const cols = await evaluate(ee.FeatureCollection(feats).reduceColumns(ee.Reducer.toList(5), ['m', 'h', 't', 'tp', 't2m']))
  return cols?.list || []
}

/** One year of observed daily weather at a cell. */
export async function observedYear({ lat, lon, year, now = Date.now() }) {
  await initEarthEngine()
  const point = ee.Geometry.Point([lon, lat])
  const start = `${year}-01-01`
  const endMs = Math.min(Date.UTC(year + 1, 0, 1), utcMidnight(now) + DAY_MS)
  const end = new Date(endMs).toISOString().slice(0, 10)
  if (endMs <= Date.UTC(year, 0, 1)) throw new SeasonWeatherError('That year has not started.')
  const [era5, chirps] = await Promise.all([
    evaluate(ee.ImageCollection(ERA5_DAILY).filterDate(start, end)
      .select(['temperature_2m_max', 'temperature_2m_min']).getRegion(point, 11132)),
    evaluate(ee.ImageCollection(CHIRPS_DAILY).filterDate(start, end).select('precipitation').getRegion(point, 5566)),
  ])
  const series = seriesFromRegion(era5, year, [
    ['temperature_2m_max', 'tmax', (v) => v - K],
    ['temperature_2m_min', 'tmin', (v) => v - K],
  ])
  seriesFromRegion(chirps, year, [['precipitation', 'rain', (v) => v]], series)
  return series
}

/**
 * The latest WeatherNext run at a cell, plus recent short-range forecasts.
 *
 * `members` start at today (UTC) and run FORECAST_DAYS. `recent` is the
 * ensemble mean of each of the last RECENT_DAYS days' 00z run over its first
 * 24 hours, starting at `recentStart`.
 */
export async function forecastAt({ lat, lon, now = Date.now() }) {
  await initEarthEngine()
  const point = ee.Geometry.Point([lon, lat])
  const wn = ee.ImageCollection(WEATHERNEXT)
  const today = utcMidnight(now)

  // Real-time runs are kept for 48 hours; three days back finds the latest
  // even when a run is late.
  const init = await evaluate(wn.filterDate(today - 3 * DAY_MS, now).aggregate_max('system:time_start'))
  if (!Number.isFinite(Number(init))) {
    throw new SeasonWeatherError('No recent WeatherNext run is visible to this Earth Engine project. Check its WeatherNext access.', 502)
  }

  const recentStart = today - RECENT_DAYS * DAY_MS
  const memberIds = Array.from({ length: RECENT_MEMBERS }, (_, i) => String(i))
  const [latest, recent] = await Promise.all([
    weatherNextRecords(wn.filterDate(Number(init), Number(init) + 60000), point),
    weatherNextRecords(
      wn.filterDate(recentStart, today)
        .filter(ee.Filter.calendarRange(0, 0, 'hour'))
        .filter(ee.Filter.inList('ensemble_member', memberIds))
        .filter(ee.Filter.lte('forecast_hour', 24)),
      point,
    ),
  ])

  const byMember = dailyByMember(latest)
  const members = Object.keys(byMember).sort((a, b) => Number(a) - Number(b))
    .map((m) => memberArrays(byMember[m], today, FORECAST_DAYS))
  const recentByMember = dailyByMember(recent)
  const recentMean = ensembleMean(Object.values(recentByMember).map((d) => memberArrays(d, recentStart, RECENT_DAYS)))

  return {
    init: new Date(Number(init)).toISOString(),
    start: new Date(today).toISOString().slice(0, 10),
    members,
    recentStart: new Date(recentStart).toISOString().slice(0, 10),
    recent: recentMean,
  }
}
