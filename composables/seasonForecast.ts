// Season outlook: when a species' remaining season should start and end near a
// place, as dates with a likely range (WANT-19).
//
// The idea is the one `phenology.ts` tests: fruiting waits for the weather, not
// the calendar. Here it is turned round into a forecast. For each species:
//
// 1. **Fit a trigger.** From past years at this place, find what the weather
//    looked like on the day the season started (growing degree days banked
//    since January, or rain over the trailing 30 days) and on the day it ended
//    (days after the first hard frost). The calendar date is always a candidate
//    too. Each candidate is scored by leave-one-year-out error in days, and the
//    best one wins, so a species whose timing the weather does not explain
//    falls back to its usual dates rather than to a confident wrong answer.
// 2. **Build scenarios.** This year's observed weather up to today, then one
//    WeatherNext ensemble member for the next 15 days, then a past year's
//    weather for the rest of the season. Every member is paired with every past
//    year, so the spread covers both forecast uncertainty and the weeks beyond
//    any forecast's reach.
// 3. **Run the trigger on every scenario.** The start and end dates across
//    scenarios are the outlook; the share of scenarios in season over the next
//    week or two is the chance of finding it.
//
// The weather itself comes from the `season-weather` function. Pure module, no
// framework imports, so it is tested without a page.

import { GDD_BASE } from './phenology'
import { median } from './statistics'

/** Overnight low, in Celsius, that counts as a hard frost. */
export const FROST_C = -2
/** Consecutive frost nights that end a season. One cold night often does not. */
export const FROST_NIGHTS = 2
/** Trailing window, in days, for the rain trigger. */
export const RAIN_WINDOW = 30
/** Fewer past seasons than this and no trigger is fitted. */
export const MIN_YEARS = 3
/** How far either side of past start dates a trigger may fire. */
export const SEARCH_MARGIN = 30
/** Finds per year near the place for that year's season to count. */
export const MIN_FINDS_PER_YEAR = 4

/** Daily weather for one year, indexed by day of year minus one. */
export interface DaySeries {
  rain: (number | null)[]
  tmax: (number | null)[]
  tmin: (number | null)[]
}

export interface Indicators {
  gdd: (number | null)[]
  rain30: (number | null)[]
  /** Day on which a hard frost run completes. */
  frost: boolean[]
}

export interface SeasonYear {
  year: number
  onset: number
  end: number
  n: number
}

export type OnsetBasis = 'doy' | 'gdd' | 'rain30'
export type EndBasis = 'doy' | 'frost'

export interface Trigger {
  onset: { basis: OnsetBasis, value: number, error: number }
  end: { basis: EndBasis, value: number, error: number }
  /** Earliest and latest day the onset may be found on. */
  window: [number, number]
  /** Usual start and end, the fallback when a scenario never fires. */
  usualOnset: number
  usualEnd: number
  years: number
}

export interface Quantiles { p10: number, p50: number, p90: number }

export interface SeasonOutlook {
  status: 'upcoming' | 'in' | 'over' | 'unlikely'
  onset: Quantiles | null
  end: Quantiles | null
  /** Share of scenarios in which the season does not start this year. */
  pNoSeason: number
  /** Share of scenarios in season at some point in the next 7 / 14 days. */
  pNext7: number
  pNext14: number
  scenarios: number
}

const DAYS = 366
const R = 6371

const finite = (v: any): v is number => typeof v === 'number' && Number.isFinite(v)

export function emptySeries(): DaySeries {
  return { rain: Array(DAYS).fill(null), tmax: Array(DAYS).fill(null), tmin: Array(DAYS).fill(null) }
}

/** Day of year, 1-based, in UTC. */
export function dayOfYear(date: Date): number {
  return Math.floor((date.getTime() - Date.UTC(date.getUTCFullYear(), 0, 1)) / 86400000) + 1
}

/** "Oct 22" for a day of a year. */
export function doyLabel(year: number, doy: number): string {
  const d = new Date(Date.UTC(year, 0, Math.round(doy)))
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return NaN
  const i = (sorted.length - 1) * q
  const lo = Math.floor(i)
  const hi = Math.ceil(i)
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

function quantiles(values: number[]): Quantiles | null {
  if (!values.length) return null
  const s = [...values].sort((a, b) => a - b)
  return { p10: Math.round(quantile(s, 0.1)), p50: Math.round(quantile(s, 0.5)), p90: Math.round(quantile(s, 0.9)) }
}

function km(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = Math.PI / 180
  const dLat = (lat2 - lat1) * toRad
  const dLon = (lon2 - lon1) * toRad
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2
  return 2 * R * Math.asin(Math.sqrt(a))
}

/**
 * Running weather totals through a year.
 *
 * GDD carries over a missing day (a hole should not reset the bank); the rain
 * window is scaled up by how much of it was covered, and left null below 70%,
 * so a gap makes it noisier rather than smaller.
 */
export function indicators(series: DaySeries, { base = GDD_BASE } = {}): Indicators {
  const gdd: (number | null)[] = Array(DAYS).fill(null)
  const rain30: (number | null)[] = Array(DAYS).fill(null)
  const frost: boolean[] = Array(DAYS).fill(false)
  let bank = 0
  let run = 0
  let sum = 0
  let covered = 0
  for (let i = 0; i < DAYS; i += 1) {
    const hi = series.tmax[i]
    const lo = series.tmin[i]
    if (finite(hi) && finite(lo)) bank += Math.max(0, (hi + lo) / 2 - base)
    gdd[i] = bank

    run = finite(lo) && lo <= FROST_C ? run + 1 : 0
    frost[i] = run === FROST_NIGHTS

    // Sliding window: add today, drop the day falling out of it.
    const v = series.rain[i]
    if (finite(v)) { sum += v; covered += 1 }
    if (i >= RAIN_WINDOW) {
      const out = series.rain[i - RAIN_WINDOW]
      if (finite(out)) { sum -= out; covered -= 1 }
    }
    if (i >= RAIN_WINDOW - 1) rain30[i] = covered >= RAIN_WINDOW * 0.7 ? sum * (RAIN_WINDOW / covered) : null
  }
  return { gdd, rain30, frost }
}

/** First day in [from, to] (1-based) the indicator reaches `value`, else null. */
function firstCrossing(values: (number | null)[], value: number, from: number, to: number): number | null {
  for (let d = Math.max(1, from); d <= Math.min(DAYS, to); d += 1) {
    const v = values[d - 1]
    if (finite(v) && v >= value) return d
  }
  return null
}

/** First day on or after `from` that completes a hard frost run, else null. */
function firstFrost(ind: Indicators, from: number): number | null {
  for (let d = Math.max(1, from); d <= DAYS; d += 1) if (ind.frost[d - 1]) return d
  return null
}

/**
 * Each species' season, per year, from finds near a place.
 *
 * Start is the 10th percentile of the year's find dates and end the 90th: the
 * first and last finds are whoever happened to go out earliest and latest.
 * Years with fewer than MIN_FINDS_PER_YEAR finds nearby are dropped, and the
 * current year is returned separately because it is not over.
 */
export function localSeasons(
  rows: any[],
  { lat, lon, radiusKm = 50, year = new Date().getUTCFullYear(), minFinds = MIN_FINDS_PER_YEAR }:
  { lat: number, lon: number, radiusKm?: number, year?: number, minFinds?: number },
): Map<string, { past: SeasonYear[], current: { n: number, first: number, last: number } | null }> {
  const by = new Map<string, Map<number, number[]>>()
  for (const r of rows || []) {
    if (!r?.species || !r.date || !finite(r.lat) || !finite(r.lon)) continue
    if (km(lat, lon, r.lat, r.lon) > radiusKm) continue
    const t = Date.parse(`${String(r.date).slice(0, 10)}T00:00:00Z`)
    if (!Number.isFinite(t)) continue
    const d = new Date(t)
    const y = d.getUTCFullYear()
    if (!by.has(r.species)) by.set(r.species, new Map())
    const years = by.get(r.species)!
    if (!years.has(y)) years.set(y, [])
    years.get(y)!.push(dayOfYear(d))
  }

  const out = new Map()
  for (const [species, years] of by) {
    const past: SeasonYear[] = []
    let current = null
    for (const [y, doys] of years) {
      const s = doys.sort((a, b) => a - b)
      if (y === year) {
        current = { n: s.length, first: s[0], last: s[s.length - 1] }
        continue
      }
      if (y > year || s.length < minFinds) continue
      past.push({ year: y, onset: Math.round(quantile(s, 0.1)), end: Math.round(quantile(s, 0.9)), n: s.length })
    }
    past.sort((a, b) => a.year - b.year)
    out.set(species, { past, current })
  }
  return out
}

interface FitYear extends SeasonYear { ind: Indicators }

function predictOnset(basis: OnsetBasis, value: number, ind: Indicators, window: [number, number]): number | null {
  if (basis === 'doy') return value
  return firstCrossing(basis === 'gdd' ? ind.gdd : ind.rain30, value, window[0], window[1])
}

function predictEnd(basis: EndBasis, value: number, ind: Indicators, onset: number, usualEnd: number): number {
  if (basis === 'doy') return Math.max(onset, value)
  const f = firstFrost(ind, onset)
  return f === null ? Math.max(onset, usualEnd) : Math.max(onset, f + value)
}

const meanAbs = (xs: number[]) => xs.reduce((a, b) => a + Math.abs(b), 0) / xs.length

/**
 * Fit the start and end triggers from past seasons and their weather.
 *
 * Every candidate is scored the same way: leave one year out, set the
 * threshold from the rest (their median), predict the held-out year, and
 * average the miss in days. A candidate that never fires in a held-out year
 * is charged the whole search margin, so "never fires" is not free.
 */
export function fitTrigger(years: { season: SeasonYear, series: DaySeries }[]): Trigger | null {
  const fit: FitYear[] = years
    .filter(({ season }) => finite(season.onset) && finite(season.end))
    .map(({ season, series }) => ({ ...season, ind: indicators(series) }))
  if (fit.length < MIN_YEARS) return null

  const onsets = fit.map((y) => y.onset)
  const window: [number, number] = [Math.min(...onsets) - SEARCH_MARGIN, Math.max(...onsets) + SEARCH_MARGIN]
  const usualOnset = Math.round(median(onsets)!)
  const usualEnd = Math.round(median(fit.map((y) => y.end))!)

  const onsetValue = (basis: OnsetBasis, y: FitYear): number | null => {
    if (basis === 'doy') return y.onset
    const v = (basis === 'gdd' ? y.ind.gdd : y.ind.rain30)[y.onset - 1]
    return finite(v) ? v : null
  }

  let bestOnset: Trigger['onset'] | null = null
  for (const basis of ['doy', 'gdd', 'rain30'] as OnsetBasis[]) {
    const errors: number[] = []
    for (let i = 0; i < fit.length; i += 1) {
      const rest = fit.filter((_, j) => j !== i).map((y) => onsetValue(basis, y)).filter(finite)
      if (rest.length < MIN_YEARS - 1) break
      const p = predictOnset(basis, median(rest)!, fit[i].ind, window)
      errors.push(p === null ? SEARCH_MARGIN : p - fit[i].onset)
    }
    if (errors.length < fit.length) continue
    const all = fit.map((y) => onsetValue(basis, y)).filter(finite)
    const candidate = { basis, value: median(all)!, error: meanAbs(errors) }
    // Ties go to the calendar: it is the simpler claim.
    if (!bestOnset || candidate.error < bestOnset.error - 0.5) bestOnset = candidate
  }

  const lag = (y: FitYear): number | null => {
    const f = firstFrost(y.ind, y.onset)
    return f === null ? null : y.end - f
  }
  let bestEnd: Trigger['end'] | null = null
  for (const basis of ['doy', 'frost'] as EndBasis[]) {
    const errors: number[] = []
    for (let i = 0; i < fit.length; i += 1) {
      const others = fit.filter((_, j) => j !== i)
      const rest = basis === 'doy' ? others.map((y) => y.end) : others.map(lag).filter(finite)
      if (rest.length < MIN_YEARS - 1) break
      const othersEnd = Math.round(median(others.map((y) => y.end))!)
      errors.push(predictEnd(basis, median(rest)!, fit[i].ind, fit[i].onset, othersEnd) - fit[i].end)
    }
    if (errors.length < fit.length) continue
    const all = basis === 'doy' ? fit.map((y) => y.end) : fit.map(lag).filter(finite)
    const candidate = { basis, value: median(all)!, error: meanAbs(errors) }
    if (!bestEnd || candidate.error < bestEnd.error - 0.5) bestEnd = candidate
  }

  return { onset: bestOnset!, end: bestEnd!, window, usualOnset, usualEnd, years: fit.length }
}

export interface Bias { tmax: number, tmin: number, rainRatio: number, days: number }

/**
 * How far WeatherNext's short-range forecasts sit from the observed record at
 * this place, over the days both cover.
 *
 * A 0.25 degree forecast sampled four times a day reads a valley's highs low
 * and its lows high, and rain differently from CHIRPS. Temperatures take the
 * mean offset; rain takes the ratio of totals, clamped to 0.5–2 and left at 1
 * when the forecast saw under 5 mm, since a ratio of near-nothings is noise.
 * Fewer than 5 days in common and nothing is corrected.
 */
export function forecastBias(observed: DaySeries, recent: { rain: (number | null)[], tmax: (number | null)[], tmin: (number | null)[] }, recentStart: number): Bias {
  const dt = { tmax: [] as number[], tmin: [] as number[] }
  let obsRain = 0
  let fcRain = 0
  let days = 0
  for (let i = 0; i < recent.rain.length; i += 1) {
    const d = recentStart + i
    if (d < 1 || d > DAYS) continue
    const o = { rain: observed.rain[d - 1], tmax: observed.tmax[d - 1], tmin: observed.tmin[d - 1] }
    if (finite(o.tmax) && finite(recent.tmax[i])) dt.tmax.push(o.tmax - recent.tmax[i]!)
    if (finite(o.tmin) && finite(recent.tmin[i])) dt.tmin.push(o.tmin - recent.tmin[i]!)
    if (finite(o.rain) && finite(recent.rain[i])) { obsRain += o.rain; fcRain += recent.rain[i]!; days += 1 }
  }
  const mean = (xs: number[]) => (xs.length >= 5 ? xs.reduce((a, b) => a + b, 0) / xs.length : 0)
  const rainRatio = days >= 5 && fcRain >= 5 ? Math.min(2, Math.max(0.5, obsRain / fcRain)) : 1
  return { tmax: mean(dt.tmax), tmin: mean(dt.tmin), rainRatio, days: Math.max(dt.tmax.length, days) }
}

/** A forecast's days with the bias taken out. */
export function correct<T extends { rain: (number | null)[], tmax: (number | null)[], tmin: (number | null)[] }>(f: T, b: Bias): T {
  return {
    ...f,
    rain: f.rain.map((v) => (finite(v) ? v * b.rainRatio : null)),
    tmax: f.tmax.map((v) => (finite(v) ? v + b.tmax : null)),
    tmin: f.tmin.map((v) => (finite(v) ? v + b.tmin : null)),
  }
}

/**
 * This year to date: the observed record, with the days it has not caught up
 * on (ERA5-Land runs about a week behind) taken from the corrected recent
 * short-range forecasts.
 */
export function currentToDate(observed: DaySeries, recent: { rain: (number | null)[], tmax: (number | null)[], tmin: (number | null)[] }, recentStart: number, today: number, bias: Bias): DaySeries {
  const fixed = correct(recent, bias)
  const s: DaySeries = { rain: [...observed.rain], tmax: [...observed.tmax], tmin: [...observed.tmin] }
  for (let i = 0; i < fixed.rain.length; i += 1) {
    const d = recentStart + i
    if (d < 1 || d >= today) continue
    for (const k of ['rain', 'tmax', 'tmin'] as const) {
      if (!finite(s[k][d - 1]) && finite(fixed[k][i])) s[k][d - 1] = fixed[k][i]
    }
  }
  for (let d = today; d <= DAYS; d += 1) { s.rain[d - 1] = null; s.tmax[d - 1] = null; s.tmin[d - 1] = null }
  return s
}

/**
 * Every combination of forecast member and past year, as whole-year series.
 *
 * Days before `today` come from `current` (observed); `today` onward takes
 * the member's forecast while it lasts, then the past year's weather on the
 * same day of year. With no members the scenarios are past years alone, the
 * climatology outlook.
 */
export function buildScenarios(
  { current, members, analogs, today }:
  { current: DaySeries, members: DaySeries[] | { rain: (number | null)[], tmax: (number | null)[], tmin: (number | null)[] }[], analogs: DaySeries[], today: number },
): DaySeries[] {
  const out: DaySeries[] = []
  const heads = members.length ? members : [null]
  for (const m of heads) {
    for (const a of analogs) {
      const s = emptySeries()
      for (let i = 0; i < DAYS; i += 1) {
        const d = i + 1
        for (const k of ['rain', 'tmax', 'tmin'] as const) {
          let v: number | null = null
          if (d < today) v = current[k][i]
          else if (m && d - today < m[k].length) v = m[k][d - today]
          if (!finite(v)) v = a[k][i]
          s[k][i] = finite(v) ? v : null
        }
      }
      out.push(s)
    }
  }
  return out
}

/**
 * The outlook for one species: run its trigger over every scenario.
 *
 * `scenarios` may be series or their `indicators`; pass the latter when
 * forecasting several species over the same scenarios.
 *
 * `seen` is this year's finds near the place, when there are any: a season
 * already under way is not something to forecast, so finds this year pull the
 * start no later than the first of them.
 */
export function forecastSeason(
  trigger: Trigger,
  scenarios: (DaySeries | Indicators)[],
  today: number,
  seen: { n: number, first: number, last: number } | null = null,
): SeasonOutlook {
  const onsets: number[] = []
  const ends: number[] = []
  let none = 0
  let in7 = 0
  let in14 = 0
  const started = seen && seen.n >= 2 ? seen.first : null

  for (const s of scenarios) {
    // Callers forecasting many species pass indicators, computed once per scenario.
    const ind = 'gdd' in s ? s : indicators(s)
    let onset = predictOnset(trigger.onset.basis, trigger.onset.value, ind, trigger.window)
    if (started !== null) onset = onset === null ? started : Math.min(onset, started)
    if (onset === null) { none += 1; continue }
    const end = predictEnd(trigger.end.basis, trigger.end.value, ind, onset, trigger.usualEnd)
    onsets.push(onset)
    ends.push(end)
    if (onset <= today + 6 && end >= today) in7 += 1
    if (onset <= today + 13 && end >= today) in14 += 1
  }

  const n = scenarios.length || 1
  const onset = quantiles(onsets)
  const end = quantiles(ends)
  const pNoSeason = none / n
  let status: SeasonOutlook['status'] = 'upcoming'
  if (!onset || pNoSeason > 0.5) status = 'unlikely'
  else if (end && end.p50 < today) status = 'over'
  else if (onset.p50 <= today) status = 'in'
  return { status, onset, end, pNoSeason, pNext7: in7 / n, pNext14: in14 / n, scenarios: scenarios.length }
}

/** Plain-language basis for a trigger, for a tooltip. */
export function describeTrigger(t: Trigger): string {
  const onset = {
    doy: 'its usual start date',
    gdd: `${Math.round(t.onset.value)} growing degree days since January`,
    rain30: `${Math.round(t.onset.value)} mm of rain over 30 days`,
  }[t.onset.basis]
  const end = t.end.basis === 'frost'
    ? `${Math.round(t.end.value)} days after the first hard frost`
    : 'its usual end date'
  return `Starts at ${onset} (±${Math.round(t.onset.error)}d in past years), ends ${end} (±${Math.round(t.end.error)}d), from ${t.years} seasons.`
}
