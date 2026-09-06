// Phenology: what moves the timing of fruiting, and by how much.
//
// The question this exists to answer is "why did this species fruit three weeks
// early in 2023" and, behind it, "is that rain or temperature, recent or
// accumulated". Answering it needs three things the rest of the app does not
// have: a season-long weather series, a timing statistic per species per year,
// and a way to tell a real shift from a shift in when people went looking.
//
// **The weather series.** Each observation carries the rain and temperature of
// the seven days before it (prcp_d0..d6, tmax_d0..d6, tmin_d0..d6). Those
// windows overlap: two finds a few days apart in the same place report the same
// calendar days. So pooling them reconstructs a daily series for a cell, and a
// daily series gives what a single observation cannot — rain accumulated since
// the start of the year, growing degree days, a trailing 30-day total.
//
// This is only worth doing because the overlaps agree. Across the shipped
// dataset, 61,794 cell-days are reported by two or more observations, and their
// median disagreement is 0.00mm with 90% inside 0.5mm. They are sampling one
// underlying series. The tail that does not agree is real: a 0.25 degree cell is
// about 28km across and a thunderstorm is not, so cell-days take the **median**
// across their reports rather than the mean, which one convective cell would
// drag.
//
// Pure module, no framework imports, so all of this is testable without a page.

import { median, spearman } from './statistics.js'

/** Base temperature for growing degree days, in Celsius. */
export const GDD_BASE = 5

/** Default grid for pooling weather. Coarse: this is climate, not terrain. */
export const WEATHER_CELL = 0.25

/** How many days of lead-up each observation carries. */
export const LEAD_DAYS = 7

const num = (v) => {
  const n = Number(v)
  return Number.isFinite(n) ? n : null
}

/** Day of year for an ISO date, 1-based, ignoring leap-day drift. */
export function doyOf(dateStr) {
  const t = Date.parse(`${String(dateStr).slice(0, 10)}T00:00:00Z`)
  if (!Number.isFinite(t)) return null
  const d = new Date(t)
  const start = Date.UTC(d.getUTCFullYear(), 0, 1)
  return Math.floor((t - start) / 86400000) + 1
}

const yearOf = (dateStr) => Number(String(dateStr).slice(0, 4)) || null

const cellKey = (lat, lon, size) => `${Math.floor(lat / size)}:${Math.floor(lon / size)}`

/**
 * A daily weather series per cell and year, stitched from the lead-up windows.
 *
 * Returns Map("cell|year" -> { rain: Map(doy -> mm), tmax: Map, tmin: Map }).
 * Each day is the median of everything that reported it, so one storm-struck
 * observation cannot define the day for a cell 20km wide.
 */
export function dailySeries(features, { cellSize = WEATHER_CELL } = {}) {
  // Collect every report first, then reduce: the median needs all of them.
  const reports = new Map()

  for (const f of features) {
    const p = f?.properties || {}
    const co = f?.geometry?.coordinates
    if (!co || !p.date) continue
    const lon = num(co[0]); const lat = num(co[1])
    if (lat === null || lon === null) continue

    const year = yearOf(p.date)
    const doy = num(p.day_of_year) ?? doyOf(p.date)
    if (!year || doy === null) continue
    const cell = cellKey(lat, lon, cellSize)

    for (let d = 0; d < LEAD_DAYS; d += 1) {
      // d days before the observation. A window can reach back over New Year;
      // those days belong to the previous year's series, and dropping them is
      // simpler than stitching across the boundary for six days of January.
      const day = doy - d
      if (day < 1) continue
      const key = `${cell}|${year}`
      let entry = reports.get(key)
      if (!entry) { entry = { rain: new Map(), tmax: new Map(), tmin: new Map() }; reports.set(key, entry) }
      for (const [field, prefix] of [['rain', 'prcp_d'], ['tmax', 'tmax_d'], ['tmin', 'tmin_d']]) {
        const v = num(p[`${prefix}${d}`])
        if (v === null) continue
        const bucket = entry[field]
        if (!bucket.has(day)) bucket.set(day, [])
        bucket.get(day).push(v)
      }
    }
  }

  const out = new Map()
  for (const [key, entry] of reports) {
    out.set(key, {
      rain: reduceDays(entry.rain),
      tmax: reduceDays(entry.tmax),
      tmin: reduceDays(entry.tmin),
    })
  }
  return out
}

function reduceDays(bucket) {
  const out = new Map()
  for (const [day, values] of bucket) out.set(day, values.length === 1 ? values[0] : median(values))
  return out
}

/**
 * Running totals through the year for one cell-year's series.
 *
 * `rain` accumulates millimetres from 1 January; `gdd` accumulates degree days
 * above GDD_BASE. Both carry a `covered` count, because a total over a series
 * with holes in it understates the real one and the caller has to be able to
 * say so rather than quietly comparing a well-sampled year against a sparse one.
 */
export function accumulate(series, { base = GDD_BASE, maxDoy = 366 } = {}) {
  const rain = new Map(); const gdd = new Map()
  let rainSum = 0; let gddSum = 0; let rainDays = 0; let gddDays = 0

  for (let d = 1; d <= maxDoy; d += 1) {
    const r = series.rain.get(d)
    if (r !== undefined) { rainSum += r; rainDays += 1 }
    const hi = series.tmax.get(d); const lo = series.tmin.get(d)
    if (hi !== undefined && lo !== undefined) {
      gddSum += Math.max(0, (hi + lo) / 2 - base)
      gddDays += 1
    }
    rain.set(d, { total: rainSum, covered: rainDays })
    gdd.set(d, { total: gddSum, covered: gddDays })
  }
  return { rain, gdd }
}

/** Rain falling in the `window` days up to and including `doy`. */
export function trailingRain(series, doy, window) {
  let total = 0; let covered = 0
  for (let d = doy - window + 1; d <= doy; d += 1) {
    const v = series.rain.get(d)
    if (v !== undefined) { total += v; covered += 1 }
  }
  return { total, covered, window }
}

/**
 * When a set of observations happened, per year.
 *
 * `median` is the timing statistic: robust to the long tail a fruiting season
 * has, and meaningful at sample sizes where a fitted peak is noise. `q25`/`q75`
 * carry the width of the season, which moves independently of its centre — a
 * season can start on time and run long.
 */
export function timingByYear(features, { minObs = 20 } = {}) {
  const byYear = new Map()
  for (const f of features) {
    const p = f?.properties || {}
    if (!p.date) continue
    const year = yearOf(p.date)
    const doy = num(p.day_of_year) ?? doyOf(p.date)
    if (!year || doy === null) continue
    if (!byYear.has(year)) byYear.set(year, [])
    byYear.get(year).push(doy)
  }

  const rows = []
  for (const [year, days] of byYear) {
    if (days.length < minObs) continue
    const sorted = [...days].sort((a, b) => a - b)
    rows.push({
      year,
      n: sorted.length,
      median: median(sorted),
      q25: quantile(sorted, 0.25),
      q75: quantile(sorted, 0.75),
      first: sorted[0],
      last: sorted[sorted.length - 1],
    })
  }
  return rows.sort((a, b) => a.year - b.year)
}

export function quantile(sorted, q) {
  if (!sorted.length) return null
  const i = (sorted.length - 1) * q
  const lo = Math.floor(i); const hi = Math.ceil(i)
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo)
}

/**
 * The same timing, with the year's general recording shifted out of it.
 *
 * A species' median moving ten days earlier means nothing if every species
 * moved ten days earlier, because that is a change in when people went out, not
 * in when anything fruited. Subtracting the median of ALL observations in the
 * same years leaves the part specific to this species.
 *
 * This is the single most important correction here, so it is the default and
 * the raw figure is kept beside it rather than replaced.
 */
export function relativeTiming(speciesRows, backgroundRows) {
  const bg = new Map(backgroundRows.map((r) => [r.year, r.median]))
  return speciesRows.map((r) => {
    const base = bg.get(r.year)
    return {
      ...r,
      background: base ?? null,
      relative: base === undefined ? null : r.median - base,
    }
  })
}

/**
 * Least-squares slope of a timing series, in days per year, with Spearman rho.
 *
 * The slope says how much and which way; rho says whether the ordering is
 * consistent enough to be worth reading. Both are reported because a slope
 * fitted through four scattered years will happily look dramatic.
 */
export function timingTrend(rows, key = 'median') {
  const pts = rows.filter((r) => Number.isFinite(r[key])).map((r) => [r.year, r[key]])
  if (pts.length < 3) return { slope: null, rho: null, n: pts.length }
  const xs = pts.map((p) => p[0]); const ys = pts.map((p) => p[1])
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length
  const my = ys.reduce((a, b) => a + b, 0) / ys.length
  let numr = 0; let den = 0
  for (let i = 0; i < xs.length; i += 1) {
    numr += (xs[i] - mx) * (ys[i] - my)
    den += (xs[i] - mx) ** 2
  }
  return {
    slope: den === 0 ? null : numr / den,
    rho: spearman(xs, ys),
    n: pts.length,
    span: [Math.min(...xs), Math.max(...xs)],
  }
}

/**
 * Mean rainfall on each of the seven days before a find, against a baseline.
 *
 * This is the short-term half of the question. A species that fruits after rain
 * shows a hump somewhere in its profile; the baseline is what the same days
 * looked like across every observation in the dataset, so the comparison is
 * against "a day somebody was out recording" rather than against zero.
 */
export function leadUpProfile(features, baseline = null) {
  const lag = (rows) => {
    const out = []
    for (let d = 0; d < LEAD_DAYS; d += 1) {
      let sum = 0; let n = 0
      for (const f of rows) {
        const v = num((f?.properties || {})[`prcp_d${d}`])
        if (v === null) continue
        sum += v; n += 1
      }
      out.push({ lag: d, mean: n ? sum / n : null, n })
    }
    return out
  }
  const species = lag(features)
  if (!baseline) return { species, baseline: null }
  const base = lag(baseline)
  return {
    species,
    baseline: base,
    // Ratio rather than difference: rainfall is not on a scale where "2mm more"
    // means the same thing in a wet region and a dry one.
    ratio: species.map((s, i) => ({
      lag: s.lag,
      ratio: base[i]?.mean ? s.mean / base[i].mean : null,
    })),
  }
}

/** Spearman correlation of x and y with the influence of z removed. */
export function partialCorrelation(xs, ys, zs) {
  const rxy = spearman(xs, ys)
  const rxz = spearman(xs, zs)
  const ryz = spearman(ys, zs)
  if (![rxy, rxz, ryz].every(Number.isFinite)) return null
  const den = Math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
  if (!(den > 1e-9)) return null
  return (rxy - rxz * ryz) / den
}

/**
 * Is the timing better explained by the calendar or by an accumulation?
 *
 * The phenological hypothesis is that fruiting waits for a threshold rather than
 * a date: enough rain in the ground, enough warmth banked. If that holds, then
 * across years the accumulated total at the moment of fruiting should vary LESS
 * than the calendar date does. Both are put on the same footing by their
 * coefficient of variation, which is unitless, so millimetres and days can be
 * compared at all.
 *
 * A `cv` below the date's is evidence for the threshold; above it is evidence
 * against. It is evidence, not proof: with six years there is not much of it.
 */
export function thresholdTest(values, dates) {
  const cv = (arr) => {
    const xs = arr.filter(Number.isFinite)
    if (xs.length < 3) return null
    const m = xs.reduce((a, b) => a + b, 0) / xs.length
    if (Math.abs(m) < 1e-9) return null
    const sd = Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
    return sd / Math.abs(m)
  }
  const cvValue = cv(values)
  const cvDate = cv(dates)
  if (cvValue === null || cvDate === null) return { cvValue, cvDate, steadier: null, ratio: null }
  return {
    cvValue,
    cvDate,
    ratio: cvValue / cvDate,
    steadier: cvValue < cvDate,
  }
}

/**
 * Which years ran their full course.
 *
 * The current year is truncated: its observations stop at today, so its median
 * find lands wherever the season had got to rather than at the season's centre.
 * Including it makes every species look dramatically early. A year counts as
 * complete when the dataset as a whole recorded as late in it as it usually
 * does; the comparison is against the other years rather than against a fixed
 * date, so a dataset that simply stops in October is not judged against
 * December.
 */
export function completeYears(allFeatures, { tolerance = 45 } = {}) {
  const lastByYear = new Map()
  for (const f of allFeatures) {
    const p = f?.properties || {}
    if (!p.date) continue
    const year = yearOf(p.date)
    const doy = num(p.day_of_year) ?? doyOf(p.date)
    if (!year || doy === null) continue
    lastByYear.set(year, Math.max(lastByYear.get(year) ?? 0, doy))
  }
  const ends = [...lastByYear.values()].sort((a, b) => a - b)
  if (!ends.length) return new Set()
  const typical = median(ends)
  const out = new Set()
  for (const [year, last] of lastByYear) if (last >= typical - tolerance) out.add(year)
  return out
}

/**
 * What conditions each year's fruiting happened under, one row per year.
 *
 * Pairs the species' timing with the weather leading up to it: rain and heat
 * over the 30, 60 and 90 days before the median find.
 *
 * **Trailing windows, not totals since January.** The reconstructed series only
 * covers days somebody was out recording, and in this dataset that is the
 * fruiting season: January has 2,024 reconstructed rain-days against August's
 * 13,159. A total "since 1 January" would therefore be a total since roughly
 * April, varying with how early people started that year, which is recording
 * effort wearing a hydrologist's coat. A trailing window sits inside the
 * covered period and is scaled by how much of itself was covered, so a gap
 * makes it noisier rather than smaller.
 *
 * `allFeatures` is the whole dataset and is not optional: the series is stitched
 * from overlapping windows, and one species alone does not overlap itself
 * densely enough to reconstruct anything. Passing only the species produced
 * empty columns, which is how this was caught.
 */
export const WINDOWS = [30, 60, 90]

export function conditionsByYear(features, timing, { allFeatures, cellSize = WEATHER_CELL } = {}) {
  const source = allFeatures || features
  const series = dailySeries(source, { cellSize })

  const cellsByYear = new Map()
  for (const f of features) {
    const p = f?.properties || {}
    const co = f?.geometry?.coordinates
    if (!co || !p.date) continue
    const lat = num(co[1]); const lon = num(co[0])
    if (lat === null || lon === null) continue
    const year = yearOf(p.date)
    if (!year) continue
    if (!cellsByYear.has(year)) cellsByYear.set(year, new Set())
    cellsByYear.get(year).add(cellKey(lat, lon, cellSize))
  }

  return timing.map((row) => {
    const cells = [...(cellsByYear.get(row.year) || [])]
    const doy = Math.round(row.median)
    const acc = {}
    for (const w of WINDOWS) acc[`rain${w}`] = []
    for (const w of WINDOWS) acc[`gdd${w}`] = []

    for (const cell of cells) {
      const s = series.get(`${cell}|${row.year}`)
      if (!s) continue
      for (const w of WINDOWS) {
        const r = trailingWindow(s, doy, w, GDD_BASE)
        // Under a third covered is not a window, it is a rumour. Scaling what
        // remains up to the full width would invent most of the total.
        if (r.rainCovered >= w / 3) acc[`rain${w}`].push(r.rain * (w / r.rainCovered))
        if (r.gddCovered >= w / 3) acc[`gdd${w}`].push(r.gdd * (w / r.gddCovered))
      }
    }

    const avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null)
    const out = { ...row, cells: cells.length }
    for (const key of Object.keys(acc)) out[key] = avg(acc[key])
    return out
  })
}

/** Rain and degree days over the `window` days ending at `doy`. */
export function trailingWindow(series, doy, window, base = GDD_BASE) {
  let rain = 0; let rainCovered = 0; let gdd = 0; let gddCovered = 0
  for (let d = doy - window + 1; d <= doy; d += 1) {
    const r = series.rain.get(d)
    if (r !== undefined) { rain += r; rainCovered += 1 }
    const hi = series.tmax.get(d); const lo = series.tmin.get(d)
    if (hi !== undefined && lo !== undefined) {
      gdd += Math.max(0, (hi + lo) / 2 - base)
      gddCovered += 1
    }
  }
  return { rain, rainCovered, gdd, gddCovered, window }
}

/**
 * The candidate drivers, ranked, with the confound between them removed.
 *
 * Rain and warmth are not independent: a warm year is often a dry one, so a
 * correlation between timing and either is partly the other showing through.
 * Each driver therefore gets both its plain correlation and its correlation
 * with the rival family held constant. When those disagree, the plain one was
 * borrowing, and the partial is the one to read.
 */
export const DRIVERS = [
  ...WINDOWS.map((w) => ({
    key: `rain${w}`, label: `Rain in the previous ${w} days`, family: 'water', unit: 'mm',
    note: `Total rainfall over the ${w} days before the median find, scaled up for gaps in the series.`,
  })),
  ...WINDOWS.map((w) => ({
    key: `gdd${w}`, label: `Degree days in the previous ${w}`, family: 'heat', unit: 'GDD',
    note: `Warmth above ${GDD_BASE} C accumulated over the ${w} days before the median find.`,
  })),
]

export function driverTable(rows, { timingKey = 'median' } = {}) {
  const usable = rows.filter((r) => Number.isFinite(r[timingKey]))
  const timing = usable.map((r) => r[timingKey])

  // The rival family, as one series, for the partial correlation to hold still.
  const familyMean = (family) => {
    const keys = DRIVERS.filter((d) => d.family === family).map((d) => d.key)
    return usable.map((r) => {
      const vs = keys.map((k) => r[k]).filter(Number.isFinite)
      return vs.length ? vs.reduce((a, b) => a + b, 0) / vs.length : null
    })
  }
  const water = familyMean('water')
  const heat = familyMean('heat')

  return DRIVERS.map((d) => {
    const pairs = usable
      .map((r, i) => [r[d.key], timing[i], d.family === 'water' ? heat[i] : water[i]])
      .filter(([v, t]) => Number.isFinite(v) && Number.isFinite(t))
    const xs = pairs.map((p) => p[0]); const ys = pairs.map((p) => p[1])
    const zs = pairs.map((p) => p[2])
    const complete = pairs.filter((p) => Number.isFinite(p[2]))
    return {
      ...d,
      n: pairs.length,
      rho: pairs.length >= 4 ? spearman(xs, ys) : null,
      // Held against the other family: heat for a water driver, water for heat.
      partial: complete.length >= 5
        ? partialCorrelation(complete.map((p) => p[0]), complete.map((p) => p[1]), complete.map((p) => p[2]))
        : null,
      controlledFor: d.family === 'water' ? 'heat' : 'water',
    }
  }).sort((a, b) => Math.abs(b.rho ?? 0) - Math.abs(a.rho ?? 0))
}
