import test from 'node:test'
import assert from 'node:assert/strict'

import {
  GDD_BASE, WINDOWS, accumulate, completeYears, conditionsByYear, dailySeries, doyOf,
  driverTable, leadUpProfile, partialCorrelation, quantile, relativeTiming, thresholdTest,
  timingByYear, timingTrend, trailingWindow,
} from '../composables/phenology.js'

/** An observation with a 7-day lead-up window ending on `doy`. */
function obs({ species = 'Test sp', year = 2024, doy, lat = 40.1, lon = -105.1, rain = [], tmax = [], tmin = [] }) {
  const p = { species, date: `${year}-01-01`, day_of_year: doy }
  for (let d = 0; d < 7; d += 1) {
    if (rain[d] !== undefined) p[`prcp_d${d}`] = rain[d]
    if (tmax[d] !== undefined) p[`tmax_d${d}`] = tmax[d]
    if (tmin[d] !== undefined) p[`tmin_d${d}`] = tmin[d]
  }
  return { properties: p, geometry: { coordinates: [lon, lat] } }
}

test('day of year counts from 1 January', () => {
  assert.equal(doyOf('2024-01-01'), 1)
  assert.equal(doyOf('2024-12-31'), 366)   // leap
  assert.equal(doyOf('2023-12-31'), 365)
  assert.equal(doyOf('2024-07-04'), 186)
  assert.equal(doyOf('not a date'), null)
})

test('lead-up windows stitch into one daily series', () => {
  // Two observations three days apart. Their windows overlap on four days, and
  // the result should be one series covering the union, not two fragments.
  const f = [
    obs({ doy: 100, rain: [1, 2, 3, 4, 5, 6, 7] }),      // covers days 94..100
    obs({ doy: 103, rain: [10, 11, 12, 1, 2, 3, 4] }),   // covers days 97..103
  ]
  const s = dailySeries(f)
  assert.equal(s.size, 1, 'same cell and year is one series')
  const { rain } = [...s.values()][0]
  assert.deepEqual([...rain.keys()].sort((a, b) => a - b),
    [94, 95, 96, 97, 98, 99, 100, 101, 102, 103])
  // Day 100 is d0 of the first (1) and d3 of the second (1): they agree.
  assert.equal(rain.get(100), 1)
  // Day 94 only the first saw.
  assert.equal(rain.get(94), 7)
  // Day 103 only the second saw.
  assert.equal(rain.get(103), 10)
})

test('a day reported twice takes the median, not the mean', () => {
  // A 28km cell can hold a thunderstorm that half of it missed. The mean would
  // let one soaked observation define the day for everybody.
  const f = [
    obs({ doy: 100, rain: [0] }),
    obs({ doy: 100, rain: [0] }),
    obs({ doy: 100, rain: [300] }),
  ]
  const { rain } = [...dailySeries(f).values()][0]
  assert.equal(rain.get(100), 0)
})

test('observations in different cells or years do not pool', () => {
  const f = [
    obs({ doy: 100, rain: [5] }),
    obs({ doy: 100, rain: [5], lat: 50 }),          // another cell
    obs({ doy: 100, rain: [5], year: 2023 }),       // another year
  ]
  assert.equal(dailySeries(f).size, 3)
})

test('a window reaching back before 1 January is dropped, not wrapped', () => {
  // Days 0 and below belong to the previous year; silently filing them under
  // this one would put December's rain in January's accumulation.
  const { rain } = [...dailySeries([obs({ doy: 3, rain: [1, 2, 3, 4, 5, 6, 7] })]).values()][0]
  assert.deepEqual([...rain.keys()].sort((a, b) => a - b), [1, 2, 3])
})

test('accumulation adds up and counts what it could see', () => {
  const series = { rain: new Map([[1, 5], [3, 10]]), tmax: new Map(), tmin: new Map() }
  const { rain } = accumulate(series, { maxDoy: 4 })
  assert.deepEqual(rain.get(1), { total: 5, covered: 1 })
  assert.deepEqual(rain.get(2), { total: 5, covered: 1 })   // day 2 unknown
  assert.deepEqual(rain.get(3), { total: 15, covered: 2 })
  assert.deepEqual(rain.get(4), { total: 15, covered: 2 })
})

test('degree days accumulate above the base and never below it', () => {
  const series = {
    rain: new Map(),
    tmax: new Map([[1, 20], [2, 4]]),   // mean 15 -> 10 GDD; mean 2 -> 0, not -3
    tmin: new Map([[1, 10], [2, 0]]),
  }
  const { gdd } = accumulate(series, { maxDoy: 2, base: GDD_BASE })
  assert.equal(gdd.get(1).total, 10)
  assert.equal(gdd.get(2).total, 10)
})

test('a trailing window sums only its own days', () => {
  const series = { rain: new Map([[10, 1], [11, 2], [12, 4], [20, 100]]), tmax: new Map(), tmin: new Map() }
  const w = trailingWindow(series, 12, 3)
  assert.equal(w.rain, 7)          // days 10, 11, 12
  assert.equal(w.rainCovered, 3)
  assert.equal(trailingWindow(series, 12, 30).rain, 7, 'day 20 is in the future')
})

test('timing is the median day, with the season width beside it', () => {
  const f = [100, 100, 110, 120, 200].map((doy) => obs({ doy }))
  const [row] = timingByYear(f, { minObs: 1 })
  assert.equal(row.year, 2024)
  assert.equal(row.n, 5)
  assert.equal(row.median, 110)
  assert.equal(row.first, 100)
  assert.equal(row.last, 200)
  assert.ok(row.q25 <= row.median && row.median <= row.q75)
})

test('a year under the sample floor is dropped rather than reported thin', () => {
  const f = [obs({ doy: 100 }), obs({ doy: 110 })]
  assert.equal(timingByYear(f, { minObs: 20 }).length, 0)
  assert.equal(timingByYear(f, { minObs: 2 }).length, 1)
})

test('quantiles interpolate', () => {
  assert.equal(quantile([1, 2, 3, 4], 0.5), 2.5)
  assert.equal(quantile([1, 2, 3, 4], 0), 1)
  assert.equal(quantile([1, 2, 3, 4], 1), 4)
  assert.equal(quantile([], 0.5), null)
})

test('relative timing subtracts the year the whole dataset had', () => {
  // The species moved 10 days earlier; so did everything else. The species-
  // specific shift is zero, and that is the number worth reading.
  const species = [{ year: 2023, median: 200 }, { year: 2024, median: 190 }]
  const background = [{ year: 2023, median: 180 }, { year: 2024, median: 170 }]
  const out = relativeTiming(species, background)
  assert.deepEqual(out.map((r) => r.relative), [20, 20])
})

test('relative timing is null where the background has no such year', () => {
  const out = relativeTiming([{ year: 1999, median: 200 }], [{ year: 2024, median: 180 }])
  assert.equal(out[0].relative, null)
})

test('a trend reports both its slope and whether the order holds', () => {
  const rows = [2020, 2021, 2022, 2023, 2024].map((year, i) => ({ year, median: 200 - 5 * i }))
  const t = timingTrend(rows)
  assert.ok(Math.abs(t.slope + 5) < 1e-9, 'five days earlier each year')
  assert.equal(t.rho, -1)
  assert.equal(t.n, 5)
  assert.deepEqual(t.span, [2020, 2024])
})

test('a trend needs three points before it will claim one', () => {
  assert.equal(timingTrend([{ year: 2023, median: 1 }, { year: 2024, median: 2 }]).slope, null)
})

test('the incomplete current year is excluded', () => {
  // 2026 stops in March because that is when the data ends; its median would
  // land in March and every species would look wildly early.
  const f = [
    ...[50, 150, 250, 340].map((doy) => obs({ year: 2024, doy })),
    ...[60, 160, 260, 350].map((doy) => obs({ year: 2025, doy })),
    ...[40, 70].map((doy) => obs({ year: 2026, doy })),
  ]
  const complete = completeYears(f)
  assert.ok(complete.has(2024) && complete.has(2025))
  assert.ok(!complete.has(2026), '2026 ended at day 70 while the others ran to ~345')
})

test('the lead-up profile reads rainfall back across the seven days', () => {
  const f = [obs({ doy: 100, rain: [0, 0, 10, 0, 0, 0, 0] }), obs({ doy: 105, rain: [0, 0, 20, 0, 0, 0, 0] })]
  const { species } = leadUpProfile(f)
  assert.equal(species.length, 7)
  assert.equal(species[2].mean, 15, 'both saw their rain two days before the find')
  assert.equal(species[0].mean, 0)
})

test('the lead-up profile compares against a baseline as a ratio', () => {
  const f = [obs({ doy: 100, rain: [0, 0, 10] })]
  const base = [obs({ doy: 100, rain: [0, 0, 5] })]
  const { ratio } = leadUpProfile(f, base)
  assert.equal(ratio[2].ratio, 2, 'twice the usual rain two days before')
})

test('partial correlation removes a shared driver', () => {
  // y is z with noise, x is z with noise: x and y correlate only through z, so
  // holding z still should collapse it. This is the whole point of the column.
  const z = [1, 2, 3, 4, 5, 6, 7, 8]
  const x = z.map((v) => v * 2)
  const y = z.map((v) => v * 3)
  assert.ok(Math.abs(partialCorrelation(x, y, z)) < 1e-9)
})

test('partial correlation keeps a relationship that is not the shared driver', () => {
  // x and y track each other exactly while both only loosely follow z, so
  // holding z still must leave the relationship standing.
  const z = [1, 2, 3, 4, 5, 6, 7, 8]
  const x = [2, 1, 4, 3, 6, 5, 8, 7]
  const y = [2, 1, 4, 3, 6, 5, 8, 7]
  assert.ok(partialCorrelation(x, y, z) > 0.9)
})

test('partial correlation declines when nothing is left after removing z', () => {
  // x and y are both perfectly explained by z. There is no residual
  // relationship to report, and a number here would be an invention.
  const z = [1, 2, 3, 4, 5, 6, 7, 8]
  assert.equal(partialCorrelation(z.map((v) => v * 2), z.map((v) => v * 3), z), null)
})

test('the threshold test says which quantity holds steadier', () => {
  // A species fruiting at a fixed rainfall total: the date wanders, the total
  // does not, which is evidence the total is what it is waiting for.
  const steady = thresholdTest([100, 101, 99, 100, 100], [180, 200, 160, 220, 150])
  assert.equal(steady.steadier, true)
  assert.ok(steady.ratio < 1)

  const wandering = thresholdTest([50, 200, 90, 300, 20], [200, 201, 199, 200, 200])
  assert.equal(wandering.steadier, false)
})

test('the threshold test declines to answer on too little', () => {
  assert.equal(thresholdTest([1, 2], [1, 2]).steadier, null)
})

test('conditions are built from the whole dataset, not one species', () => {
  // The bug this catches: stitching the series from only the focal species
  // leaves it too sparse to reconstruct anything, and every driver comes back
  // empty. Here one species is rare and the rest of the dataset is dense.
  const all = []
  for (let doy = 30; doy <= 210; doy += 3) all.push(obs({ species: 'Common sp', doy, rain: [2, 2, 2, 2, 2, 2, 2] }))
  const focal = [150, 152, 154].map((doy) => obs({ species: 'Rare sp', doy, rain: [2, 2, 2, 2, 2, 2, 2] }))
  const timing = timingByYear(focal, { minObs: 1 })

  const starved = conditionsByYear(focal, timing)
  const fed = conditionsByYear(focal, timing, { allFeatures: [...all, ...focal] })
  assert.equal(starved.rain90, undefined)
  assert.equal(starved[0].rain90, null, 'the species alone cannot reconstruct 90 days')
  assert.ok(Number.isFinite(fed[0].rain90), 'the whole dataset can')
  assert.ok(fed[0].rain90 > 0)
})

test('a window too sparse to trust is left null rather than scaled up wildly', () => {
  // One covered day in ninety would otherwise be multiplied by 90 and reported
  // as a season's rainfall.
  const focal = [obs({ doy: 200, rain: [5] })]
  const timing = timingByYear(focal, { minObs: 1 })
  const rows = conditionsByYear(focal, timing, { allFeatures: focal })
  assert.equal(rows[0].rain90, null)
})

test('the driver table ranks by strength and names what it held constant', () => {
  const rows = [2018, 2019, 2020, 2021, 2022, 2023].map((year, i) => ({
    year, median: 200 - i * 5, relative: 0,
    rain30: i * 10, rain60: 50, rain90: 60, gdd30: 300, gdd60: 500, gdd90: 700,
  }))
  const table = driverTable(rows)
  assert.equal(table.length, WINDOWS.length * 2)
  const rain30 = table.find((d) => d.key === 'rain30')
  assert.ok(Math.abs(rain30.rho + 1) < 1e-9, 'rain30 rises as the date falls')
  assert.equal(rain30.controlledFor, 'heat')
  assert.equal(table.find((d) => d.key === 'gdd30').controlledFor, 'water')
  // Sorted strongest first, ignoring sign.
  const strengths = table.map((d) => Math.abs(d.rho ?? 0))
  assert.deepEqual(strengths, [...strengths].sort((a, b) => b - a))
})

test('the driver table stays quiet when there are too few years', () => {
  const rows = [2022, 2023].map((year, i) => ({ year, median: 200 + i, rain30: i, gdd30: i }))
  for (const d of driverTable(rows)) assert.equal(d.rho, null)
})
