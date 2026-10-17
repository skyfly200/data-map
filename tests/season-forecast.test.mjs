import test from 'node:test'
import assert from 'node:assert/strict'

import {
  FROST_C, buildScenarios, correct, currentToDate, emptySeries, fitTrigger, forecastBias, forecastSeason,
  indicators, localSeasons,
} from '../composables/seasonForecast.js'
import {
  dailyByMember, ensembleMean, memberArrays, seriesFromRegion, snapCell,
} from '../netlify/lib/season-weather.mjs'
import { handleSeasonWeather } from '../netlify/functions/season-weather.mjs'

/**
 * A year whose mean temperature runs a sine from winter to summer, shifted by
 * `warm` degrees, with `rainDay` starting a wet spell of 6 mm/day for 20 days
 * and a hard frost from `frostDay`.
 */
function year({ warm = 0, rainDay = 230, frostDay = 300 } = {}) {
  const s = emptySeries()
  for (let d = 1; d <= 366; d += 1) {
    const mean = 10 + warm - 12 * Math.cos(((d - 15) / 365) * 2 * Math.PI)
    s.tmax[d - 1] = mean + 6
    s.tmin[d - 1] = d >= frostDay ? FROST_C - 3 : mean - 6
    s.rain[d - 1] = d >= rainDay && d < rainDay + 20 ? 6 : 0.5
  }
  return s
}

/** The day the series' GDD first reaches `g`. */
const gddDay = (s, g) => indicators(s).gdd.findIndex((v) => v >= g) + 1

test('indicators: GDD accumulates, rain window covers 30 days, frost run completes on the 2nd night', () => {
  const s = year({ frostDay: 300 })
  const ind = indicators(s)
  assert.ok(ind.gdd[200] > ind.gdd[100])
  assert.equal(ind.rain30[28], null)
  assert.ok(Math.abs(ind.rain30[29] - 15) < 1e-9)
  assert.equal(ind.frost[298], false)
  assert.equal(ind.frost[300], true)   // day 301 = second night
})

test('fitTrigger picks GDD when the season follows banked warmth', () => {
  const warms = [-2, -1, 0, 1, 2, -1.5]
  const years = warms.map((warm, i) => {
    const series = year({ warm })
    const onset = gddDay(series, 1500)
    return { season: { year: 2018 + i, onset, end: onset + 40, n: 20 }, series }
  })
  // The calendar moves with the warmth; the trigger should not.
  const onsets = years.map((y) => y.season.onset)
  assert.ok(Math.max(...onsets) - Math.min(...onsets) > 15)

  const t = fitTrigger(years)
  assert.equal(t.onset.basis, 'gdd')
  assert.ok(t.onset.error < 3)
  assert.ok(Math.abs(t.onset.value - 1500) < 40)
})

test('fitTrigger picks rain when the season follows a wet spell', () => {
  const rains = [215, 230, 245, 225, 250, 235]
  const years = rains.map((rainDay, i) => ({
    season: { year: 2015 + i, onset: rainDay + 12, end: rainDay + 50, n: 10 },
    series: year({ rainDay }),
  }))
  const t = fitTrigger(years)
  assert.equal(t.onset.basis, 'rain30')
  assert.ok(t.onset.error < 4)
})

test('fitTrigger ends on frost when the season stops after a hard frost', () => {
  const frosts = [285, 300, 310, 295, 305]
  const years = frosts.map((frostDay, i) => ({
    season: { year: 2016 + i, onset: 240, end: frostDay + 1 + 5, n: 10 },
    series: year({ frostDay }),
  }))
  const t = fitTrigger(years)
  assert.equal(t.end.basis, 'frost')
  assert.equal(t.end.value, 5)
  assert.ok(t.end.error < 1)
})

test('fitTrigger falls back to the calendar when weather explains nothing', () => {
  const years = [0, 1, -1, 2].map((warm, i) => ({
    season: { year: 2019 + i, onset: 250, end: 290, n: 10 },
    series: year({ warm }),
  }))
  const t = fitTrigger(years)
  assert.equal(t.onset.basis, 'doy')
  assert.equal(t.onset.value, 250)
  assert.equal(fitTrigger(years.slice(0, 2)), null)
})

test('buildScenarios: observed, then forecast member, then analog', () => {
  const current = emptySeries()
  current.rain[0] = 1
  const member = { rain: [9, 9], tmax: [20, 20], tmin: [5, null] }
  const analog = year()
  const [s] = buildScenarios({ current, members: [member], analogs: [analog], today: 2 })
  assert.equal(s.rain[0], 1)          // observed
  assert.equal(s.rain[1], 9)          // forecast day 0
  assert.equal(s.tmin[2], analog.tmin[2])  // forecast null -> analog
  assert.equal(s.rain[5], analog.rain[5])  // past the forecast
  assert.equal(buildScenarios({ current, members: [member, member], analogs: [analog, analog, analog], today: 2 }).length, 6)
  assert.equal(buildScenarios({ current, members: [], analogs: [analog, analog], today: 2 }).length, 2)
})

test('forecastSeason: a warm forecast brings a GDD season forward', () => {
  const years = [-2, -1, 0, 1, 2].map((warm, i) => {
    const series = year({ warm })
    const onset = gddDay(series, 1500)
    return { season: { year: 2018 + i, onset, end: onset + 40, n: 20 }, series }
  })
  const t = fitTrigger(years)
  const analogs = years.map((y) => y.series)
  const today = 150
  const cool = year({ warm: -2 })
  const hot = year({ warm: 3 })
  const run = (current) => forecastSeason(t, buildScenarios({ current, members: [], analogs, today }), today)
  const early = run(hot)
  const late = run(cool)
  assert.equal(early.status, 'upcoming')
  assert.ok(early.onset.p50 < late.onset.p50)
  assert.ok(early.onset.p10 <= early.onset.p50 && early.onset.p50 <= early.onset.p90)
  assert.equal(early.scenarios, 5)
})

test('forecastSeason: finds this year mean the season has started', () => {
  const years = [0, 1, 2].map((warm, i) => ({ season: { year: 2020 + i, onset: 250, end: 290, n: 9 }, series: year({ warm }) }))
  const t = fitTrigger(years)
  const sc = buildScenarios({ current: year(), members: [], analogs: years.map((y) => y.series), today: 245 })
  const o = forecastSeason(t, sc, 245, { n: 3, first: 240, last: 244 })
  assert.equal(o.status, 'in')
  assert.equal(o.onset.p50, 240)
  assert.equal(o.pNext7, 1)
})

test('forecastSeason: past the usual end reads as over', () => {
  const years = [0, 1, 2].map((warm, i) => ({ season: { year: 2020 + i, onset: 200, end: 230, n: 9 }, series: year({ warm }) }))
  const t = fitTrigger(years)
  const sc = buildScenarios({ current: year(), members: [], analogs: years.map((y) => y.series), today: 280 })
  const o = forecastSeason(t, sc, 280)
  assert.equal(o.status, 'over')
  assert.equal(o.pNext14, 0)
})

test('localSeasons: per-species years near a place, current year apart', () => {
  const rows = []
  const add = (species, date, lat = 40, lon = -105) => rows.push({ species, date, lat, lon })
  for (let d = 1; d <= 10; d += 1) add('A', `2022-08-${String(d + 10).padStart(2, '0')}`)
  for (let d = 1; d <= 2; d += 1) add('A', `2023-08-${String(d + 10).padStart(2, '0')}`)   // too few
  add('A', '2026-08-20'); add('A', '2026-08-22')
  add('A', '2022-08-15', 45, -105)   // far away
  const out = localSeasons(rows, { lat: 40, lon: -105, year: 2026 })
  const a = out.get('A')
  assert.equal(a.past.length, 1)
  assert.equal(a.past[0].year, 2022)
  assert.equal(a.past[0].n, 10)
  assert.ok(a.past[0].onset < a.past[0].end)
  assert.deepEqual(a.current, { n: 2, first: 232, last: 234 })
})

test('forecastBias and currentToDate fill the days ERA5 has not reached', () => {
  const observed = emptySeries()
  for (let d = 100; d < 120; d += 1) { observed.tmax[d - 1] = 20; observed.tmin[d - 1] = 5; observed.rain[d - 1] = 2 }
  const recent = { rain: Array(30).fill(1), tmax: Array(30).fill(18), tmin: Array(30).fill(7) }
  const b = forecastBias(observed, recent, 100)
  assert.equal(b.tmax, 2)
  assert.equal(b.tmin, -2)
  assert.equal(b.rainRatio, 2)
  const fixed = correct(recent, b)
  assert.equal(fixed.tmax[0], 20)
  const cur = currentToDate(observed, recent, 100, 125, b)
  assert.equal(cur.tmax[118], 20)      // observed kept
  assert.equal(cur.tmax[121], 20)      // filled, corrected
  assert.equal(cur.rain[121], 2)
  assert.equal(cur.tmax[125], null)    // today onward left for the forecast
  assert.deepEqual(forecastBias(emptySeries(), recent, 100), { tmax: 0, tmin: 0, rainRatio: 1, days: 0 })
})

test('snapCell snaps to 0.1 degree centres and rejects junk', () => {
  assert.deepEqual(snapCell('40.0123', '-105.27'), { lat: 40.05, lon: -105.25 })
  assert.throws(() => snapCell('x', 1))
  assert.throws(() => snapCell(91, 1))
})

test('seriesFromRegion reads getRegion rows into days of the year', () => {
  const t = (iso) => Date.parse(`${iso}T00:00:00Z`)
  const rows = [
    ['id', 'longitude', 'latitude', 'time', 'temperature_2m_max'],
    ['a', 0, 0, t('2024-01-01'), 283.15],
    ['b', 0, 0, t('2024-12-31'), null],
    ['c', 0, 0, t('2024-03-01'), 293.15],
  ]
  const s = seriesFromRegion(rows, 2024, [['temperature_2m_max', 'tmax', (v) => v - 273.15]])
  assert.equal(s.tmax[0], 10)
  assert.equal(s.tmax[60], 20)     // leap year: Mar 1 is day 61
  assert.equal(s.tmax[365], null)
})

test('dailyByMember: 6-hour rain goes to the day its period ends in; temps give max and min', () => {
  const init = Date.parse('2026-10-15T00:00:00Z')
  const recs = [6, 12, 18, 24, 30].map((h, i) => ['3', h, init, 0.001, 273.15 + i])
  const byDay = dailyByMember(recs)['3']
  const day0 = Date.parse('2026-10-15T00:00:00Z')
  const day1 = day0 + 86400000
  assert.equal(byDay[day0].periods, 4)       // 06, 12, 18, and the period ending at midnight
  assert.ok(Math.abs(byDay[day0].rain - 4) < 1e-9)
  assert.equal(byDay[day0].tmax, 2)          // samples at 06, 12, 18
  assert.equal(byDay[day1].tmin, 3)          // midnight sample belongs to the next day
  const arr = memberArrays(byDay, day0, 2)
  assert.equal(arr.rain[0], 4)
  assert.equal(arr.rain[1], null)            // one period only: incomplete
  assert.deepEqual(ensembleMean([arr, { ...arr, tmax: [4, null] }]).tmax, [3, 4])
})

test('season-weather handler: validates, caches past years, and passes forecast through', async () => {
  const calls = []
  const store = new Map()
  const deps = {
    now: Date.parse('2026-10-15T10:00:00Z'),
    eeReady: () => true,
    store: {
      get: async (k) => store.get(k) ?? null,
      setJSON: async (k, v) => { store.set(k, v) },
    },
    runner: {
      observedYear: async (a) => { calls.push(['year', a.year]); return emptySeries() },
      forecastAt: async () => { calls.push(['fc']); return { init: 'x', members: [], recent: {} } },
    },
  }
  const get = (q) => handleSeasonWeather(new Request(`http://x/f?${q}`), deps)

  let res = await get('lat=40&lon=-105&part=year&year=2019')
  assert.equal(res.status, 200)
  res = await get('lat=40.01&lon=-104.92&part=year&year=2019')
  assert.equal((await res.json()).cached, true)
  assert.deepEqual(calls, [['year', 2019]])
  assert.equal(store.get('year/40.05_-104.95/2019').expires, null)

  await get('lat=40&lon=-105&part=year&year=2026')
  assert.ok(store.get('year/40.05_-104.95/2026').expires > deps.now)

  res = await get('lat=40&lon=-105&part=forecast')
  assert.equal((await res.json()).init, 'x')

  assert.equal((await get('lat=40&lon=-105&part=year&year=1900')).status, 400)
  assert.equal((await get('lat=abc&lon=-105&part=forecast')).status, 400)
  assert.equal((await get('lat=40&lon=-105&part=nope')).status, 400)
  assert.equal((await handleSeasonWeather(new Request('http://x/f?lat=40&lon=-105&part=forecast'), { ...deps, eeReady: () => false })).status, 503)
})

test('outlookCandidates: a named species is kept whatever its dates; outlookSpecies lists the forecastable ones', async () => {
  const { outlookCandidates, outlookSpecies } = await import('../composables/useSeasonOutlook.js')
  const seasons = new Map([
    ['Early', { past: [1, 2, 3].map((i) => ({ year: 2020 + i, onset: 100, end: 120, n: 5 })), current: null }],
    ['Late', { past: [1, 2, 3].map((i) => ({ year: 2020 + i, onset: 280, end: 310, n: 9 })), current: null }],
    ['Sparse', { past: [{ year: 2022, onset: 280, end: 300, n: 5 }], current: null }],
  ])
  assert.deepEqual(outlookCandidates(seasons, 288).map((s) => s.name), ['Late'])
  assert.deepEqual(outlookCandidates(seasons, 288, { species: 'Early' }).map((s) => s.name), ['Early'])
  assert.deepEqual(outlookCandidates(seasons, 288, { species: 'Sparse' }), [])
  assert.deepEqual(outlookSpecies(seasons), ['Early', 'Late'])
})
