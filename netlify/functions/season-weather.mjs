// Daily weather at a place, for the season outlook (WANT-19).
// Auth: Bearer token (members; it spends the shared Earth Engine quota).
//
//   GET ?lat=&lon=&part=year&year=2019
//     -> { ok, cell:{lat,lon}, year, series:{ rain[366], tmax[366], tmin[366] }, cached? }
//        Observed ERA5-Land temperature (C) and CHIRPS rain (mm) by day of year.
//        Past years are cached for good; the current year for six hours.
//   GET ?lat=&lon=&part=forecast
//     -> { ok, cell, init, start, members:[{rain[15],tmax[15],tmin[15]}],
//          recentStart, recent:{rain[30],tmax[30],tmin[30]}, cached? }
//        The latest WeatherNext 2 run (members from today, UTC) and the recent
//        short-range forecasts, for gap fill and bias. Cached per run.
//
// The point is snapped to a 0.1 degree cell so neighbours share a cache entry.
// Errors: 400 bad input, 401/403 auth, 502 Earth Engine, 503 not configured.

import { requireTier } from '../lib/auth.mjs'
import { earthEngineConfigured } from '../lib/ee-runner.mjs'
import { SeasonWeatherError, forecastAt, observedYear, snapCell } from '../lib/season-weather.mjs'
import { getStore } from '../lib/storage.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

const SIX_HOURS = 6 * 3600 * 1000
const FIRST_YEAR = 1981   // CHIRPS starts here; ERA5-Land earlier

function cache() {
  try { return getStore('season-weather') } catch { return null }
}

async function cached(store, key, now, build) {
  if (store) {
    const hit = await store.get(key, { type: 'json' }).catch(() => null)
    if (hit && (!hit.expires || hit.expires > now)) return { ...hit.value, cached: true }
  }
  const { value, ttl } = await build()
  if (store) await store.setJSON(key, { value, expires: ttl ? now + ttl : null }).catch(() => {})
  return value
}

export async function handleSeasonWeather(request, { runner, store, eeReady, now = Date.now() }) {
  if (request.method !== 'GET') return json({ ok: false, error: 'Use GET.' }, 405)
  const url = new URL(request.url)
  try {
    const cell = snapCell(url.searchParams.get('lat'), url.searchParams.get('lon'))
    const part = url.searchParams.get('part')
    const cellKey = `${cell.lat.toFixed(2)}_${cell.lon.toFixed(2)}`
    const thisYear = new Date(now).getUTCFullYear()

    if (part === 'year') {
      const year = Number(url.searchParams.get('year'))
      if (!Number.isInteger(year) || year < FIRST_YEAR || year > thisYear) {
        return json({ ok: false, error: `year must be ${FIRST_YEAR}–${thisYear}.` }, 400)
      }
      if (!eeReady()) return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
      const out = await cached(store, `year/${cellKey}/${year}`, now, async () => ({
        value: { series: await runner.observedYear({ ...cell, year, now }) },
        ttl: year === thisYear ? SIX_HOURS : 0,
      }))
      return json({ ok: true, cell, year, ...out })
    }

    if (part === 'forecast') {
      if (!eeReady()) return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
      // A new run lands every six hours; keying on the slot keeps one per run.
      const slot = Math.floor(now / SIX_HOURS)
      const out = await cached(store, `forecast/${cellKey}/${slot}`, now, async () => ({
        value: await runner.forecastAt({ ...cell, now }),
        ttl: SIX_HOURS,
      }))
      return json({ ok: true, cell, ...out })
    }

    return json({ ok: false, error: 'part must be year or forecast.' }, 400)
  } catch (err) {
    if (err instanceof SeasonWeatherError) return json({ ok: false, error: err.message }, err.status)
    return json({ ok: false, error: String(err?.message || err).slice(0, 300) }, err?.status || 502)
  }
}

export default async function handler(request) {
  const auth = await requireTier(request, 'member', { message: 'Season outlooks are a membership benefit.' })
  if (!auth.ok) return auth.response
  return handleSeasonWeather(request, {
    runner: { observedYear, forecastAt },
    store: cache(),
    eeReady: earthEngineConfigured,
  })
}
