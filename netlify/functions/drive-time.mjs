// Drive-time area for the foray planner (isochrone).
//
//   GET /.netlify/functions/drive-time?lat=40.015&lon=-105.27&min=45
//
// Response (200): { ok: true, source: 'ors' | 'valhalla', minutes, geom: MultiPolygon }
// Errors: { ok:false, error } with 400 (bad query) / 502 (no routing service answered).
// The planner falls back to straight-line distance on any error.

import { ORS_MAX_MIN, ORS_URL, orsRequest, parseDriveQuery, toMultiPolygon, valhallaUrl } from '../lib/drive-time.mjs'

const json = (body, status = 200, cache = 'no-store') => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': cache },
})

async function fetchJson(url, init, ms = 15000) {
  const ctl = new AbortController()
  const t = setTimeout(() => ctl.abort(), ms)
  try {
    const res = await fetch(url, { ...init, signal: ctl.signal })
    return res.ok ? await res.json() : null
  } catch { return null } finally { clearTimeout(t) }
}

export async function handleDriveTime(request, env = process.env, doFetch = fetchJson) {
  let q
  try {
    q = parseDriveQuery(new URL(request.url).searchParams)
  } catch (err) {
    return json({ ok: false, error: err.message }, 400)
  }
  const key = env.ORS_API_KEY
  if (key && q.minutes <= ORS_MAX_MIN) {
    const geom = toMultiPolygon(await doFetch(ORS_URL, {
      method: 'POST',
      headers: { Authorization: key, 'content-type': 'application/json', Accept: 'application/geo+json' },
      body: JSON.stringify(orsRequest(q)),
    }))
    if (geom) return json({ ok: true, source: 'ors', minutes: q.minutes, geom }, 200, 'public, max-age=86400')
  }
  const geom = toMultiPolygon(await doFetch(valhallaUrl(q), { headers: { Accept: 'application/json' } }))
  if (geom) return json({ ok: true, source: 'valhalla', minutes: q.minutes, geom }, 200, 'public, max-age=86400')
  return json({ ok: false, error: 'No routing service answered.' }, 502)
}

export default (request) => handleDriveTime(request)
