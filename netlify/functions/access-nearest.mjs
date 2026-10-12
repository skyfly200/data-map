// Walk-in estimate for the foray planner (WANT-17). Public reference data (OSM).
//
//   POST /.netlify/functions/access-nearest  { points: [[lon, lat], ...] }   (max 1500 points)
//
// Response (200):
//   {
//     ok: true,
//     max_m: 5000,
//     results: [{ i, parking, accessible, trailhead, road_m }]   // one per point, in order
//       parking / accessible / trailhead: { m, name, lon, lat[, fee] } | null  (nearest within max_m)
//       accessible: parking tagged wheelchair=yes or with disabled spaces
//       road_m: metres to the nearest public road line, or null beyond max_m
//   }
// Nulls mean "nothing within max_m" OR "not loaded here"; the planner says which
// using the access regions it already has.
// Errors: { ok:false, error } with 400 (bad body) / 503 (no database).

import { adminClient } from '../lib/auth.mjs'

export const MAX_POINTS = 1500
export const NEAREST_MAX_M = 5000

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

/** Validate `[[lon, lat], ...]`; throws a message on bad input. Coordinates are rounded to ~1 m. */
export function parsePoints(points) {
  if (!Array.isArray(points) || !points.length) throw new Error('Provide points: [[lon, lat], ...].')
  if (points.length > MAX_POINTS) throw new Error(`At most ${MAX_POINTS} points per request.`)
  return points.map((p, i) => {
    const lon = Number(p?.[0]), lat = Number(p?.[1])
    if (!Number.isFinite(lon) || !Number.isFinite(lat) || lon < -180 || lon > 180 || lat < -90 || lat > 90) {
      throw new Error(`Point ${i} is not a [lon, lat] pair.`)
    }
    return [Math.round(lon * 1e5) / 1e5, Math.round(lat * 1e5) / 1e5]
  })
}

export async function handleAccessNearest(request, client = adminClient()) {
  if (request.method !== 'POST') return json({ ok: false, error: 'Use POST.' }, 405)
  const body = await request.json().catch(() => null)
  let points
  try {
    points = parsePoints(body?.points)
  } catch (err) {
    return json({ ok: false, error: err.message }, 400)
  }
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)
  const { data, error } = await client.rpc('access_nearest', { p_points: points, p_max_m: NEAREST_MAX_M })
  if (error) return json({ ok: false, error: error.message }, 500)
  const byI = new Map((data || []).map((r) => [Number(r.i), r]))
  const results = points.map((_, i) => {
    const r = byI.get(i) || {}
    return { i, parking: r.parking ?? null, accessible: r.accessible ?? null, trailhead: r.trailhead ?? null, road_m: r.road_m ?? null }
  })
  return json({ ok: true, max_m: NEAREST_MAX_M, results })
}

export default (request) => handleAccessNearest(request)
