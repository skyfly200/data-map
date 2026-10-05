// Access database read endpoint (WANT-17). Public reference data (PAD-US + OSM).
//
//   GET /.netlify/functions/access?bbox=w,s,e,n[&free=1][&public=1][&collecting=allowed|likely_allowed|restricted|prohibited|unknown][&lines=1]
//
// Response (200):
//   {
//     ok: true,
//     bbox: [w, s, e, n],
//     areas: GeoJSON FeatureCollection; geometry simplified (Polygon/MultiPolygon);
//       feature.properties = { id, name, manager_type, public_access, fee_status,
//         fee_source, collecting, collecting_source }
//       fee_status 'free'|'fee'|'unknown'; fee_source 'estimated'|'ridb'|null
//       collecting 'allowed'|'likely_allowed'|'restricted'|'prohibited'|'unknown'; collecting_source 'estimated'|null
//       public_access 'open'|'restricted'|'closed'|'unknown'
//       'estimated' values are heuristics from manager type, not regulations: label them as such.
//     lines: GeoJSON FeatureCollection | null; properties = { id, kind: 'road'|'trail', highway, name }.
//       Only with lines=1 AND a bbox <= 0.5 deg per side; otherwise null with lines_omitted set.
//     lines_omitted: 'not_requested' | 'bbox_too_large' | null
//     truncated: { areas: boolean, lines: boolean }   // true when the row cap was hit
//     regions: [{ name, status, bbox, loaded_at }]     // regions overlapping the bbox
//     loaded: boolean   // true when a region with status 'loaded' fully covers the bbox
//   }
// Filters: free=1 -> fee_status 'free' only; public=1 -> public_access 'open' only;
//   collecting=<value> -> exact match. Caps: bbox <= 3 deg per side, 400 areas, 3000 lines.
// Errors: { ok:false, error } with 400 (bad params) / 503 (no database).

import { adminClient } from '../lib/auth.mjs'
import { findCoveringRegion } from '../lib/access-regions.mjs'

export const MAX_BBOX_DEG = 3
export const LINES_MAX_BBOX_DEG = 0.5
export const MAX_AREAS = 400
export const MAX_LINES = 3000
const COLLECTING = ['allowed', 'likely_allowed', 'restricted', 'prohibited', 'unknown']

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'public, max-age=300' },
})

const features = (rows, props) => ({
  type: 'FeatureCollection',
  features: rows.map((r) => ({ type: 'Feature', geometry: r.geometry, properties: props(r) })),
})

const overlaps = (a, b) => a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]

export async function handleAccess(request, client = adminClient()) {
  if (request.method !== 'GET') return json({ ok: false, error: 'Use GET.' }, 405)
  const q = new URL(request.url).searchParams
  const bbox = String(q.get('bbox') || '').split(',').map(Number)
  const [w, s, e, n] = bbox
  if (bbox.length !== 4 || bbox.some((v) => !Number.isFinite(v)) || !(w < e && s < n)
    || w < -180 || e > 180 || s < -90 || n > 90) {
    return json({ ok: false, error: 'Provide bbox=west,south,east,north.' }, 400)
  }
  if (e - w > MAX_BBOX_DEG || n - s > MAX_BBOX_DEG) {
    return json({ ok: false, error: `bbox may span at most ${MAX_BBOX_DEG} degrees per side.` }, 400)
  }
  const collecting = q.get('collecting')
  if (collecting && !COLLECTING.includes(collecting)) {
    return json({ ok: false, error: `collecting must be one of ${COLLECTING.join(', ')}.` }, 400)
  }
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)

  const tol = Math.max(e - w, n - s) / 500
  const wantLines = q.get('lines') === '1'
  const linesOk = wantLines && e - w <= LINES_MAX_BBOX_DEG && n - s <= LINES_MAX_BBOX_DEG

  const [areasRes, regionsRes] = await Promise.all([
    client.rpc('access_areas_in_bbox', {
      p_w: w, p_s: s, p_e: e, p_n: n, p_free: q.get('free') === '1', p_public: q.get('public') === '1',
      p_collecting: collecting || null, p_tol: tol, p_limit: MAX_AREAS + 1,
    }),
    client.from('access_regions').select('name, status, bbox, loaded_at'),
  ])
  if (areasRes.error) return json({ ok: false, error: areasRes.error.message }, 500)
  const areaRows = areasRes.data || []

  let lineRows = [], linesTruncated = false
  if (linesOk) {
    const res = await client.rpc('access_lines_in_bbox', {
      p_w: w, p_s: s, p_e: e, p_n: n, p_tol: tol / 5, p_limit: MAX_LINES + 1,
    })
    if (res.error) return json({ ok: false, error: res.error.message }, 500)
    lineRows = res.data || []
    linesTruncated = lineRows.length > MAX_LINES
    lineRows = lineRows.slice(0, MAX_LINES)
  }

  const regions = (regionsRes.data || []).filter((r) => Array.isArray(r.bbox) && overlaps(r.bbox, [w, s, e, n]))
  return json({
    ok: true,
    bbox: [w, s, e, n],
    areas: features(areaRows.slice(0, MAX_AREAS), (r) => ({
      id: r.id, name: r.name, manager_type: r.manager_type, public_access: r.public_access,
      fee_status: r.fee_status, fee_source: r.fee_source, collecting: r.collecting,
      collecting_source: r.collecting_source,
    })),
    lines: linesOk ? features(lineRows, (r) => ({ id: r.id, kind: r.kind, highway: r.highway, name: r.name })) : null,
    lines_omitted: linesOk ? null : wantLines ? 'bbox_too_large' : 'not_requested',
    truncated: { areas: areaRows.length > MAX_AREAS, lines: linesTruncated },
    regions: regions.map(({ name, status, bbox: b, loaded_at }) => ({ name, status, bbox: b, loaded_at })),
    loaded: Boolean(findCoveringRegion(regionsRes.data, [w, s, e, n])),
  })
}

export default (request) => handleAccess(request)
