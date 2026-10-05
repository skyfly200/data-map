// Access data ingestion (WANT-17 phase 4): PAD-US polygons and OSM ways ->
// rows for access_areas / access_lines. Pure; fetching is the caller's job so
// this runs against fixtures.

const TRAIL = new Set(['path', 'footway', 'track', 'bridleway', 'cycleway'])
const ROAD = new Set(['residential', 'unclassified', 'tertiary', 'secondary', 'primary', 'service', 'living_street'])

/** PAD-US Pub_Access: Open / Restricted / Closed / Unknown. */
export function padusAccessClass(v) {
  if (v === 'Open') return 'open'
  if (v === 'Closed') return 'closed'
  if (v === 'Restricted') return 'restricted'
  return 'unknown'
}

// ---- Classification: ESTIMATES only ---------------------------------------
// Everything below is a heuristic from manager type / designation. It is NOT a
// statement of any regulation. Rules vary by unit, season and permit; the
// *_source columns say 'estimated' so consumers can label it. Collecting is
// never estimated as 'allowed'. PAD-US Mang_Name/Des_Tp codes are mapped by
// pattern and have not been verified against a live PAD-US load.

/** Normalised manager type from PAD-US Mang_Name (agency code) / Mang_Type. */
export function managerType(mangName, mangType) {
  const n = String(mangName || '').trim().toUpperCase()
  const t = String(mangType || '').trim().toUpperCase()
  if (n === 'BLM') return 'blm'
  if (n === 'USFS' || n === 'FS') return 'usfs'
  if (n === 'NPS') return 'nps'
  if (n === 'FWS') return 'fws'
  if (n === 'SPR' || n === 'SPRK' || /STATE PARK/.test(n)) return 'state_park'
  if (n === 'TRIB' || t === 'TRIB') return 'tribal'
  if (n === 'PVT' || n === 'NGO' || t === 'PVT' || t === 'NGO') return 'private'
  if (t === 'STAT') return 'state'
  if (t === 'LOC' || n === 'CITY' || n === 'CNTY') return 'local'
  if (t === 'FED') return 'federal_other'
  return n || t ? 'other' : 'unknown'
}

/** Fee estimate -> { fee_status, fee_source }. Source is null when nothing is known. */
export function estimateFee({ managerType: mt, designation }) {
  const des = String(designation || '').toUpperCase()
  // Heuristic: dispersed use on BLM/USFS land is generally free, though
  // developed sites on that land often charge; we cannot see sites here.
  if (mt === 'blm' || mt === 'usfs') return { fee_status: 'free', fee_source: 'estimated' }
  if (mt === 'nps' || mt === 'state_park' || des === 'SP' || des === 'SRA') {
    return { fee_status: 'fee', fee_source: 'estimated' }
  }
  // Recreation areas, refuges, local, private etc.: too variable to guess.
  return { fee_status: 'unknown', fee_source: null }
}

/** Collecting estimate -> { collecting, collecting_source }. Never 'allowed'. */
export function estimateCollecting({ managerType: mt, designation, access }) {
  const des = String(designation || '').toUpperCase()
  if (access === 'closed') return { collecting: 'prohibited', collecting_source: 'estimated' } // closed to the public
  // Heuristic: parks/refuges/wilderness-type units tend to restrict or bar
  // collecting; the actual rule needs checking with the manager.
  if (mt === 'nps' || mt === 'state_park' || mt === 'fws' || des === 'SP' || des === 'WA' || des === 'WSA') {
    return { collecting: 'restricted', collecting_source: 'estimated' }
  }
  if (access === 'restricted') return { collecting: 'restricted', collecting_source: 'estimated' }
  return { collecting: 'unknown', collecting_source: null }
}

/** All estimated classification columns for one area. */
export function classifyArea({ mangName, mangType, designation, access }) {
  const mt = managerType(mangName, mangType)
  return {
    manager_type: mt,
    ...estimateFee({ managerType: mt, designation }),
    ...estimateCollecting({ managerType: mt, designation, access }),
  }
}

const toMulti =(g) => (g?.type === 'Polygon' ? { type: 'MultiPolygon', coordinates: [g.coordinates] }
  : g?.type === 'MultiPolygon' ? g : null)

/** GeoJSON FeatureCollection (PAD-US export) -> access_areas rows. Skips non-polygons and unnamed units. */
export function parsePadUs(fc) {
  const rows = []
  for (const f of fc?.features || []) {
    const p = f.properties || {}
    const geom = toMulti(f.geometry)
    if (!geom || !p.Unit_Nm) continue
    rows.push({
      source: 'padus', source_id: String(p.OBJECTID ?? `${p.Unit_Nm}|${p.Des_Tp ?? ''}`),
      name: p.Unit_Nm, manager: p.Mang_Name ?? null, designation: p.Des_Tp ?? null,
      access_class: padusAccessClass(p.Pub_Access), public_access: padusAccessClass(p.Pub_Access), geom,
      ...classifyArea({ mangName: p.Mang_Name, mangType: p.Mang_Type, designation: p.Des_Tp, access: padusAccessClass(p.Pub_Access) }),
    })
  }
  return rows
}

/** Overpass `out geom` JSON -> access_lines rows. Skips private/no-access ways and unknown highway types. */
export function parseOsmWays(overpass) {
  const rows = []
  for (const el of overpass?.elements || []) {
    if (el.type !== 'way' || !el.geometry || el.geometry.length < 2) continue
    const t = el.tags || {}
    const kind = TRAIL.has(t.highway) ? 'trail' : ROAD.has(t.highway) ? 'road' : null
    if (!kind) continue
    if (t.access === 'private' || t.access === 'no') continue
    rows.push({
      osm_id: el.id, kind, highway: t.highway, name: t.name ?? null,
      geom: { type: 'LineString', coordinates: el.geometry.map((n) => [n.lon, n.lat]) },
    })
  }
  return rows
}

const M_PER_DEG = 111_320
/** Distance in metres from a point to a set of lines (equirectangular; fine at foray scale). */
export function distanceToLinesM([lon, lat], lines) {
  const k = Math.cos((lat * Math.PI) / 180)
  let best = Infinity
  for (const ln of lines) {
    const c = ln.geom.coordinates
    for (let i = 0; i < c.length - 1; i++) {
      const ax = (c[i][0] - lon) * k, ay = c[i][1] - lat
      const bx = (c[i + 1][0] - lon) * k, by = c[i + 1][1] - lat
      const dx = bx - ax, dy = by - ay
      const len2 = dx * dx + dy * dy
      const t = len2 ? Math.max(0, Math.min(1, -(ax * dx + ay * dy) / len2)) : 0
      best = Math.min(best, Math.hypot(ax + t * dx, ay + t * dy))
    }
  }
  return best * M_PER_DEG
}

/** Ray-cast point-in-multipolygon (outer rings only; holes ignored). */
export function pointInArea([lon, lat], area) {
  for (const poly of area.geom.coordinates) {
    const ring = poly[0]
    let inside = false
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j]
      if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside
    }
    if (inside) return true
  }
  return false
}

/** Access summary for a cell centre: legal class + nearest road/trail. */
export function accessFor(point, areas, lines) {
  const hit = areas.find((a) => pointInArea(point, a))
  return {
    legal: hit ? hit.access_class : 'unknown',
    publicLand: hit ? hit.access_class === 'open' : false,
    trailM: distanceToLinesM(point, lines.filter((l) => l.kind === 'trail')),
    roadM: distanceToLinesM(point, lines.filter((l) => l.kind === 'road')),
  }
}

// ---- Recreation.gov RIDB fee overlay ---------------------------------------
// RIDB facility records -> upgrade fee_status to source 'ridb' on areas whose
// name matches and which contain/adjoin the facility. Needs env RIDB_API_KEY;
// without it callers skip the overlay. Field names follow the public RIDB
// /facilities schema as documented; not verified against a live response.

/** RIDB /facilities JSON -> [{ id, name, lon, lat, fee: 'fee'|'free'|null, feeText }]. */
export function parseRidbFacilities(json) {
  const out = []
  for (const f of json?.RECDATA || []) {
    const lat = Number(f.FacilityLatitude), lon = Number(f.FacilityLongitude)
    if (!f.FacilityName || !Number.isFinite(lat) || !Number.isFinite(lon) || (!lat && !lon)) continue
    const text = String(f.FacilityUseFeeDescription || '').trim()
    let fee = null
    if (text) {
      if (/\$\s*\d/.test(text)) fee = 'fee'
      else if (/\b(no fee|free)\b/i.test(text)) fee = 'free'
      else if (/\d/.test(text)) fee = 'fee'
    }
    out.push({ id: String(f.FacilityID), name: f.FacilityName, lon, lat, fee, feeText: text || null })
  }
  return out
}

const STOP = new Set(['the', 'of', 'and', 'national', 'forest', 'area', 'state', 'park', 'recreation', 'campground', 'trailhead'])
const tokens = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').split(/\s+/).filter((w) => w && !STOP.has(w))

/** Name match: substring either way, or >= half of the smaller distinctive-token set shared. */
export function namesMatch(a, b) {
  const x = String(a || '').toLowerCase(), y = String(b || '').toLowerCase()
  if (!x || !y) return false
  if (x.includes(y) || y.includes(x)) return true
  const ta = new Set(tokens(a)), tb = tokens(b)
  if (!ta.size || !tb.length) return false
  const shared = tb.filter((w) => ta.has(w)).length
  return shared > 0 && shared / Math.min(ta.size, tb.length) >= 0.5
}

function nearArea(point, area, maxM) {
  if (pointInArea(point, area)) return true
  for (const poly of area.geom.coordinates) {
    if (distanceToLinesM(point, [{ geom: { coordinates: poly[0] } }]) <= maxM) return true
  }
  return false
}

/**
 * Upgrade fee_status/fee_source to 'ridb' for areas matched by facility name
 * AND proximity (inside, or within maxM). Name is required so one campground
 * does not mark a whole national forest. Mutates areas; returns { matched }.
 */
export function applyRidbFees(areas, facilities, { maxM = 2000 } = {}) {
  let matched = 0
  for (const a of areas) {
    for (const f of facilities) {
      if (!f.fee || !namesMatch(a.name, f.name) || !nearArea([f.lon, f.lat], a, maxM)) continue
      a.fee_status = f.fee
      a.fee_source = 'ridb'
      matched++
      break
    }
  }
  return { matched }
}

// ---- Fetch builders (the network is the caller's) --------------------------

/** Split [w,s,e,n] into tiles of at most `deg` degrees. */
export function tileBbox([w, s, e, n], deg) {
  const tiles = []
  for (let y = s; y < n - 1e-9; y += deg) {
    for (let x = w; x < e - 1e-9; x += deg) {
      tiles.push([x, y, Math.min(x + deg, e), Math.min(y + deg, n)].map((v) => Math.round(v * 1e6) / 1e6))
    }
  }
  return tiles
}

export const OVERPASS_URL = 'https://overpass-api.de/api/interpreter'
// Unverified default; override with env PADUS_FEATURE_URL (an ArcGIS FeatureServer layer URL).
export const PADUS_FEATURE_URL = 'https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/Manager_Name/FeatureServer/0'
export const RIDB_URL = 'https://ridb.recreation.gov/api/v1/facilities'

export function overpassQuery([w, s, e, n]) {
  const hw = [...TRAIL, ...ROAD].join('|')
  return `[out:json][timeout:60];way["highway"~"^(${hw})$"](${s},${w},${n},${e});out geom tags;`
}

export function padusQueryUrl(base, [w, s, e, n], offset = 0) {
  const q = new URLSearchParams({
    where: '1=1', geometry: `${w},${s},${e},${n}`, geometryType: 'esriGeometryEnvelope', inSR: '4326',
    spatialRel: 'esriSpatialRelIntersects', outFields: 'OBJECTID,Unit_Nm,Mang_Name,Mang_Type,Des_Tp,Pub_Access',
    outSR: '4326', f: 'geojson', resultOffset: String(offset), resultRecordCount: '500',
  })
  return `${base}/query?${q}`
}

export function ridbUrl([w, s, e, n], offset = 0, limit = 50) {
  const lat = (s + n) / 2, lon = (w + e) / 2
  const radiusMi = Math.min(100, Math.ceil(Math.hypot((n - s) * 69, (e - w) * 69 * Math.cos((lat * Math.PI) / 180)) / 2))
  return `${RIDB_URL}?${new URLSearchParams({ latitude: String(lat), longitude: String(lon), radius: String(radiusMi), limit: String(limit), offset: String(offset) })}`
}
