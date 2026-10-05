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

const toMulti = (g) => (g?.type === 'Polygon' ? { type: 'MultiPolygon', coordinates: [g.coordinates] }
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
      access_class: padusAccessClass(p.Pub_Access), geom,
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
