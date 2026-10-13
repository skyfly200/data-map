// Drive-time area (isochrone) for the foray planner. Pure helpers; the function
// in netlify/functions/drive-time.mjs does the fetching.
//
// OpenRouteService is used when ORS_API_KEY is set (its free tier caps driving
// isochrones at 60 minutes); otherwise, or when ORS fails, the public FOSSGIS
// Valhalla server, which needs no key. Both answer with GeoJSON polygons that
// are normalised here to one MultiPolygon.

export const DRIVE_MIN_LIMITS = { min: 10, max: 120 }
export const ORS_MAX_MIN = 60
export const VALHALLA_URL = 'https://valhalla1.openstreetmap.de/isochrone'
export const ORS_URL = 'https://api.openrouteservice.org/v2/isochrones/driving-car'

/** Query -> { lat, lon, minutes }; throws a message on bad input. Origin rounded to ~100 m so caches hit. */
export function parseDriveQuery(sp) {
  const lat = Number(sp.get('lat')), lon = Number(sp.get('lon')), minutes = Math.round(Number(sp.get('min')))
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || Math.abs(lat) > 90 || Math.abs(lon) > 180) throw new Error('Provide lat and lon.')
  if (!Number.isFinite(minutes) || minutes < DRIVE_MIN_LIMITS.min || minutes > DRIVE_MIN_LIMITS.max) {
    throw new Error(`min must be ${DRIVE_MIN_LIMITS.min}-${DRIVE_MIN_LIMITS.max}.`)
  }
  return { lat: Math.round(lat * 1e3) / 1e3, lon: Math.round(lon * 1e3) / 1e3, minutes }
}

export function valhallaUrl({ lat, lon, minutes }) {
  const body = {
    locations: [{ lat, lon }], costing: 'auto', contours: [{ time: minutes }],
    polygons: true, denoise: 0.5, generalize: 150,
  }
  return `${VALHALLA_URL}?json=${encodeURIComponent(JSON.stringify(body))}`
}

export function orsRequest({ lat, lon, minutes }) {
  return { locations: [[lon, lat]], range: [minutes * 60], range_type: 'time', smoothing: 10 }
}

/** GeoJSON FeatureCollection / Feature / geometry -> MultiPolygon, or null without a polygon. */
export function toMultiPolygon(json) {
  const geoms = []
  const add = (g) => {
    if (!g) return
    if (g.type === 'FeatureCollection') (g.features || []).forEach((f) => add(f?.geometry))
    else if (g.type === 'Feature') add(g.geometry)
    else if (g.type === 'Polygon' && Array.isArray(g.coordinates?.[0])) geoms.push(g.coordinates)
    else if (g.type === 'MultiPolygon' && Array.isArray(g.coordinates)) geoms.push(...g.coordinates)
  }
  add(json)
  const polys = geoms.filter((p) => Array.isArray(p?.[0]) && p[0].length >= 4)
  return polys.length ? { type: 'MultiPolygon', coordinates: polys } : null
}
