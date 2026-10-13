// Nearest place name for a foray suggestion, from OpenStreetMap's Nominatim
// reverse geocoder. Pure label building here; usePlaceNames does the fetching
// (one request a second, per Nominatim's usage policy) and caching.

/** Cache key for a point: about 100 m, so neighbouring lookups share a name. */
export const placeKey = (lat: number, lon: number) => `${lat.toFixed(3)},${lon.toFixed(3)}`

export function reverseUrl(lat: number, lon: number): string {
  return `https://nominatim.openstreetmap.org/reverse?format=jsonv2&zoom=14&lat=${lat.toFixed(5)}&lon=${lon.toFixed(5)}`
}

const LOCALITY = ['hamlet', 'village', 'town', 'city', 'suburb', 'locality', 'municipality', 'county']

/**
 * Nominatim reverse hit -> short name, e.g. "Brainard Lake, near Ward" or
 * "near Ward". Feature name first (a lake, peak or park), then the nearest
 * settlement; null when the hit has neither.
 */
export function placeLabel(hit: any): string | null {
  if (!hit || hit.error) return null
  const a = hit.address || {}
  const town = LOCALITY.map((k) => a[k]).find((v) => typeof v === 'string' && v.trim()) || null
  const name = typeof hit.name === 'string' && hit.name.trim() ? hit.name.trim() : null
  if (name && town && name !== town) return `${name}, near ${town}`
  if (name) return name
  return town ? `near ${town}` : null
}
