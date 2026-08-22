// Pure, testable helpers for building the observations GeoJSON.
//
// The scheduled Netlify function uses these to fetch fresh iNaturalist
// sightings and merge them onto the fully-enriched baseline that the offline
// Python pipeline commits. No Netlify or filesystem imports here so the logic
// can be unit-tested in isolation.

const INAT_API = 'https://api.inaturalist.org/v1/observations'

// Property shape shared with the Python export (export_geojson.py). New points
// from the light Node refresh start with the satellite-derived fields null;
// they are filled in on the next GitHub Action run of the full pipeline.
const EMPTY_PROPS = {
  elevation: null,
  land_cover_label: null,
  ndvi: null,
  soil_moisture: null,
  solar_exposure: null,
  wind_exposure: null,
  water_retention: null,
  slope: null,
  aspect: null,
  cluster: null,
}

export function buildInatUrl({
  taxonName = 'morchella', lat = 40.0, lng = -105.0, radius = 500,
  perPage = 100, qualityGrade = 'research',
} = {}) {
  const params = new URLSearchParams({
    taxon_name: taxonName,
    lat: String(lat),
    lng: String(lng),
    radius: String(radius),
    quality_grade: qualityGrade,
    geo: 'true',
    captive: 'false',
    per_page: String(perPage),
    order: 'desc',
    order_by: 'created_at',
  })
  return `${INAT_API}?${params.toString()}`
}

function dayOfYear(dateStr) {
  if (!dateStr) return null
  const d = new Date(`${dateStr}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return null
  const start = Date.UTC(d.getUTCFullYear(), 0, 0)
  return Math.floor((d.getTime() - start) / 86400000)
}

export function inatResultToFeature(obs) {
  const coords = obs?.geojson?.coordinates
  if (!coords || coords[0] == null || coords[1] == null) return null
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [coords[0], coords[1]] },
    properties: {
      uuid: obs.uuid ?? null,
      inat_id: obs.id ?? null,
      species: obs.taxon?.name ?? null,
      date: obs.observed_on ?? null,
      day_of_year: dayOfYear(obs.observed_on),
      location: obs.place_guess ?? null,
      num_identification_agreements: obs.num_identification_agreements ?? null,
      ...EMPTY_PROPS,
    },
  }
}

export async function fetchInatFeatures(opts = {}, fetchImpl = fetch) {
  const res = await fetchImpl(buildInatUrl(opts))
  if (!res.ok) throw new Error(`iNaturalist HTTP ${res.status}`)
  const data = await res.json()
  return (data.results || []).map(inatResultToFeature).filter(Boolean)
}

export function baselineUuidSet(baseline) {
  return new Set((baseline?.features || []).map((f) => f.properties?.uuid).filter(Boolean))
}

// Fresh features whose uuid is not already in the (authoritative) baseline.
export function newFeatures(baseline, fresh) {
  const seen = baselineUuidSet(baseline)
  return fresh.filter((f) => {
    const id = f.properties?.uuid
    return id && !seen.has(id)
  })
}

// Baseline is authoritative: return it plus any extras not already present.
// This keeps the committed, fully-enriched data (incl. clustering) winning over
// an older blob — the blob only contributes genuinely-new interim sightings.
export function overlay(baseline, extras) {
  const seen = baselineUuidSet(baseline)
  const add = (extras || []).filter((f) => {
    const id = f.properties?.uuid
    return !id || !seen.has(id)
  })
  return { type: 'FeatureCollection', features: [...(baseline?.features || []), ...add] }
}

// Add only the fresh features whose uuid is not already in the baseline.
export function mergeByUuid(baseline, fresh) {
  const features = [...(baseline?.features || [])]
  const seen = new Set(features.map((f) => f.properties?.uuid).filter(Boolean))
  const addedUuids = []
  for (const f of fresh) {
    const id = f.properties?.uuid
    if (id && seen.has(id)) continue
    if (id) seen.add(id)
    features.push(f)
    addedUuids.push(id)
  }
  return {
    collection: { type: 'FeatureCollection', features },
    added: addedUuids.length,
    addedUuids,
  }
}
