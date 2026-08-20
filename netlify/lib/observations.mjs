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

export function inatResultToFeature(obs) {
  const coords = obs?.geojson?.coordinates
  if (!coords || coords[0] == null || coords[1] == null) return null
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [coords[0], coords[1]] },
    properties: {
      uuid: obs.uuid ?? null,
      species: obs.taxon?.name ?? null,
      date: obs.observed_on ?? null,
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
