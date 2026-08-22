// Shared observation data + display helpers for every view (map, table, charts).
// Data is fetched once on the client and cached in Nuxt state across pages.

// Validated colour-blind-safe categorical palette (worst adjacent CVD ΔE 9.1),
// indexed by cluster id. Identity is never colour-alone: the map has a legend,
// popups name the cluster, charts direct-label, and a full table view exists.
export const PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
                        '#e87ba4', '#008300', '#4a3aa7', '#e34948']
export const UNCLUSTERED = '#9aa0a6'
export const SERIES_1 = '#2a78d6'  // single-series charts

export function colorFor(cluster) {
  if (cluster === null || cluster === undefined || Number.isNaN(cluster)) return UNCLUSTERED
  return PALETTE[cluster % PALETTE.length]
}

// Canonical iNaturalist URL. Prefers the numeric id; falls back to the UUID,
// which the iNaturalist observations route also resolves.
export function inatUrl(p) {
  const id = p?.inat_id ?? p?.uuid
  return id ? `https://www.inaturalist.org/observations/${id}` : null
}

const num1 = (v) => Number(v).toFixed(1)
const num2 = (v) => Number(v).toFixed(2)
const num3 = (v) => Number(v).toFixed(3)

// Enriched attributes shown in the popup and table. key, label, formatter.
export const FIELDS = [
  ['date', 'Observed', (v) => v],
  ['elevation', 'Elevation', (v) => `${Math.round(v)} m`],
  ['land_cover_label', 'Land cover', (v) => v],
  ['ndvi', 'NDVI', num3],
  ['soil_moisture', 'Soil moisture', num3],
  ['solar_exposure', 'Solar exposure', num2],
  ['wind_exposure', 'Wind exposure', num2],
  ['water_retention', 'Water retention', num2],
  ['slope', 'Slope', (v) => `${num1(v)}°`],
  ['aspect', 'Aspect', (v) => `${num1(v)}°`],
]

export function hasValue(v) {
  return v !== null && v !== undefined && v !== ''
}

async function fetchObservations() {
  // Prefer the live function (blob-backed, includes interim new sightings);
  // fall back to the committed static file if the function isn't available.
  try {
    const res = await fetch('/.netlify/functions/observations')
    if (res.ok) return await res.json()
  } catch {
    // fall through
  }
  const res = await fetch('/data/observations.geojson')
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return await res.json()
}

export function useObservations() {
  const data = useState('observations', () => null)
  const error = useState('observations-error', () => '')
  const pending = useState('observations-pending', () => false)

  async function load() {
    if (data.value || pending.value) return
    pending.value = true
    try {
      data.value = await fetchObservations()
    } catch (e) {
      error.value = e.message
    } finally {
      pending.value = false
    }
  }

  // Convenience: a flat array of property objects (with lon/lat attached).
  const rows = computed(() => (data.value?.features || []).map((f) => ({
    ...f.properties,
    lon: f.geometry?.coordinates?.[0],
    lat: f.geometry?.coordinates?.[1],
  })))

  return { data, rows, error, pending, load }
}
