// Shared observation data + display helpers for every view (map, table, charts).
// Data is fetched once on the client and cached in Nuxt state across pages.

export const OBSERVATION_DATASETS = [
  { id: 'latest', label: 'Latest processed', path: '/mushroom_observations.geojson' },
  { id: 'baseline', label: 'Static baseline', path: '/data/observations.geojson' },
  { id: 'morchella', label: 'Morchella sample 1', path: '/mushroom_observations_morchella_40.0_-105.0_500.0_20260824T224804Z.geojson' },
  { id: 'amanita', label: 'Amanita sample', path: '/mushroom_observations_amanita_40.0_-105.0_500.0_20260824T225224Z.geojson' },
]

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
// Elevation is handled separately so it can follow the reactive unit (useUnits).
export const FIELDS = [
  ['date', 'Observed', (v) => v],
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

async function fetchObservations(datasetPath = '/mushroom_observations.geojson') {
  // Prefer a direct dataset selection when one is set. If not, use the latest
  // processed GeoJSON first, then fall back to the Netlify blob and static baseline.
  for (const candidate of [datasetPath, '/.netlify/functions/observations', '/data/observations.geojson']) {
    if (!candidate) continue
    try {
      const res = await fetch(candidate)
      if (!res.ok) continue
      const json = await res.json()
      // Only accept an enriched GeoJSON FeatureCollection. Raw iNaturalist
      // record arrays (no `.features`) would leave every view empty, so skip
      // them and fall back to a dataset that actually renders.
      if (json && Array.isArray(json.features)) return json
    } catch {
      // fall through to the next candidate
    }
  }
  throw new Error('No observation dataset was available to load.')
}

export function useObservations() {
  const route = useRoute()
  const routeScope = (route.path || '/').replace(/^\/+|\/+$/g, '') || 'root'
  const keyPrefix = `observations-${routeScope.replace(/\//g, '-')}`

  const data = useState(`${keyPrefix}-data`, () => null)
  const error = useState(`${keyPrefix}-error`, () => '')
  const pending = useState(`${keyPrefix}-pending`, () => false)
  const selectedDataset = useState(`${keyPrefix}-dataset`, () => {
    if (import.meta.client) {
      const saved = localStorage.getItem(`${keyPrefix}-dataset`)
      if (saved) return saved
    }
    return OBSERVATION_DATASETS[0].path
  })
  const availableDatasets = OBSERVATION_DATASETS

  async function load() {
    if (pending.value) return
    pending.value = true
    try {
      data.value = await fetchObservations(selectedDataset.value)
      error.value = ''
    } catch (e) {
      error.value = e.message
    } finally {
      pending.value = false
    }
  }

  function setDataset(path) {
    selectedDataset.value = path
    if (import.meta.client) localStorage.setItem(`${keyPrefix}-dataset`, path)
    data.value = null
    error.value = ''
    return load()
  }

  // Convenience: a flat array of property objects (with lon/lat attached).
  const rows = computed(() => (data.value?.features || []).map((f) => ({
    ...f.properties,
    lon: f.geometry?.coordinates?.[0],
    lat: f.geometry?.coordinates?.[1],
  })))

  return { data, rows, error, pending, load, setDataset, selectedDataset, availableDatasets }
}
