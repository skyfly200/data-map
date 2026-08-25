// Shared observation data + display helpers for every view (map, table, charts).
// Data is fetched once on the client and cached in Nuxt state across pages.

// Fallback dataset list. The real list is loaded at runtime from the manifest
// public/data/datasets.json (written by export_geojson.py), which lists the
// combined dataset plus one enriched GeoJSON per species.
export const DATASET_MANIFEST = '/data/datasets.json'
export const DEFAULT_DATASET = '/data/observations.geojson'
export const OBSERVATION_DATASETS = [
  { id: 'all', label: 'All species', path: DEFAULT_DATASET },
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

const DATASET_KEY = 'observations-dataset'

export function useObservations() {
  // One shared dataset selection + loaded data across every view (map, table,
  // charts), so the header dataset picker is consistent everywhere.
  const data = useState('observations-data', () => null)
  const error = useState('observations-error', () => '')
  const pending = useState('observations-pending', () => false)
  const selectedDataset = useState(DATASET_KEY, () => {
    if (import.meta.client) {
      const saved = localStorage.getItem(DATASET_KEY)
      if (saved) return saved
    }
    return DEFAULT_DATASET
  })
  // The dataset list is the same for every view, so keep it in global state.
  const availableDatasets = useState('observation-datasets', () => OBSERVATION_DATASETS)

  async function loadDatasets() {
    const manifestUrl = useRuntimeConfig().public.datasetsManifestUrl || DATASET_MANIFEST
    try {
      const res = await fetch(manifestUrl)
      if (!res.ok) return
      const list = await res.json()
      if (!Array.isArray(list) || !list.length) return
      availableDatasets.value = list
      // Keep the current selection valid against the (possibly Supabase) paths.
      const paths = list.map((d) => d.path)
      const saved = import.meta.client ? localStorage.getItem(DATASET_KEY) : null
      if (!paths.includes(selectedDataset.value) && !(saved && paths.includes(saved))) {
        selectedDataset.value = list[0].path
      }
    } catch {
      // keep the fallback list
    }
  }

  async function load() {
    if (data.value || pending.value) return // already loaded (shared across views)
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
    if (import.meta.client) localStorage.setItem(DATASET_KEY, path)
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

  return { data, rows, error, pending, load, loadDatasets, setDataset, selectedDataset, availableDatasets }
}
