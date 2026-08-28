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

// Deterministic colour for a category value, so the same value (a species, a
// year, a land-cover class) gets the SAME colour on the map and in every chart.
// A stable string hash → palette index; identity is never colour-alone (legends
// everywhere), so palette collisions across distant values are acceptable.
export function stableColor(value) {
  if (value === null || value === undefined || value === '') return UNCLUSTERED
  const s = String(value)
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
  return PALETTE[h % PALETTE.length]
}

// Colour for a (field, value) pair. Cluster keeps its index-based palette
// (already consistent everywhere); every other category uses the stable hash.
export function categoryColor(field, value) {
  if (value === null || value === undefined || value === '') return UNCLUSTERED
  if (field === 'cluster' || field === 'live_cluster') {
    const n = Number(String(value).replace(/^[CK]/, ''))
    return Number.isFinite(n) ? colorFor(n) : UNCLUSTERED
  }
  return stableColor(value)
}

// Canonical iNaturalist URL. Prefers the numeric id; falls back to the UUID,
// which the iNaturalist observations route also resolves.
export async function fetchObservationDetails(id) {
  if (!id) return null
  try {
    const res = await fetch(`https://api.inaturalist.org/v1/observations/${id}`)
    if (!res.ok) return null
    const data = await res.json()
    const obs = data.results?.[0]
    return obs || null
  } catch {
    return null
  }
}

export function inatUrl(obs) {
  if (!obs) return null
  const id = obs.inat_id ?? obs.uuid
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

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

// Columns that mark an observation as environmentally enriched. Its
// "enrichment level" is how many of these carry data — a quick coverage lens.
const ENRICH_COLS = ['ndvi', 'soil_moisture', 'slope', 'solar_exposure', 'land_cover_label', 'prcp_d0']

// Derive lightweight fields the views can group/colour/plot by without a
// pipeline change — genus (from the binomial), year/month (from the date), and
// an enrichment-level bucket. Mutates features in place; idempotent.
function deriveFields(geojson) {
  for (const f of geojson?.features || []) {
    const p = f.properties
    if (!p) continue
    if (!hasValue(p.genus) && hasValue(p.species)) {
      p.genus = String(p.species).trim().split(/\s+/)[0]
    }
    if (!hasValue(p.year) && hasValue(p.date)) {
      const d = new Date(p.date)
      if (!Number.isNaN(d.getTime())) {
        p.year = d.getUTCFullYear()
        p.month = d.getUTCMonth() + 1
        p.month_name = MONTH_NAMES[d.getUTCMonth()]
      }
    }
    if (!hasValue(p.enrichment_level)) {
      const n = ENRICH_COLS.reduce((s, c) => s + (hasValue(p[c]) ? 1 : 0), 0)
      p.enrichment_level = n === 0 ? 'none' : n >= ENRICH_COLS.length ? 'full' : 'partial'
    }
  }
  return geojson
}

// Datasets fetched on the fly (Data tab) and held in memory for the session.
const inlineDatasets = new Map()

async function fetchObservations(datasetPath = '/mushroom_observations.geojson') {
  if (inlineDatasets.has(datasetPath)) return inlineDatasets.get(datasetPath)
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
      const paths = new Set(list.map((d) => d.path))
      const canonical = list.find((d) => d.id === 'all')?.path || list[0].path
      const saved = import.meta.client ? localStorage.getItem(DATASET_KEY) : null
      const current = selectedDataset.value

      // A stale single-species dataset can linger in local storage; always prefer
      // the canonical all-species dataset when it is available so the UI shows
      // the enriched multi-species baseline instead of a morel-only snapshot.
      if (!paths.has(current) || current.includes('/species/')) {
        selectedDataset.value = canonical
        if (import.meta.client) localStorage.setItem(DATASET_KEY, canonical)
        return
      }

      if (saved && saved !== current && paths.has(saved)) {
        selectedDataset.value = saved
        if (import.meta.client) localStorage.setItem(DATASET_KEY, saved)
      }
    } catch {
      // keep the fallback list
    }
  }

  const showFiltered = useState('observations-show-filtered', () => {
    if (import.meta.client) {
      const saved = localStorage.getItem('observations-show-filtered')
      return saved === 'true'
    }
    return false
  })

  function setShowFiltered(includeFiltered) {
    showFiltered.value = !!includeFiltered
    if (import.meta.client) {
      localStorage.setItem('observations-show-filtered', String(showFiltered.value))
    }
  }

  async function load() {
    if (data.value || pending.value) return // already loaded (shared across views)
    pending.value = true
    try {
      data.value = deriveFields(await fetchObservations(selectedDataset.value))
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

  // Add a dataset fetched on the fly (Data tab) and switch to it.
  function addInlineDataset(entry, geojson) {
    deriveFields(geojson)
    inlineDatasets.set(entry.path, geojson)
    if (!availableDatasets.value.some((d) => d.path === entry.path)) {
      availableDatasets.value = [...availableDatasets.value, entry]
    }
    speciesFilter.value = []
    data.value = geojson
    selectedDataset.value = entry.path
    // Persist only real (servable) paths; in-memory ones can't survive reload.
    if (import.meta.client && !String(entry.path).startsWith('mem:')) {
      try { localStorage.setItem(DATASET_KEY, entry.path) } catch { /* ignore */ }
    }
  }

  // Convenience: a flat array of property objects (with lon/lat attached).
  // Optional species filter (a Set of species names). Empty = show all.
  // Applied to every view (map/table/charts/explore) via filteredData + rows.
  const speciesFilter = useState('observations-species-filter', () => [])

  const { filters } = useFilters()

  const filteredData = computed(() => {
    const feats = data.value?.features || []
    const sel = speciesFilter.value
    const set = sel.length ? new Set(sel) : null
    const f = filters.value
    const hideFiltered = !showFiltered.value
    const out = feats.filter((feat) => {
      // Non-productive land cover (water) is hidden unless the user opts in.
      if (hideFiltered && feat.properties?.water_mask) return false
      if (set && !set.has(feat.properties?.species)) return false
      return matchesFilters(feat, f)
    })
    return { type: 'FeatureCollection', features: out }
  })

  // Species present in the loaded dataset with counts (unfiltered), for pickers.
  const speciesOptions = computed(() => {
    const counts = new Map()
    for (const f of data.value?.features || []) {
      const s = f.properties?.species
      if (!s) continue
      counts.set(s, (counts.get(s) || 0) + 1)
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([species, count]) => ({ species, count }))
  })

  function setSpeciesFilter(list) { speciesFilter.value = [...list] }

  // A single observation the user asked to "open on the map" (from a chart).
  // The map watches this, selects the matching point, and pans to it.
  const focusObservation = useState('focus-observation', () => null)
  function setFocusObservation(obs) {
    focusObservation.value = obs ? { ...obs } : null
  }

  // Distinct location + time values present in the loaded data, for the filter
  // dropdowns. Species filter is intentionally ignored here so the options
  // reflect the whole dataset, not the current narrowing.
  const filterOptions = computed(() => {
    const countries = new Set(), states = new Set(), counties = new Set(), years = new Set()
    for (const feat of data.value?.features || []) {
      const p = feat.properties || {}
      const place = parsePlace(p.location)
      if (place.country) countries.add(place.country)
      if (place.state) states.add(place.state)
      if (place.county) counties.add(place.county)
      if (p.date) years.add(p.date.slice(0, 4))
    }
    const sort = (s) => [...s].sort()
    return {
      countries: sort(countries), states: sort(states), counties: sort(counties),
      years: sort(years).reverse(),
    }
  })

  const rows = computed(() => (filteredData.value?.features || []).map((f) => ({
    ...f.properties,
    lon: f.geometry?.coordinates?.[0],
    lat: f.geometry?.coordinates?.[1],
  })))

  return {
    data, filteredData, rows, error, pending, load, loadDatasets, setDataset, addInlineDataset,
    selectedDataset, availableDatasets, speciesFilter, speciesOptions, setSpeciesFilter, filterOptions,
    showFiltered, setShowFiltered, focusObservation, setFocusObservation,
  }
}
