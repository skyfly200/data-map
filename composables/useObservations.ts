// Shared observation data + display helpers for every view (map, table, charts).
// Data is fetched once on the client and cached in Nuxt state across pages.

import { markRaw } from 'vue'
import { TAXON_RANKS } from '~/composables/useChartFields'
import { resetMapChunks } from '~/composables/useMapChunks'

export interface Observation {
  [key: string]: any
  species?: string
  genus?: string
  date?: string
  year?: number
  month?: number
  month_name?: string
  enrichment_level?: 'none' | 'partial' | 'full'
  lon: number
  lat: number
}

export interface ObservationFeature {
  type: 'Feature'
  geometry: {
    type: 'Point'
    coordinates: [number, number]
  }
  properties: Observation
}

export interface ObservationCollection {
  type: 'FeatureCollection'
  features: ObservationFeature[]
}

// Fallback dataset list.
export const DATASET_MANIFEST = '/data/datasets.json'
export const DEFAULT_DATASET = '/mushroom_observations.geojson'
export const OBSERVATION_DATASETS = [
  { id: 'all', label: 'All species', path: DEFAULT_DATASET },
]


export async function fetchObservationDetails(id: string | number): Promise<any | null> {
  if (!id) return null
  try {
    const res = await fetch(`https://api.inaturalist.org/v1/observations/${id}`)
    if (!res.ok) return null
    const data = await res.json()
    return data.results?.[0] || null
  } catch {
    return null
  }
}

export function inatUrl(obs: any): string | null {
  if (!obs) return null
  const id = obs.inat_id ?? obs.uuid
  return id ? `https://www.inaturalist.org/observations/${id}` : null
}

const INAT_PHOTO_SIZES = ['square', 'small', 'medium', 'large', 'original']

export function inatPhotoUrl(photo: any, size: string = 'large'): string | null {
  if (!photo) return null
  const base = photo[`${size}_url`] || photo.large_url || photo.medium_url
    || photo.original_url || photo.url
  if (!base) return null
  return String(base).replace(
    new RegExp(`/(${INAT_PHOTO_SIZES.join('|')})(\\.[a-z0-9]+)(\\?|$)`, 'i'),
    `/${size}$2$3`,
  )
}

const num1 = (v: any) => Number(v).toFixed(1)
const num2 = (v: any) => Number(v).toFixed(2)
const num3 = (v: any) => Number(v).toFixed(3)

export const FIELDS: [string, string, (v: any) => string][] = [
  ['date', 'Observed', (v) => v],
  ['land_cover_label', 'Land cover', (v) => v],
  ['ndvi', 'NDVI', num3],
  ['soil_moisture', 'Soil moisture', num3],
  ['solar_exposure', 'Solar exposure', num2],
  ['wind_exposure', 'Wind exposure', num2],
  ['water_retention', 'Water retention', num2],
  ['slope', 'Slope', (v) => `${num1(v)}°`],
  ['aspect', 'Aspect', (v) => `${num1(v)}°`],
  ['location_precision', 'Location', (v) => LOCATION_PRECISION_LABELS[v as keyof typeof LOCATION_PRECISION_LABELS] || v],
]

export const LOCATION_PRECISION_LABELS = {
  precise: 'Precise',
  coarse: 'Coarse (>1km)',
  obscured: 'Obscured (~20km)',
  unknown: 'Not reported',
}

export const IMPRECISE_PRECISIONS = new Set(['obscured', 'coarse'])

export function hasValue(v: any): boolean {
  return v !== null && v !== undefined && v !== ''
}

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const ENRICH_COLS = ['ndvi', 'soil_moisture', 'slope', 'solar_exposure', 'land_cover_label', 'prcp_d0']

function deriveFields(geojson: ObservationCollection): ObservationCollection {
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

const inlineDatasets = new Map<string, ObservationCollection>()

async function fetchObservations(datasetPath = '/mushroom_observations.geojson'): Promise<ObservationCollection> {
  if (inlineDatasets.has(datasetPath)) return inlineDatasets.get(datasetPath)!
  for (const candidate of [datasetPath, '/.netlify/functions/observations', '/data/observations.geojson']) {
    if (!candidate) continue
    try {
      const res = await fetch(candidate)
      if (!res.ok) continue
      const json = await res.json()
      if (json && Array.isArray(json.features)) return json
    } catch {
      // fall through
    }
  }
  throw new Error('No observation dataset was available to load.')
}

const DATASET_KEY = 'observations-dataset'

export function useObservations() {
  const data = useState<ObservationCollection | null>('observations-data', () => null)
  const error = useState<string>('observations-error', () => '')
  const pending = useState<boolean>('observations-pending', () => false)
  const selectedDataset = useState<string>(DATASET_KEY, () => {
    if (import.meta.client) {
      const saved = localStorage.getItem(DATASET_KEY)
      if (saved) return saved
    }
    return DEFAULT_DATASET
  })
  const availableDatasets = useState<any[]>('observation-datasets', () => OBSERVATION_DATASETS)

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

  function setShowFiltered(includeFiltered: boolean | null | undefined) {
    showFiltered.value = !!includeFiltered
    if (import.meta.client) {
      localStorage.setItem('observations-show-filtered', String(showFiltered.value))
    }
  }

  async function load() {
    if ((data.value && !partial.value) || pending.value) return
    pending.value = true
    try {
      data.value = markRaw(deriveFields(await fetchObservations(selectedDataset.value)))
      partial.value = false
      error.value = ''
    } catch (e: any) {
      error.value = e.message
    } finally {
      pending.value = false
    }
  }

  async function loadProgressive(view = {}): Promise<boolean> {
    if (!import.meta.client) return false
    if (selectedDataset.value !== DEFAULT_DATASET) return false

    await chunks.loadIndex()
    if (!chunks.available.value) return false

    const before = chunks.version.value
    await chunks.loadOverview()
    await chunks.loadForView(view)
    if (data.value && chunks.version.value === before) return true

    const features = chunks.features.value
    if (!features.length) return false
    data.value = markRaw(deriveFields({ type: 'FeatureCollection', features }))
    partial.value = chunks.stats.value.loaded < chunks.stats.value.total
    error.value = ''
    return true
  }

  function setDataset(path: string) {
    selectedDataset.value = path
    if (import.meta.client) localStorage.setItem(DATASET_KEY, path)
    data.value = null
    error.value = ''
    // Clear the chunk cache so stale cells from the previous dataset don't bleed
    // through when the new dataset uses the same progressive-load infrastructure.
    resetMapChunks()
    return load()
  }

  function addInlineDataset(entry: any, geojson?: ObservationCollection) {
    if (!availableDatasets.value.some((d) => d.path === entry.path)) {
      availableDatasets.value = [...availableDatasets.value, entry]
    }
    speciesFilter.value = []
    if (geojson) {
      deriveFields(geojson)
      inlineDatasets.set(entry.path, geojson)
      data.value = markRaw(geojson)
      selectedDataset.value = entry.path
    } else {
      // Geojson was persisted to storage; load it from the returned path.
      setDataset(entry.path)
    }
    if (import.meta.client && !String(entry.path).startsWith('mem:')) {
      try { localStorage.setItem(DATASET_KEY, entry.path) } catch { /* ignore */ }
    }
  }

  const speciesFilter = useState<string[]>('observations-species-filter', () => [])
  const taxonRank = useState<string>('observations-taxon-rank', () => 'species')

  const { filters } = useFilters()
  const partial = useState('observations-partial', () => false)
  const chunks = useMapChunks()

  const filteredData = computed(() => {
    const feats = data.value?.features || []
    const sel = speciesFilter.value
    const set = sel.length ? new Set(sel) : null
    const f = filters.value
    const hideFiltered = !showFiltered.value
    let out = feats.filter((feat) => {
      if (hideFiltered && feat.properties?.water_mask) return false
      if (set && !set.has(feat.properties?.[taxonRank.value])) return false
      return matchesFilters(feat, f)
    })

    const keep = taxaAboveThreshold(out, f.minObsField || 'species', f.minObs)
    if (keep) out = out.filter((feat) => keep.has(feat.properties?.[f.minObsField || 'species']))

    return { type: 'FeatureCollection', features: out }
  })

  const speciesOptions = computed(() => {
    const counts = new Map<string, number>()
    const rank = taxonRank.value
    for (const f of data.value?.features || []) {
      const s = f.properties?.[rank]
      if (!s) continue
      counts.set(s, (counts.get(s) || 0) + 1)
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([species, count]) => ({ species, count }))
  })

  const availableRanks = computed(() => {
    const feats = data.value?.features || []
    return TAXON_RANKS.filter((r) => feats.some((f) => hasValue(f.properties?.[r.key])))
  })

  function setSpeciesFilter(list: string[]) { speciesFilter.value = [...list] }

  function setTaxonRank(rank: string) {
    if (rank === taxonRank.value) return
    taxonRank.value = rank
    speciesFilter.value = []
  }

  const focusObservation = useState('focus-observation', () => null)
  function setFocusObservation(obs: any) {
    focusObservation.value = obs ? { ...obs } : null
  }

  const filterOptions = computed(() => {
    const countries = new Set<string>(), states = new Set<string>(), counties = new Set<string>(), years = new Set<string>()
    for (const feat of data.value?.features || []) {
      const p = feat.properties || {}
      const place = parsePlace(p.location)
      if (place.country) countries.add(place.country)
      if (place.state) states.add(place.state)
      if (place.county) counties.add(place.county)
      if (p.date) years.add(p.date.slice(0, 4))
    }
    const sort = (s: Set<string>) => [...s].sort()
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
    partial, loadProgressive, chunks,
    taxonRank, setTaxonRank, availableRanks, TAXON_RANKS,
    showFiltered, setShowFiltered, focusObservation, setFocusObservation,
  }
}
