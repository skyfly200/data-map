// Shared location + time filters applied across every view (map, table, charts,
// explore) via useObservations.filteredData. State lives in useState so the
// filter panel (Data tab) and all the views stay in sync.
//
// The data model: each feature has coordinates (→ radius filter), an ISO `date`
// (→ year / month / week / date-range filters), and a freeform iNaturalist
// `location` string like "Ward, Boulder County, Colorado, US". There are no
// structured admin fields, so we parse country/state/county out of that string
// and populate the dropdowns from whatever parsing actually yields in the
// loaded data — the options always match what the filter can match.

import { matchesTaxon } from '../netlify/lib/dataset-taxa.mjs'

export interface Place {
  country: string | null
  state: string | null
  county: string | null
}

export interface ObservationFilters {
  center: { lat: number, lng: number } | null
  radiusKm: number | null
  search: string
  taxon: string
  elevMin: number | null
  elevMax: number | null
  country: string
  state: string
  county: string
  year: string
  month: string
  week: string
  dateFrom: string
  dateTo: string
  minObs: number
  minObsField: string
  preciseOnly: boolean
}

const COUNTRY_ALIASES: Record<string, string> = { us: 'United States', usa: 'United States', 'united states': 'United States' }
const ADMIN_HINT = /\b(County|Parish|Borough|Municipality|Census Area)\b/i

export function parsePlace(location: string | null | undefined): Place {
  if (!location) return { country: null, state: null, county: null }
  const parts = String(location).split(',').map((s) => s.trim()).filter(Boolean)
  if (!parts.length) return { country: null, state: null, county: null }

  let country = parts[parts.length - 1] || null
  if (country) {
    const key = country.toLowerCase()
    country = COUNTRY_ALIASES[key] || country
  }
  const county = parts.find((p) => ADMIN_HINT.test(p)) || null
  // State: the part just before the country, unless that part is the county.
  let state = parts.length >= 2 ? parts[parts.length - 2] : null
  if (state && county && state === county) state = parts.length >= 3 ? parts[parts.length - 3] : null
  return { country, state, county }
}

export function haversineKm(a: { lat: number, lng: number }, b: { lat: number, lng: number }) {
  const R = 6371
  const toRad = (d: number) => (d * Math.PI) / 180
  const dLat = toRad(b.lat - a.lat)
  const dLng = toRad(b.lng - a.lng)
  const s = Math.sin(dLat / 2) ** 2
    + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)))
}

export function isoWeek(dateStr: string | null | undefined) {
  if (!dateStr) return null
  const d = new Date(`${dateStr}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return null
  const day = (d.getUTCDay() + 6) % 7
  d.setUTCDate(d.getUTCDate() - day + 3)
  const firstThursday = new Date(Date.UTC(d.getUTCFullYear(), 0, 4))
  const week = 1 + Math.round(((d - firstThursday) / 86400000 - 3 + ((firstThursday.getUTCDay() + 6) % 7)) / 7)
  return week
}

export const EMPTY_FILTERS: ObservationFilters = {
  center: null, radiusKm: null,
  search: '',
  taxon: '',
  elevMin: null, elevMax: null,
  country: '', state: '', county: '',
  year: '', month: '', week: '', dateFrom: '', dateTo: '',
  minObs: 0, minObsField: 'species',
  preciseOnly: false,
}

export function taxaAboveThreshold(features: any[], field: string, min: number): Set<string> | null {
  if (!min || min < 2) return null
  const counts = new Map<string, number>()
  for (const f of features) {
    const v = f.properties?.[field]
    if (v !== null && v !== undefined && v !== '') counts.set(String(v), (counts.get(String(v)) || 0) + 1)
  }
  const keep = new Set<string>()
  for (const [value, n] of counts) if (n >= min) keep.add(value)
  return keep
}

export function searchHaystack(props = {}): string {
  return [props.species, props.common_name, props.genus, props.family, props.location]
    .filter(Boolean).join(' ').toLowerCase()
}

export function matchesFilters(feature: any, f: ObservationFilters): boolean {
  const p = feature.properties || {}
  const coords = feature.geometry?.coordinates
  const lng = coords?.[0], lat = coords?.[1]

  if (f.search) {
    const q = String(f.search).trim().toLowerCase()
    if (q && !searchHaystack(p).includes(q)) return false
  }
  if (f.taxon && !matchesTaxon(p, f.taxon)) return false
  if (f.elevMin != null || f.elevMax != null) {
    const raw = p.elevation
    const e = (raw === null || raw === undefined || raw === '') ? NaN : Number(raw)
    if (!Number.isFinite(e)) return false
    if (f.elevMin != null && e < f.elevMin) return false
    if (f.elevMax != null && e > f.elevMax) return false
  }

  if (f.center && f.radiusKm && lat != null && lng != null) {
    if (haversineKm(f.center, { lat, lng }) > f.radiusKm) return false
  }
  if (f.country || f.state || f.county) {
    // Overview features (from the thinned pre-paint file) may lack `location`
    // until their cell loads and the full record replaces them. Excluding them
    // outright makes the map look blank the moment any admin-area filter is
    // applied. Pass them through; the cell's arrival will correct any mismatch.
    if (p.location) {
      const place = parsePlace(p.location)
      if (f.country && place.country !== f.country) return false
      if (f.state && place.state !== f.state) return false
      if (f.county && place.county !== f.county) return false
    }
  }
  const date = p.date
  if (f.year || f.month || f.week || f.dateFrom || f.dateTo) {
    if (!date) return false
    if (f.dateFrom && date < f.dateFrom) return false
    if (f.dateTo && date > f.dateTo) return false
    if (f.year && date.slice(0, 4) !== String(f.year)) return false
    if (f.month && date.slice(5, 7) !== String(f.month).padStart(2, '0')) return false
    if (f.week && isoWeek(date) !== Number(f.week)) return false
  }
  if (f.preciseOnly && p.location_precision && p.location_precision !== 'precise') return false
  return true
}

export function useFilters() {
  const filters = useState<ObservationFilters>('observations-filters', () => ({ ...EMPTY_FILTERS }))

  function setFilter(key: keyof ObservationFilters, value: any) { filters.value = { ...filters.value, [key]: value } }
  function setCenter(center: { lat: number, lng: number } | null, radiusKm: number | null) { filters.value = { ...filters.value, center, radiusKm } }
  function reset() { filters.value = { ...EMPTY_FILTERS } }

  const activeCount = computed(() => {
    const f = filters.value
    let n = 0
    if (f.center && f.radiusKm) n++
    for (const k of ['search', 'taxon', 'country', 'state', 'county',
      'year', 'month', 'week', 'dateFrom', 'dateTo']) {
      if ((f as any)[k]) n++
    }
    if (f.elevMin != null || f.elevMax != null) n++
    if (f.minObs > 1) n++
    if (f.preciseOnly) n++
    return n
  })

  return { filters, setFilter, setCenter, reset, activeCount }
}
