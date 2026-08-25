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

const COUNTRY_ALIASES = { us: 'United States', usa: 'United States', 'united states': 'United States' }
const ADMIN_HINT = /\b(County|Parish|Borough|Municipality|Census Area)\b/i

export function parsePlace(location) {
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

export function haversineKm(a, b) {
  const R = 6371
  const toRad = (d) => (d * Math.PI) / 180
  const dLat = toRad(b.lat - a.lat)
  const dLng = toRad(b.lng - a.lng)
  const s = Math.sin(dLat / 2) ** 2
    + Math.cos(toRad(a.lat)) * Math.cos(toRad(b.lat)) * Math.sin(dLng / 2) ** 2
  return 2 * R * Math.asin(Math.min(1, Math.sqrt(s)))
}

// ISO week (1..53) for a YYYY-MM-DD date, matching the week-of-year charts.
export function isoWeek(dateStr) {
  if (!dateStr) return null
  const d = new Date(`${dateStr}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return null
  const day = (d.getUTCDay() + 6) % 7
  d.setUTCDate(d.getUTCDate() - day + 3)
  const firstThursday = new Date(Date.UTC(d.getUTCFullYear(), 0, 4))
  const week = 1 + Math.round(((d - firstThursday) / 86400000 - 3 + ((firstThursday.getUTCDay() + 6) % 7)) / 7)
  return week
}

export const EMPTY_FILTERS = {
  center: null, radiusKm: null, // { lat, lng } + radius
  country: '', state: '', county: '',
  year: '', month: '', week: '', dateFrom: '', dateTo: '',
}

// Pure predicate: does one feature pass the active filters?
export function matchesFilters(feature, f) {
  const p = feature.properties || {}
  const coords = feature.geometry?.coordinates
  const lng = coords?.[0], lat = coords?.[1]

  // Location — radius from a chosen center.
  if (f.center && f.radiusKm && lat != null && lng != null) {
    if (haversineKm(f.center, { lat, lng }) > f.radiusKm) return false
  }
  // Location — admin (parsed from the place string).
  if (f.country || f.state || f.county) {
    const place = parsePlace(p.location)
    if (f.country && place.country !== f.country) return false
    if (f.state && place.state !== f.state) return false
    if (f.county && place.county !== f.county) return false
  }
  // Time.
  const date = p.date
  if (f.year || f.month || f.week || f.dateFrom || f.dateTo) {
    if (!date) return false
    if (f.dateFrom && date < f.dateFrom) return false
    if (f.dateTo && date > f.dateTo) return false
    if (f.year && date.slice(0, 4) !== String(f.year)) return false
    if (f.month && date.slice(5, 7) !== String(f.month).padStart(2, '0')) return false
    if (f.week && isoWeek(date) !== Number(f.week)) return false
  }
  return true
}

export function useFilters() {
  const filters = useState('observations-filters', () => ({ ...EMPTY_FILTERS }))

  function setFilter(key, value) { filters.value = { ...filters.value, [key]: value } }
  function setCenter(center, radiusKm) { filters.value = { ...filters.value, center, radiusKm } }
  function reset() { filters.value = { ...EMPTY_FILTERS } }

  const activeCount = computed(() => {
    const f = filters.value
    let n = 0
    if (f.center && f.radiusKm) n++
    for (const k of ['country', 'state', 'county', 'year', 'month', 'week', 'dateFrom', 'dateTo']) if (f[k]) n++
    return n
  })

  return { filters, setFilter, setCenter, reset, activeCount }
}
