// Named filter subsets: save the current filter state under a name and get back
// to it in one click.
//
// Reconstructing "Colorado morels, September, at least 5 records" by hand every
// time is the kind of friction that stops people asking follow-up questions. A
// subset captures the whole filter state — place, dates, radius, species
// selection and the minimum-observation threshold — so it restores exactly.
//
// Stored alongside the other preferences, so it rides the same account sync.

const KEY = 'saved-filters'

/** Everything that defines a subset, as a plain object. */
export function snapshotFilters(filters, species = []) {
  const f = filters || {}
  return {
    center: f.center || null,
    radiusKm: f.radiusKm || null,
    country: f.country || '',
    state: f.state || '',
    county: f.county || '',
    year: f.year || '',
    month: f.month || '',
    week: f.week || '',
    dateFrom: f.dateFrom || '',
    dateTo: f.dateTo || '',
    minObs: f.minObs || 0,
    minObsField: f.minObsField || 'species',
    species: [...(species || [])],
  }
}

/**
 * A short human description of a subset, for the chip that restores it.
 *
 * Built from whatever is actually set rather than a fixed template, so a subset
 * that only narrows species does not read as though it also narrowed dates.
 */
export function describeFilters(snapshot) {
  const s = snapshot || {}
  const parts = []
  if (s.species?.length) {
    parts.push(s.species.length === 1 ? s.species[0] : `${s.species.length} species`)
  }
  const place = [s.county, s.state, s.country].filter(Boolean)[0]
  if (place) parts.push(place)
  if (s.center && s.radiusKm) parts.push(`within ${s.radiusKm} km`)
  if (s.year) parts.push(String(s.year))
  if (s.month) {
    const names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    parts.push(names[Number(s.month) - 1] || `month ${s.month}`)
  }
  if (s.week) parts.push(`week ${s.week}`)
  if (s.dateFrom || s.dateTo) parts.push(`${s.dateFrom || '…'} → ${s.dateTo || '…'}`)
  if (s.minObs > 1) parts.push(`≥${s.minObs} per ${s.minObsField || 'species'}`)
  return parts.length ? parts.join(' · ') : 'No filters'
}

export function useSavedFilters() {
  const subsets = useState('saved-filters', () => [])
  const cloud = safeCloudSync()
  const { filters } = useFilters()
  const { speciesFilter, setSpeciesFilter } = useObservations()

  function persist() {
    if (!import.meta.client) return
    try {
      localStorage.setItem(KEY, JSON.stringify(subsets.value))
      cloud?.schedulePush()
    } catch { /* ignore */ }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const raw = JSON.parse(localStorage.getItem(KEY) || 'null')
      if (Array.isArray(raw)) subsets.value = raw
    } catch {
      subsets.value = []
    }
  }

  /** Save the live filter state under `name`. Re-saving a name replaces it. */
  function save(name) {
    const label = String(name || '').trim()
    if (!label) return null
    const entry = {
      id: `f${Date.now()}${Math.floor(Math.random() * 1000)}`,
      name: label,
      snapshot: snapshotFilters(filters.value, speciesFilter.value),
      savedAt: new Date().toISOString(),
    }
    const existing = subsets.value.findIndex((s) => s.name.toLowerCase() === label.toLowerCase())
    if (existing >= 0) {
      const next = [...subsets.value]
      next[existing] = { ...entry, id: next[existing].id }
      subsets.value = next
    } else {
      subsets.value = [...subsets.value, entry]
    }
    persist()
    return entry.id
  }

  /** Put a saved subset back on. */
  function apply(id) {
    const entry = subsets.value.find((s) => s.id === id)
    if (!entry) return false
    const { species, ...rest } = entry.snapshot
    filters.value = { ...filters.value, ...rest }
    setSpeciesFilter(species || [])
    return true
  }

  function remove(id) {
    subsets.value = subsets.value.filter((s) => s.id !== id)
    persist()
  }

  function rename(id, name) {
    const label = String(name || '').trim()
    if (!label) return
    subsets.value = subsets.value.map((s) => (s.id === id ? { ...s, name: label } : s))
    persist()
  }

  // True when the live filters match a saved subset, so the UI can show which
  // one is in effect rather than leaving every chip looking equally inactive.
  const activeId = computed(() => {
    const now = JSON.stringify(snapshotFilters(filters.value, speciesFilter.value))
    return subsets.value.find((s) => JSON.stringify(s.snapshot) === now)?.id || null
  })

  return {
    subsets, activeId,
    loadFromStorage, save, apply, remove, rename,
    snapshotFilters, describeFilters,
  }
}
