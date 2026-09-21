// The taxa this viewer asks iNaturalist for, as a setting.
//
// Stored and synced like every other preference, so a list built on a laptop is
// there on a phone. The rules for what a list IS live in composables/taxonList
// .js, which is pure and knows nothing about storage; this is only the part
// that remembers.

import { computed } from 'vue'
import { FRMS_GENERA, formatTaxa, parseTaxa } from './taxonList'

const STORAGE_KEY = 'inat-taxa'

export function useTaxonList() {
  // Defaults to the list the project has been collecting, so an app with no
  // saved settings behaves the way it already did rather than asking a new
  // reader to build a list before anything works.
  const taxa = useState<string[]>('inat-taxa', () => [...FRMS_GENERA])

  const cloud = safeCloudSync()

  function persist() {
    if (!import.meta.client) return
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(taxa.value))
      cloud?.schedulePush()
    } catch { /* quota or private mode */ }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
      if (!saved) return
      // Parsed rather than trusted: this arrives from another device through
      // the sync table, and it is about to be interpolated into a request to
      // somebody else's API.
      const clean = parseTaxa(saved)
      if (clean.length) taxa.value = clean
    } catch { /* keep the default */ }
  }

  /** Replace the whole list, e.g. from a preset or a pasted env var. */
  function setTaxa(next: string | string[]) {
    taxa.value = parseTaxa(next)
    persist()
  }

  /** Add one name. Returns false when it is not a name, or is already there. */
  function addTaxon(name: string): boolean {
    const before = taxa.value.length
    const next = parseTaxa([...taxa.value, name])
    if (next.length === before) return false
    taxa.value = next
    persist()
    return true
  }

  function removeTaxon(name: string) {
    const key = String(name).toLowerCase()
    taxa.value = taxa.value.filter((t) => t.toLowerCase() !== key)
    persist()
  }

  function reset() {
    taxa.value = [...FRMS_GENERA]
    persist()
  }

  const asText = computed(() => formatTaxa(taxa.value))

  return { taxa, asText, setTaxa, addTaxon, removeTaxon, reset, persist, loadFromStorage }
}
