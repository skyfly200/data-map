// Per-viewer layout for the preset chart gallery: which charts are shown and in
// what order. Persisted in localStorage, so a reader's arrangement survives
// reloads. Order is applied with CSS grid `order`, so reordering never
// re-renders (or re-computes) a chart.

const KEY = 'chart-layout'

// Canonical gallery charts. `id` is the stable key used for order/hidden state;
// `title` labels the chart in the "hidden" list, where the chart itself isn't
// rendered to speak for itself. Adding a chart here (and tagging its card with
// the same id) is all that's needed for it to join the layout.
export const GALLERY_CHARTS = [
  { id: 'clusters', title: 'Observations per environmental cluster' },
  { id: 'rain-leadup', title: 'Avg. rain in the 7 days before' },
  { id: 'coverage', title: 'Enrichment coverage' },
  { id: 'by-month', title: 'Observations by month' },
  { id: 'by-week', title: 'Observations by week of year' },
  { id: 'temp-leadup', title: 'Avg. daily high in the 7 days before' },
  { id: 'temp-dist', title: 'Observation-day temperature' },
  { id: 'elevation-dist', title: 'Elevation distribution' },
  { id: 'land-cover', title: 'Land cover' },
  { id: 'top-species', title: 'Top species' },
  { id: 'elev-vs-doy', title: 'Elevation vs. day of year' },
  { id: 'elev-vs-temp', title: 'Elevation vs. observation-day high temp' },
  { id: 'rain-vs-doy', title: '7-day rain total vs. day of year' },
  { id: 'phenology', title: 'Fruiting season by species' },
  { id: 'elevation-by-species', title: 'Elevation range by species' },
  { id: 'cluster-profile', title: 'Environmental cluster profiles' },
  { id: 'species-landcover', title: 'Species × land cover' },
  { id: 'antecedent-rain', title: 'Antecedent rainfall' },
  { id: 'aspect', title: 'Slope aspect of finds' },
]

const DEFAULT_ORDER = GALLERY_CHARTS.map((c) => c.id)
const titleOf = (id) => GALLERY_CHARTS.find((c) => c.id === id)?.title || id

export function useChartLayout() {
  const cloud = safeCloudSync()
  const order = useState('chart-layout-order', () => [...DEFAULT_ORDER])
  const hidden = useState('chart-layout-hidden', () => [])
  const editing = useState('chart-layout-editing', () => false)
  // Ids of cards actually on screen. Some presets carry a data condition (no
  // temperature history, no cluster profile, …) and never render, so "shown"
  // has to come from the cards themselves, not from order minus hidden.
  const rendered = useState('chart-layout-rendered', () => [])
  function register(id) { if (!rendered.value.includes(id)) rendered.value = [...rendered.value, id] }
  function unregister(id) { rendered.value = rendered.value.filter((x) => x !== id) }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem(KEY) || 'null')
      if (!saved) return
      // Keep only ids we still ship, then append any newly-added charts so a
      // stored layout never hides a chart added in a later release.
      const known = new Set(DEFAULT_ORDER)
      const kept = (saved.order || []).filter((id) => known.has(id))
      order.value = [...kept, ...DEFAULT_ORDER.filter((id) => !kept.includes(id))]
      hidden.value = (saved.hidden || []).filter((id) => known.has(id))
    } catch { /* keep defaults */ }
  }

  function persist() {
    if (!import.meta.client) return
    try {
      localStorage.setItem(KEY, JSON.stringify({ order: order.value, hidden: hidden.value }))
      cloud?.schedulePush()
    } catch { /* ignore */ }
  }

  const isVisible = (id) => !hidden.value.includes(id)
  // Index in the order list → the CSS `order` value for that card.
  const orderOf = (id) => {
    const i = order.value.indexOf(id)
    return i === -1 ? DEFAULT_ORDER.length : i
  }

  // Move among the VISIBLE charts, so a hidden neighbour doesn't eat a click.
  function move(id, dir) {
    const visible = order.value.filter(isVisible)
    const vi = visible.indexOf(id)
    const target = visible[vi + dir]
    if (vi < 0 || target === undefined) return
    const next = [...order.value]
    const a = next.indexOf(id), b = next.indexOf(target)
    ;[next[a], next[b]] = [next[b], next[a]]
    order.value = next
    persist()
  }

  function hide(id) {
    if (!hidden.value.includes(id)) { hidden.value = [...hidden.value, id]; persist() }
  }
  function show(id) {
    hidden.value = hidden.value.filter((x) => x !== id)
    persist()
  }
  function showAll() { hidden.value = []; persist() }
  function reset() { order.value = [...DEFAULT_ORDER]; hidden.value = []; persist() }

  const hiddenCharts = computed(() => hidden.value.map((id) => ({ id, title: titleOf(id) })))
  const visibleCount = computed(() => rendered.value.length)

  return {
    order, hidden, editing, hiddenCharts, visibleCount,
    loadFromStorage, isVisible, orderOf, move, hide, show, showAll, reset,
    register, unregister,
  }
}
