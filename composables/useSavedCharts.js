// Custom charts the user saves from the chart builder, shown (reorderable) at
// the top of the Charts page.
//
// localStorage is the working copy so the feature works signed out; when an
// account is connected, every change is mirrored to Supabase so the charts
// follow the viewer to another device. A cloud write that fails is logged and
// swallowed — losing the chart locally because the network blipped would be a
// far worse outcome than being briefly out of sync.

const KEY = 'saved-charts'

export function useSavedCharts() {
  const charts = useState('saved-charts', () => [])
  const cloud = useCloudSync()

  async function pushCloud() {
    if (!cloud.enabled.value) return
    try {
      const saved = await cloud.pushCharts(charts.value)
      // Adopt the server-assigned ids so a later edit updates the same rows.
      if (saved.length === charts.value.length) {
        charts.value = saved
        persist()
      }
    } catch (err) {
      console.warn('Could not sync charts to your account:', err?.message || err)
    }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const raw = localStorage.getItem(KEY)
      if (raw) charts.value = JSON.parse(raw)
    } catch {
      charts.value = []
    }
  }

  function persist() {
    if (import.meta.client) {
      try { localStorage.setItem(KEY, JSON.stringify(charts.value)) } catch { /* ignore */ }
    }
  }

  function add(config) {
    const entry = { id: `c${Date.now()}${Math.floor(Math.random() * 1000)}`, ...config }
    charts.value = [...charts.value, entry]
    persist()
    pushCloud()
    return entry.id
  }

  function remove(id) {
    charts.value = charts.value.filter((c) => c.id !== id)
    persist()
    pushCloud()
  }

  function move(id, dir) {
    const i = charts.value.findIndex((c) => c.id === id)
    const j = i + dir
    if (i < 0 || j < 0 || j >= charts.value.length) return
    const next = [...charts.value]
    ;[next[i], next[j]] = [next[j], next[i]]
    charts.value = next
    persist()
    pushCloud()
  }

  return { charts, loadFromStorage, persist, pushCloud, add, remove, move }
}
