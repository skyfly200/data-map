// Custom charts the user saves from Explore, persisted per viewer in
// localStorage and shown (reorderable) at the top of the Charts page.

const KEY = 'saved-charts'

export function useSavedCharts() {
  const charts = useState('saved-charts', () => [])

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
    return entry.id
  }

  function remove(id) {
    charts.value = charts.value.filter((c) => c.id !== id)
    persist()
  }

  function move(id, dir) {
    const i = charts.value.findIndex((c) => c.id === id)
    const j = i + dir
    if (i < 0 || j < 0 || j >= charts.value.length) return
    const next = [...charts.value]
    ;[next[i], next[j]] = [next[j], next[i]]
    charts.value = next
    persist()
  }

  return { charts, loadFromStorage, add, remove, move }
}
