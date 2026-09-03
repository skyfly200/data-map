// Registering and driving the service worker that makes the app readable with
// no signal.
//
// Everything here is opt-in beyond the app shell. Silently pulling a 48 MB
// dataset and a few hundred map tiles onto someone's phone data is not a
// feature, so saving is something the viewer asks for and is told the size of
// first.

import { computed, ref } from 'vue'

const supported = ref(false)
const registered = ref(false)
const online = ref(true)
const busy = ref('')            // '', 'data', 'tiles', 'shell'
const progress = ref({ done: 0, total: 0 })
const usage = ref(null)
const error = ref('')

let bound = false

/** Ask the active worker something and wait for its answer. */
function ask(message, { onProgress = null, timeout = 600000 } = {}) {
  return new Promise((resolve, reject) => {
    const worker = navigator.serviceWorker?.controller
    if (!worker) { reject(new Error('Offline storage is not active yet — reload once.')); return }
    const channel = new MessageChannel()
    const timer = setTimeout(() => {
      channel.port1.close()
      reject(new Error('Saving timed out.'))
    }, timeout)
    channel.port1.onmessage = (event) => {
      const data = event.data || {}
      if (data.type === 'progress') { onProgress?.(data); return }
      clearTimeout(timer)
      channel.port1.close()
      if (data.type === 'error') reject(new Error(data.message || 'Saving failed.'))
      else resolve(data)
    }
    worker.postMessage(message, [channel.port2])
  })
}

export function useOffline() {
  if (import.meta.client && !bound) {
    bound = true
    supported.value = 'serviceWorker' in navigator && 'caches' in window
    online.value = navigator.onLine
    window.addEventListener('online', () => { online.value = true })
    window.addEventListener('offline', () => { online.value = false })
  }

  async function register() {
    if (!import.meta.client || !supported.value) return
    // Dev runs an unbundled app whose asset URLs change constantly; a worker
    // serving yesterday's module graph there is a debugging trap, not a
    // feature. Production only.
    if (import.meta.dev) return
    try {
      await navigator.serviceWorker.register('/sw.js', { scope: '/' })
      await navigator.serviceWorker.ready
      registered.value = !!navigator.serviceWorker.controller
      if (registered.value) refreshUsage()
    } catch (err) {
      error.value = `Could not enable offline use (${err.message}).`
    }
  }

  async function refreshUsage() {
    if (!navigator.serviceWorker?.controller) return
    try {
      const res = await ask({ type: 'usage' }, { timeout: 60000 })
      usage.value = res.usage || null
    } catch { /* a usage read is not worth surfacing */ }
  }

  async function run(kind, message) {
    error.value = ''
    busy.value = kind
    progress.value = { done: 0, total: 0 }
    try {
      const res = await ask(message, {
        onProgress: (p) => { progress.value = { done: p.done, total: p.total } },
      })
      await refreshUsage()
      return res
    } catch (err) {
      error.value = err.message
      return null
    } finally {
      busy.value = ''
      progress.value = { done: 0, total: 0 }
    }
  }

  /** Save the current dataset so the map, table and charts all work offline. */
  const saveData = (urls) => run('data', { type: 'save-data', urls })

  /** Save basemap tiles for an area. */
  const saveTiles = (urls) => run('tiles', { type: 'save-tiles', urls })

  /**
   * Save the app itself: the pages and the JS/CSS behind them.
   *
   * The worker caches these as they are used anyway, but only for pages that
   * have been visited — someone who saves a dataset from the map and then opens
   * Charts offline would find nothing there. Naming the routes up front is what
   * makes "works offline" mean the whole app.
   */
  const saveShell = () => run('shell', {
    type: 'save-shell',
    urls: ['/', '/map', '/charts', '/analysis', '/data', '/coverage', '/guide', '/options'],
  })

  async function clear(which = 'all') {
    error.value = ''
    try {
      await ask({ type: 'clear', which }, { timeout: 60000 })
      await refreshUsage()
    } catch (err) {
      error.value = err.message
    }
  }

  const savedBytes = computed(() => {
    const u = usage.value
    if (!u) return 0
    return (u.shell?.bytes || 0) + (u.data?.bytes || 0) + (u.tiles?.bytes || 0)
  })
  const savedTiles = computed(() => usage.value?.tiles?.count || 0)
  const hasData = computed(() => (usage.value?.data?.count || 0) > 0)

  return {
    supported, registered, online, busy, progress, usage, error,
    register, refreshUsage, saveData, saveTiles, saveShell, clear,
    savedBytes, savedTiles, hasData,
  }
}
