// Registering and driving the service worker that makes the app readable with
// no signal.
//
// Everything here is opt-in beyond the app shell. Silently pulling a 48 MB
// dataset and a few hundred map tiles onto someone's phone data is not a
// feature, so saving is something the viewer asks for and is told the size of
// first.

import { computed, ref } from 'vue'

import { guidePaths } from '~/composables/guidePages'

import {
  eeMapId, keysToDrop, makeArea, saveTargets,
} from '~/composables/offlineTiles'

const supported = ref(false)
const registered = ref(false)
const online = ref(true)
const busy = ref('')            // '', 'data', 'tiles', 'shell'
const progress = ref({ done: 0, total: 0 })
const usage = ref(null)
const error = ref('')
// The places saved for reading with no signal, newest first. Held by the
// worker, which is the only thing that outlives a tab.
const areas = ref([])

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
      // A newly installed worker claims its clients from its activate handler,
      // which lands after this. Waiting for that rather than reading the
      // controller once is the difference between the panel working now and
      // telling a first-time visitor to reload — which is what it used to do,
      // to everyone, on the visit where they went looking for it.
      if (!navigator.serviceWorker.controller) {
        await new Promise((resolve) => {
          const done = () => resolve()
          navigator.serviceWorker.addEventListener('controllerchange', done, { once: true })
          setTimeout(done, 3000)
        })
      }
      registered.value = !!navigator.serviceWorker.controller
      if (registered.value) {
        refreshUsage()
        loadAreas()
      }
    } catch (err) {
      error.value = `Could not enable offline use (${err.message}).`
    }
  }

  /**
   * What a URL costs to download, from its Content-Length.
   *
   * Asked before anything is saved, because "Save" on the observations is the
   * one button here that can spend tens of megabytes of somebody's mobile data
   * and nothing on the page said so. A transfer size rather than an occupied
   * size: the dataset is served precompressed, so this is what it costs to
   * fetch, which is the number a person on a phone is actually deciding about.
   *
   * Null when it cannot be known — a HEAD the server refuses, a chunked
   * response with no length. Null is rendered as nothing rather than as zero,
   * since "0 MB" next to a 48 MB download is worse than no number at all.
   */
  const measured = useState('offline-measured', () => ({}))

  async function measure(url) {
    if (!url || url.startsWith('mem:')) return null
    if (url in measured.value) return measured.value[url]
    let bytes = null
    try {
      const res = await fetch(url, { method: 'HEAD' })
      const len = Number(res.headers.get('content-length'))
      if (res.ok && Number.isFinite(len) && len > 0) bytes = len
    } catch { /* an unmeasurable URL is reported as unmeasured */ }
    measured.value = { ...measured.value, [url]: bytes }
    return bytes
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

  // ── Saved areas ────────────────────────────────────────────────────────────

  const origin = () => (import.meta.client ? window.location.origin : '')

  /**
   * A structured-cloneable copy of an area.
   *
   * Anything held in `areas` is wrapped in Vue's reactive proxy, and
   * postMessage refuses a proxy outright — "could not be cloned" — so a rename
   * built by spreading a stored area failed silently on its way to the worker
   * while the page went on showing the new name until the next reload put the
   * old one back. Area records are plain JSON by construction, so a round trip
   * is both sufficient and honest about what may live in one.
   */
  const plain = (value) => JSON.parse(JSON.stringify(value))

  async function loadAreas() {
    if (!navigator.serviceWorker?.controller) return areas.value
    try {
      const res = await ask({ type: 'areas' }, { timeout: 30000 })
      areas.value = res.areas || []
    } catch { /* an empty list is the right answer when the worker is not up */ }
    return areas.value
  }

  /**
   * Save a named place, with the layers that are on, for reading offline.
   *
   * `sources` are the layers to save: a template, plus a stable id for the ones
   * whose URL expires. See composables/offlineTiles.js — an Earth Engine layer
   * saved under its URL is unreachable within hours, which is the whole reason
   * a source is more than a string.
   */
  async function saveArea({ name, bounds, minZoom, maxZoom, sources, id }) {
    const area = plain(makeArea({ id, name, bounds, minZoom, maxZoom, sources }))
    const entries = saveTargets(area, origin())
    const res = await run('tiles', { type: 'save-tiles', entries, area })
    await loadAreas()
    return res ? { ...res, area } : null
  }

  /** Fetch again whatever is missing, under the layers given now. */
  const resaveArea = (area, sources) => saveArea(plain({
    ...area, sources: sources?.length ? sources : area.sources,
  }))

  /**
   * Forget a place, and the tiles nothing else still needs.
   *
   * The difference is worked out here rather than in the worker because it is
   * the page that holds every area and the arithmetic: two areas over the same
   * valley share tiles, and deleting one must not punch a hole in the other.
   */
  async function deleteArea(id) {
    const area = areas.value.find((a) => a.id === id)
    if (!area) return
    error.value = ''
    try {
      const keys = keysToDrop(area, areas.value, origin())
      const res = await ask({ type: 'drop-area', id, keys }, { timeout: 120000 })
      areas.value = res.areas || []
      await refreshUsage()
    } catch (err) {
      error.value = err.message
    }
  }

  async function renameArea(id, name) {
    const area = areas.value.find((a) => a.id === id)
    if (!area) return
    const next = plain({ ...area, name: (name || '').trim() || area.name })
    try {
      const res = await ask({ type: 'put-area', area: next }, { timeout: 30000 })
      areas.value = res.areas || []
    } catch (err) {
      error.value = err.message
    }
  }

  /**
   * Tell the worker which layer a freshly minted Earth Engine map id belongs to.
   *
   * Called every time a template is minted. Without it the worker sees a tile
   * URL whose token it has never met and cannot tell which layer's saved tiles
   * would answer it, so an area saved this morning goes blank this afternoon.
   */
  function registerEeTemplate(layerId, template) {
    const mapId = eeMapId(template)
    if (!mapId || !navigator.serviceWorker?.controller) return
    // Fire and forget: nothing waits on this, and a failure costs a cache hit
    // rather than a feature.
    ask({ type: 'map-ee', mapId, layerId }, { timeout: 15000 }).catch(() => {})
  }

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
    // The guide is nine pages, listed from its own manifest rather than typed
    // out here — a page added to the guide and forgotten here is a page that
    // works until you need it, standing somewhere with no signal.
    urls: ['/', '/map', '/charts', '/analysis', '/data', '/options', ...guidePaths()],
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

  /**
   * What this is actually occupying.
   *
   * The browser's own figure where it exists, because ours cannot be right: a
   * tile is fetched no-cors from a host that may send no CORS headers, and an
   * opaque response reports neither a body nor a length. Summing what the
   * responses admit to counted the shell and the dataset and almost none of the
   * tiles — which is how a few hundred saved tiles came to read as "~2 MB".
   */
  const savedBytes = computed(() => {
    const u = usage.value
    if (!u) return 0
    if (Number.isFinite(u.quota?.usage)) return u.quota.usage
    return (u.shell?.bytes || 0) + (u.data?.bytes || 0) + (u.tiles?.bytes || 0)
  })
  /** Whether that figure is the browser's or our own undercount. */
  const bytesAreMeasured = computed(() => Number.isFinite(usage.value?.quota?.usage))
  const quotaBytes = computed(() => usage.value?.quota?.quota || 0)
  const savedTiles = computed(() => usage.value?.tiles?.count || 0)
  const hasData = computed(() => (usage.value?.data?.count || 0) > 0)
  const hasShell = computed(() => (usage.value?.shell?.count || 0) > 0)

  return {
    supported, registered, online, busy, progress, usage, error, areas,
    register, refreshUsage, saveData, saveTiles, saveShell, clear,
    loadAreas, saveArea, resaveArea, deleteArea, renameArea, registerEeTemplate,
    savedBytes, savedTiles, hasData, hasShell, bytesAreMeasured, quotaBytes,
    measure, measured,
    // What the app shell and the dataset occupy once they are saved, from the
    // worker's own accounting. Zero before either has been saved, which is why
    // `measure` exists for the one that is worth knowing beforehand.
    shellBytes: computed(() => usage.value?.shell?.bytes || 0),
    dataBytes: computed(() => usage.value?.data?.bytes || 0),
  }
}
