/* Offline support.
 *
 * The point of this app is often to be read where there is no signal — the
 * observations are of places in the woods, and the question "what is around
 * here" is asked standing in it. So the app shell, the dataset and, on request,
 * the map tiles for a chosen area are kept in the browser's cache.
 *
 * Three caches, because they have three different lifetimes:
 *   shell — the app's own JS/CSS/HTML. Replaced on every deploy, so it is
 *           revalidated in the background and the version is swept on activate.
 *   data  — the observation GeoJSON. Tens of megabytes, saved only when asked
 *           for, and worth serving from cache even when online.
 *   tiles — basemap imagery. Saved only for an area the viewer picks, and
 *           bounded so a stray pan cannot fill the disk.
 *
 * Nothing here caches on its own except the shell. Silently downloading 48 MB
 * on someone's phone data is not a feature.
 */

const VERSION = 'v1'
const SHELL_VERSION = 'v2'
const SHELL = `nexstrata-shell-${SHELL_VERSION}`
const DATA = `nexstrata-data-${VERSION}`
const TILES = `nexstrata-tiles-${VERSION}`
const OURS = new Set([SHELL, DATA, TILES])

// Beyond this the tile cache starts dropping its oldest entries. A saved area
// at three zoom levels is a few hundred tiles; this leaves room for several
// without letting an accidental "save" of a whole continent run away.
const TILE_LIMIT = 3000

self.addEventListener('install', (event) => {
  // Take over as soon as the new worker is ready rather than waiting for every
  // tab to close — a stale shell serving a new deploy's asset URLs is a blank
  // page, and the sweep below is what prevents it.
  event.waitUntil(self.skipWaiting())
})

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys()) {
      // Only our own caches, and only versions that are not the current one.
      if (name.startsWith('nexstrata-') && !OURS.has(name)) await caches.delete(name)
    }
    await self.clients.claim()
  })())
})

const isTile = (url) => /\/\d+\/\d+\/\d+(\.\w+)?(\?|$)/.test(url.pathname)
  || url.searchParams.has('TILEROW') || url.pathname.endsWith('/export')

const isDataset = (url) => url.pathname.startsWith('/data/') && url.pathname.endsWith('.geojson')

const isBuildMeta = (url) => url.pathname.startsWith('/_nuxt/builds/')

const EMPTY_MANIFEST = JSON.stringify({
  id: '',
  timestamp: 0,
  matcher: { static: {}, wildcard: {}, dynamic: {} },
  prerendered: [],
})

self.addEventListener('fetch', (event) => {
  const { request } = event
  if (request.method !== 'GET') return

  let url
  try { url = new URL(request.url) } catch { return }

  // Cross-origin: only tiles, and only ones already saved. A miss goes to the
  // network untouched — this worker never caches a tile the viewer did not ask
  // to save, so a normal pan costs nothing extra.
  if (url.origin !== self.location.origin) {
    if (!isTile(url)) return
    event.respondWith(caches.open(TILES).then(async (cache) => {
      const hit = await cache.match(request)
      if (hit) return hit
      return fetch(request)
    }))
    return
  }

  // The dataset: cache-first once saved. It is immutable for a given path — a
  // new export gets a new file — so revalidating tens of megabytes would be
  // spending someone's data to confirm what we already have.
  if (isDataset(url)) {
    event.respondWith(caches.open(DATA).then(async (cache) => {
      const hit = await cache.match(request, { ignoreSearch: true })
      return hit || fetch(request)
    }))
    return
  }

  // Nuxt app manifest / build metadata: if requested (e.g. by an older cached shell),
  // try network first, but if it returns 404 or fails, return an empty manifest
  // response so the client never encounters a 404 error.
  if (isBuildMeta(url)) {
    event.respondWith(
      fetch(request).then((res) => {
        if (res.ok) return res
        return new Response(EMPTY_MANIFEST, {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        })
      }).catch(() => new Response(EMPTY_MANIFEST, {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }))
    )
    return
  }

  // Everything else same-origin: serve from cache if we have it, and refresh in
  // the background. Offline, the cached copy is the answer; online, the viewer
  // gets an instant page and the next load gets the update.
  event.respondWith((async () => {
    const cache = await caches.open(SHELL)
    const hit = await cache.match(request)
    const network = fetch(request).then((res) => {
      if (res && res.ok && res.type === 'basic') cache.put(request, res.clone()).catch(() => {})
      return res
    }).catch(() => null)

    if (hit) { network.catch(() => {}); return hit }
    const res = await network
    if (res) return res
    // A navigation with nothing cached and no network: fall back to any shell
    // page we do have, so the app opens rather than showing the browser error.
    if (request.mode === 'navigate') {
      const fallback = await cache.match('/') || await cache.match('/map')
      if (fallback) return fallback
    }
    return new Response('Offline and not saved for offline use.', {
      status: 503, headers: { 'Content-Type': 'text/plain' },
    })
  })())
})

/** Keep a cache under a ceiling, dropping oldest-first. */
async function trim(cacheName, limit) {
  const cache = await caches.open(cacheName)
  const keys = await cache.keys()
  // Cache API returns keys in insertion order, so the front is the oldest.
  for (let i = 0; i < keys.length - limit; i += 1) await cache.delete(keys[i])
}

/**
 * Save a list of URLs, reporting progress back to the page.
 *
 * Tiles go out `no-cors`: they come from hosts that do not all send CORS
 * headers, and an opaque response still serves an <img> perfectly well. It
 * cannot be inspected, so a failure shows up as a request that resolved rather
 * than as a status — which is why the count below is of what was stored, not of
 * what was asked for.
 */
async function saveAll(urls, cacheName, port) {
  const cache = await caches.open(cacheName)
  let done = 0
  let failed = 0
  const total = urls.length
  const CONCURRENCY = 6

  async function worker(queue) {
    for (;;) {
      const url = queue.pop()
      if (!url) return
      try {
        if (await cache.match(url)) { done += 1; continue }
        const res = await fetch(url, { mode: cacheName === TILES ? 'no-cors' : 'cors' })
        // An opaque response has status 0 and is still worth storing.
        if (res.ok || res.type === 'opaque') await cache.put(url, res)
        else failed += 1
        done += 1
      } catch {
        failed += 1
        done += 1
      }
      if (port && done % 10 === 0) port.postMessage({ type: 'progress', done, total })
    }
  }

  const queue = [...urls]
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker(queue)))
  if (cacheName === TILES) await trim(TILES, TILE_LIMIT)
  if (port) port.postMessage({ type: 'done', done, total, failed })
  return { done, total, failed }
}

/** What is currently held, by cache, with a rough byte count. */
async function usage() {
  const out = {}
  for (const [key, name] of [['shell', SHELL], ['data', DATA], ['tiles', TILES]]) {
    const cache = await caches.open(name)
    const keys = await cache.keys()
    let bytes = 0
    // Reading every body to size it would be as expensive as the download, so
    // Content-Length is used where the response carries one and the rest are
    // left uncounted rather than guessed at.
    for (const req of keys) {
      const res = await cache.match(req)
      const len = Number(res?.headers.get('content-length'))
      if (Number.isFinite(len)) bytes += len
    }
    out[key] = { count: keys.length, bytes }
  }
  if (navigator.storage?.estimate) {
    try {
      const est = await navigator.storage.estimate()
      out.quota = { usage: est.usage, quota: est.quota }
    } catch { /* not available everywhere */ }
  }
  return out
}

self.addEventListener('message', (event) => {
  const msg = event.data || {}
  const port = event.ports?.[0]
  const reply = (payload) => port?.postMessage(payload)

  if (msg.type === 'save-tiles') {
    event.waitUntil(saveAll(msg.urls || [], TILES, port).catch((e) => reply({ type: 'error', message: String(e) })))
  } else if (msg.type === 'save-data') {
    event.waitUntil(saveAll(msg.urls || [], DATA, port).catch((e) => reply({ type: 'error', message: String(e) })))
  } else if (msg.type === 'save-shell') {
    event.waitUntil(saveAll(msg.urls || [], SHELL, port).catch((e) => reply({ type: 'error', message: String(e) })))
  } else if (msg.type === 'usage') {
    event.waitUntil(usage().then((u) => reply({ type: 'usage', usage: u })))
  } else if (msg.type === 'clear') {
    const names = msg.which === 'all' ? [SHELL, DATA, TILES]
      : msg.which === 'tiles' ? [TILES]
      : msg.which === 'data' ? [DATA]
      : msg.which === 'shell' ? [SHELL] : []
    event.waitUntil(Promise.all(names.map((n) => caches.delete(n))).then(() => reply({ type: 'cleared' })))
  }
})
