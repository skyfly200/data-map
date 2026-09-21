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
// Where the list of saved areas lives, and the map ids below. Separate from the
// tiles so clearing one does not lose the other.
const META = `nexstrata-meta-${VERSION}`
const OURS = new Set([SHELL, DATA, TILES, META])

// A single save is capped rather than trimmed. The cache used to drop its
// oldest entries past a ceiling, which quietly ate tiles belonging to an area
// the viewer had saved and been told was saved — the failure only visible in
// the woods, as blank squares. Refusing an oversized save is a worse moment and
// a far better promise.
const MAX_AREA_TILES = 20000

const META_AREAS = '/__offline/areas.json'
const META_MAPIDS = '/__offline/mapids.json'
const TILE_KEY_PREFIX = '/__tile/'

// Earth Engine hands out a tile URL carrying a token that expires in hours, so
// the URL is no use as a cache key: by the time anyone reads the area offline
// the layer is asking for a different one. Tiles are filed under the layer
// instead, and this is what turns a live request back into that filing.
const EE_TILE_RE = /^https:\/\/earthengine\.googleapis\.com\/.*\/maps\/([^/]+)\/tiles\/(\d+)\/(\d+)\/(\d+)/

/** Read a JSON blob out of the meta cache. */
async function readMeta(path, fallback) {
  try {
    const cache = await caches.open(META)
    const hit = await cache.match(new Request(new URL(path, self.location.origin)))
    return hit ? await hit.json() : fallback
  } catch {
    return fallback
  }
}

async function writeMeta(path, value) {
  const cache = await caches.open(META)
  await cache.put(
    new Request(new URL(path, self.location.origin)),
    new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' } }),
  )
  return value
}

const readAreas = () => readMeta(META_AREAS, [])
const readMapIds = () => readMeta(META_MAPIDS, {})

/**
 * Remember which layer a map id belongs to.
 *
 * Bounded and newest-wins: a long session re-mints the same handful of layers
 * over and over, and only the ids actually sitting in tile URLs matter.
 */
async function rememberMapId(mapId, layerId) {
  if (!mapId || !layerId) return
  const held = await readMapIds()
  if (held[mapId] === layerId) return
  held[mapId] = layerId
  const keys = Object.keys(held)
  for (const stale of keys.slice(0, Math.max(0, keys.length - 200))) delete held[stale]
  await writeMeta(META_MAPIDS, held)
}

/** The cache key for a live Earth Engine tile request, or '' if it is not one. */
async function eeCacheKey(url) {
  const m = url.href.match(EE_TILE_RE)
  if (!m) return ''
  const [, mapId, z, x, y] = m
  const layerId = (await readMapIds())[mapId]
  if (!layerId) return ''
  return `${self.location.origin}${TILE_KEY_PREFIX}${encodeURIComponent(layerId)}/${z}/${x}/${y}`
}

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
    event.respondWith((async () => {
      const cache = await caches.open(TILES)
      // An Earth Engine tile is filed under its layer rather than its URL,
      // because the URL carries a token that will have expired by the time
      // anyone reads this area with no signal. Look there first.
      const keyed = await eeCacheKey(url)
      if (keyed) {
        const saved = await cache.match(keyed)
        if (saved) return saved
      }
      const hit = await cache.match(request)
      if (hit) return hit
      // Caught, not left to reject. A tile host that goes down — Terrascope
      // started answering ERR_HTTP2_PROTOCOL_ERROR and took ESA WorldCover with
      // it — makes this fetch throw, and a rejected promise inside respondWith
      // becomes an unhandled rejection for every tile on screen. The outcome is
      // identical either way (Leaflet sees a failed tile and fires tileerror);
      // the difference is a console nobody can read afterwards.
      return fetch(request).catch(() => Response.error())
    })())
    return
  }

  // The dataset: cache-first once saved. It is immutable for a given path — a
  // new export gets a new file — so revalidating tens of megabytes would be
  // spending someone's data to confirm what we already have.
  if (isDataset(url)) {
    event.respondWith(caches.open(DATA).then(async (cache) => {
      const hit = await cache.match(request, { ignoreSearch: true })
      if (hit) return hit
      // Same reason as the tile branch above: offline with nothing saved, this
      // rejects, and the page sees an unhandled rejection rather than a failed
      // request it can report.
      return fetch(request).catch(() => Response.error())
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
    let cache
    try {
      cache = await caches.open(SHELL)
    } catch (e) {
      // Cache API failed (e.g. UnknownError). Fall back to network.
      return fetch(request).catch(() => new Response('Cache error and offline.', { status: 503 }))
    }

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

/**
 * Save a list of things, reporting progress back to the page.
 *
 * Each entry is a URL to fetch and the key to file it under. They are usually
 * the same string; they differ for a layer whose URL expires, which is filed
 * under the layer instead so it can still be found afterwards.
 *
 * Tiles go out `no-cors`: they come from hosts that do not all send CORS
 * headers, and an opaque response still serves an <img> perfectly well. It
 * cannot be inspected, so a failure shows up as a request that resolved rather
 * than as a status — which is why the count below is of what was stored, not of
 * what was asked for.
 */
async function saveAll(entries, cacheName, port, after) {
  const cache = await caches.open(cacheName)
  let done = 0
  let failed = 0
  const total = entries.length
  const CONCURRENCY = 6

  async function worker(queue) {
    for (;;) {
      const item = queue.pop()
      if (!item) return
      const url = typeof item === 'string' ? item : item.url
      const key = typeof item === 'string' ? item : (item.key || item.url)
      try {
        if (await cache.match(key)) { done += 1; continue }
        const res = await fetch(url, { mode: cacheName === TILES ? 'no-cors' : 'cors' })
        // An opaque response has status 0 and is still worth storing.
        if (res.ok || res.type === 'opaque') await cache.put(key, res)
        else failed += 1
        done += 1
      } catch {
        failed += 1
        done += 1
      }
      if (port && done % 10 === 0) port.postMessage({ type: 'progress', done, total })
    }
  }

  const queue = [...entries]
  await Promise.all(Array.from({ length: CONCURRENCY }, () => worker(queue)))
  // Anything that has to be true before the page is told the save finished —
  // writing the area record, in particular. Told first, the page would reload
  // its list and race the record it had just asked for, and the area it just
  // saved would be missing from the list until something else refreshed it.
  if (after) await after({ done, total, failed })
  if (port) port.postMessage({ type: 'done', done, total, failed })
  return { done, total, failed }
}

/** Store or replace one area record, newest first. */
async function putArea(area) {
  const areas = await readAreas()
  const without = areas.filter((a) => a.id !== area.id)
  return writeMeta(META_AREAS, [area, ...without])
}

/**
 * Forget an area and the tiles nothing else needs.
 *
 * The keys to delete are worked out by the page, which has the other areas and
 * the arithmetic: two areas over the same valley share tiles, and deleting one
 * must not punch a hole in the other.
 */
async function dropArea(id, keys) {
  const cache = await caches.open(TILES)
  for (const key of keys || []) await cache.delete(key)
  const areas = await readAreas()
  return writeMeta(META_AREAS, areas.filter((a) => a.id !== id))
}

/**
 * What is currently held, by cache.
 *
 * Counts are exact; bytes are not, and the distinction is reported rather than
 * smoothed over. A tile is fetched no-cors from a host that may send no CORS
 * headers, and an opaque response has neither a readable body nor a
 * Content-Length — so summing what the responses admit to undercounts the tile
 * cache by approximately all of it, which is how "Saved: ~2 MB" came to sit
 * under four hundred saved tiles.
 *
 * The browser's own estimate is the honest total, so that is what the portal
 * leads with; the per-cache numbers below are counts plus whatever sizes were
 * actually legible.
 */
async function usage() {
  const out = {}
  for (const [key, name] of [['shell', SHELL], ['data', DATA], ['tiles', TILES]]) {
    const cache = await caches.open(name)
    const keys = await cache.keys()
    let bytes = 0
    let sized = 0
    for (const req of keys) {
      const res = await cache.match(req)
      const len = Number(res?.headers.get('content-length'))
      if (Number.isFinite(len) && len > 0) { bytes += len; sized += 1 }
    }
    // `sized` is how many of the entries could be measured at all. Where it is
    // far below `count`, the byte figure is a floor and the caller says so.
    out[key] = { count: keys.length, bytes, sized }
  }
  if (navigator.storage?.estimate) {
    try {
      const est = await navigator.storage.estimate()
      out.quota = { usage: est.usage, quota: est.quota }
    } catch { /* not available everywhere */ }
  }
  out.areas = (await readAreas()).length
  return out
}

self.addEventListener('message', (event) => {
  const msg = event.data || {}
  const port = event.ports?.[0]
  const reply = (payload) => port?.postMessage(payload)
  const guard = (promise) => event.waitUntil(
    promise.catch((e) => reply({ type: 'error', message: String(e?.message || e) })),
  )

  if (msg.type === 'save-tiles') {
    const entries = msg.entries || msg.urls || []
    if (entries.length > MAX_AREA_TILES) {
      // Refused whole rather than trimmed silently. A saved area that is
      // quietly missing its edges is worse than one that was never saved,
      // because it is only discovered where there is no signal to fix it.
      reply({
        type: 'error',
        message: `That area is ${entries.length.toLocaleString()} tiles, past the `
          + `${MAX_AREA_TILES.toLocaleString()} limit. Save a smaller area, or fewer zoom levels.`,
      })
      return
    }
    // The record is written after the tiles and before the page is told, so an
    // interrupted save leaves no area claiming to hold what it never fetched,
    // and a finished one is in the list the moment the page goes looking.
    guard(saveAll(entries, TILES, port, msg.area
      ? (res) => putArea({ ...msg.area, tiles: res.done - res.failed })
      : null))
  } else if (msg.type === 'save-data') {
    guard(saveAll(msg.urls || [], DATA, port))
  } else if (msg.type === 'save-shell') {
    guard(saveAll(msg.urls || [], SHELL, port))
  } else if (msg.type === 'usage') {
    guard(usage().then((u) => reply({ type: 'usage', usage: u })))
  } else if (msg.type === 'areas') {
    guard(readAreas().then((areas) => reply({ type: 'areas', areas })))
  } else if (msg.type === 'put-area') {
    guard(putArea(msg.area).then((areas) => reply({ type: 'areas', areas })))
  } else if (msg.type === 'drop-area') {
    guard(dropArea(msg.id, msg.keys).then((areas) => reply({ type: 'areas', areas })))
  } else if (msg.type === 'map-ee') {
    // The page minted a fresh Earth Engine template; remember which layer this
    // map id belongs to so its tiles can be found after the token rotates.
    guard(rememberMapId(msg.mapId, msg.layerId).then(() => reply({ type: 'ok' })))
  } else if (msg.type === 'clear') {
    const names = msg.which === 'all' ? [SHELL, DATA, TILES, META]
      : msg.which === 'tiles' ? [TILES]
      : msg.which === 'data' ? [DATA]
      : msg.which === 'shell' ? [SHELL] : []
    guard(Promise.all(names.map((n) => caches.delete(n)))
      // Clearing the tiles leaves every area record pointing at nothing, which
      // would list places as saved that are not.
      .then(() => (names.includes(TILES) && !names.includes(META)
        ? writeMeta(META_AREAS, []) : null))
      .then(() => reply({ type: 'cleared' })))
  }
})
