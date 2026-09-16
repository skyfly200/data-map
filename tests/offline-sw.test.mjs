/* The service worker, driven against a stand-in Cache API.
 *
 * This is the half of offline support that cannot be checked by looking at it.
 * The worker decides what a saved area actually holds, what a delete takes
 * away, and — the part that is easy to get wrong and impossible to notice —
 * whether a tile saved an hour ago can still be found once Earth Engine has
 * rotated the token in its URL.
 *
 * sw.js is a classic worker script rather than a module, so it is evaluated
 * here in a context carrying the globals it expects. That means it is the real
 * file being tested, not a copy of its logic.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import vm from 'node:vm'

const SW_PATH = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'sw.js')
const ORIGIN = 'https://app.test'

const keyOf = (req) => (typeof req === 'string' ? new URL(req, ORIGIN).href : req.url)

class FakeCache {
  constructor() { this.entries = new Map() }
  async match(req) { return this.entries.get(keyOf(req)) }
  async put(req, res) { this.entries.set(keyOf(req), res) }
  async delete(req) { return this.entries.delete(keyOf(req)) }
  async keys() { return [...this.entries.keys()].map((u) => new Request(u)) }
}

/** Boot sw.js with a fake environment and hand back the levers to drive it. */
function bootWorker({ fetchImpl } = {}) {
  const caches = new Map()
  const fetched = []

  const cacheStorage = {
    async open(name) {
      if (!caches.has(name)) caches.set(name, new FakeCache())
      return caches.get(name)
    },
    async keys() { return [...caches.keys()] },
    async delete(name) { return caches.delete(name) },
  }

  const listeners = {}
  const self = {
    location: new URL(`${ORIGIN}/sw.js`),
    addEventListener: (type, fn) => { listeners[type] = fn },
    skipWaiting: async () => {},
    clients: { claim: async () => {} },
  }

  const fakeFetch = fetchImpl || (async (url) => {
    fetched.push(typeof url === 'string' ? url : url.url)
    // Tiles come back opaque from hosts that send no CORS headers, which is
    // exactly why sizes cannot be read back off them.
    return new Response(null, { status: 200 })
  })

  const context = {
    self, caches: cacheStorage, fetch: fakeFetch,
    Request, Response, URL, URLSearchParams, console,
    navigator: { storage: { estimate: async () => ({ usage: 1234, quota: 99999 }) } },
    setTimeout, clearTimeout,
  }
  context.globalThis = context
  vm.createContext(context)
  vm.runInContext(readFileSync(SW_PATH, 'utf8'), context, { filename: 'sw.js' })

  /** Send the worker a message and wait for its reply. */
  const send = (msg) => new Promise((resolve, reject) => {
    const pending = []
    const progress = []
    let replied = null
    const port = {
      postMessage: (payload) => {
        if (payload?.type === 'progress') progress.push(payload)
        else if (!replied) replied = payload
      },
    }
    const event = {
      data: msg,
      ports: [port],
      waitUntil: (p) => pending.push(Promise.resolve(p)),
    }
    try { listeners.message(event) } catch (e) { reject(e); return }
    Promise.all(pending)
      .then((results) => resolve({ reply: replied, progress, result: results[0] }))
      .catch(reject)
  })

  /**
   * Send a message and resolve the moment the worker replies — which is what a
   * real MessageChannel does, and what the page does with the answer.
   *
   * `send` above waits for the worker's own waitUntil promises to settle, so it
   * cannot see work that happens after the reply has gone out. That difference
   * is the entire bug class this exists to catch: a reply posted before the
   * area record is written leaves the page refreshing a list that does not yet
   * contain what it just saved.
   */
  const sendUntilReply = (msg) => new Promise((resolve, reject) => {
    const port = {
      postMessage: (payload) => {
        if (payload?.type !== 'progress') resolve(payload)
      },
    }
    const event = { data: msg, ports: [port], waitUntil: (p) => Promise.resolve(p).catch(reject) }
    try { listeners.message(event) } catch (e) { reject(e) }
  })

  /** Ask the worker to handle a fetch, as the browser would. */
  const doFetch = (url) => new Promise((resolve) => {
    let answered = null
    const event = { request: new Request(url), respondWith: (p) => { answered = p } }
    listeners.fetch(event)
    resolve(answered ? Promise.resolve(answered) : null)
  })

  return { send, sendUntilReply, doFetch, caches, fetched, cacheStorage }
}

const tileCache = (w) => w.caches.get('nexstrata-tiles-v1')
const EE_URL = (mapId, z, x, y) =>
  `https://earthengine.googleapis.com/v1/projects/p/maps/${mapId}/tiles/${z}/${x}/${y}`

// ── Saving ───────────────────────────────────────────────────────────────────

test('a save stores each tile under the key it was given', async () => {
  const w = bootWorker()
  const { reply } = await w.send({
    type: 'save-tiles',
    entries: [
      { url: 'https://tile.test/5/1/2.png', key: 'https://tile.test/5/1/2.png' },
      { url: EE_URL('TOKEN1', 5, 1, 2), key: `${ORIGIN}/__tile/ee%3Afire/5/1/2` },
    ],
  })
  assert.equal(reply.type, 'done')
  assert.equal(reply.done, 2)
  assert.equal(reply.failed, 0)

  const cache = tileCache(w)
  assert.ok(await cache.match('https://tile.test/5/1/2.png'))
  assert.ok(await cache.match(`${ORIGIN}/__tile/ee%3Afire/5/1/2`))
  // The fetch went to the real Earth Engine URL even though the filing did not.
  assert.ok(w.fetched.includes(EE_URL('TOKEN1', 5, 1, 2)))
})

test('a plain list of urls still works, as an older page would send', async () => {
  const w = bootWorker()
  const { reply } = await w.send({ type: 'save-tiles', urls: ['https://tile.test/1/0/0.png'] })
  assert.equal(reply.done, 1)
  assert.ok(await tileCache(w).match('https://tile.test/1/0/0.png'))
})

test('an oversized save is refused whole rather than trimmed to fit', async () => {
  // The old worker dropped oldest-first past a ceiling, which ate tiles from an
  // area the viewer had been told was saved. That is only discovered where
  // there is no signal to fix it.
  const w = bootWorker()
  const entries = Array.from({ length: 20001 }, (_, i) => ({
    url: `https://tile.test/12/${i}/0.png`, key: `https://tile.test/12/${i}/0.png`,
  }))
  const { reply } = await w.send({ type: 'save-tiles', entries })
  assert.equal(reply.type, 'error')
  assert.match(reply.message, /limit/i)
  assert.equal(w.fetched.length, 0, 'nothing should have been fetched')
  assert.equal(tileCache(w), undefined)
})

test('a tile already held is not fetched again', async () => {
  const w = bootWorker()
  const entry = { url: 'https://tile.test/5/1/2.png', key: 'https://tile.test/5/1/2.png' }
  await w.send({ type: 'save-tiles', entries: [entry] })
  const before = w.fetched.length
  const { reply } = await w.send({ type: 'save-tiles', entries: [entry] })
  assert.equal(w.fetched.length, before, 'a re-save should not refetch what is held')
  assert.equal(reply.done, 1)
})

test('the area record is written only after its tiles', async () => {
  const w = bootWorker()
  const area = { id: 'a1', name: 'North ridge', bounds: {}, minZoom: 10, maxZoom: 11, sources: [] }
  await w.send({
    type: 'save-tiles', area,
    entries: [{ url: 'https://tile.test/5/1/2.png', key: 'https://tile.test/5/1/2.png' }],
  })
  const { reply } = await w.send({ type: 'areas' })
  assert.equal(reply.areas.length, 1)
  assert.equal(reply.areas[0].name, 'North ridge')
  // Counted from what was actually stored, not from what was asked for.
  assert.equal(reply.areas[0].tiles, 1)
})

test('a failed tile is not counted as saved', async () => {
  const w = bootWorker({ fetchImpl: async () => { throw new Error('offline') } })
  await w.send({
    type: 'save-tiles',
    area: { id: 'a1', name: 'x', bounds: {}, minZoom: 1, maxZoom: 1, sources: [] },
    entries: [{ url: 'https://tile.test/5/1/2.png', key: 'https://tile.test/5/1/2.png' }],
  })
  const { reply } = await w.send({ type: 'areas' })
  assert.equal(reply.areas[0].tiles, 0)
})

// ── Earth Engine tokens ──────────────────────────────────────────────────────

test('a saved Earth Engine tile is still found after the token rotates', async () => {
  // The reason any of this keying exists. Save under one map id, then ask for
  // the same tile through a different one, as the app will an hour later.
  const w = bootWorker()
  const key = `${ORIGIN}/__tile/ee%3Afire/5/1/2`

  await w.send({ type: 'map-ee', mapId: 'TOKEN1', layerId: 'ee:fire' })
  await w.send({ type: 'save-tiles', entries: [{ url: EE_URL('TOKEN1', 5, 1, 2), key }] })

  await w.send({ type: 'map-ee', mapId: 'TOKEN2', layerId: 'ee:fire' })
  const before = w.fetched.length
  const res = await w.doFetch(EE_URL('TOKEN2', 5, 1, 2))
  assert.ok(res, 'the worker should have answered')
  assert.equal(w.fetched.length, before, 'it should have come from the cache, not the network')
})

test('an unknown map id falls through to the network rather than guessing', async () => {
  const w = bootWorker()
  await w.send({ type: 'map-ee', mapId: 'TOKEN1', layerId: 'ee:fire' })
  await w.send({
    type: 'save-tiles',
    entries: [{ url: EE_URL('TOKEN1', 5, 1, 2), key: `${ORIGIN}/__tile/ee%3Afire/5/1/2` }],
  })
  const before = w.fetched.length
  await w.doFetch(EE_URL('UNSEEN', 5, 1, 2))
  assert.ok(w.fetched.length > before, 'an unmapped id must not serve another layer\'s tile')
})

test('map ids for different layers do not cross', async () => {
  const w = bootWorker()
  await w.send({ type: 'map-ee', mapId: 'T-FIRE', layerId: 'ee:fire' })
  await w.send({ type: 'map-ee', mapId: 'T-TREE', layerId: 'custom:tree-cover' })
  await w.send({
    type: 'save-tiles',
    entries: [{ url: EE_URL('T-TREE', 5, 1, 2), key: `${ORIGIN}/__tile/custom%3Atree-cover/5/1/2` }],
  })
  const before = w.fetched.length
  // The fire layer has nothing saved, so it must reach for the network even
  // though a tile at the same z/x/y is sitting in the cache.
  await w.doFetch(EE_URL('T-FIRE', 5, 1, 2))
  assert.ok(w.fetched.length > before, 'the fire layer must not be served the tree layer\'s tile')
})

test('the map id registry survives a worker restart', async () => {
  // Service workers are killed aggressively between events. An in-memory map
  // would mean the first tile request after a restart missed.
  const w = bootWorker()
  await w.send({ type: 'map-ee', mapId: 'TOKEN1', layerId: 'ee:fire' })
  await w.send({
    type: 'save-tiles',
    entries: [{ url: EE_URL('TOKEN1', 5, 1, 2), key: `${ORIGIN}/__tile/ee%3Afire/5/1/2` }],
  })

  // Boot a second worker over the same caches, which is what a restart is.
  const restarted = bootWorker()
  restarted.caches.set('nexstrata-tiles-v1', w.caches.get('nexstrata-tiles-v1'))
  restarted.caches.set('nexstrata-meta-v1', w.caches.get('nexstrata-meta-v1'))

  const before = restarted.fetched.length
  await restarted.doFetch(EE_URL('TOKEN1', 5, 1, 2))
  assert.equal(restarted.fetched.length, before, 'the restarted worker should still know the id')
})

// ── Deleting ─────────────────────────────────────────────────────────────────

test('dropping an area removes the keys it is given and forgets the record', async () => {
  const w = bootWorker()
  const keep = 'https://tile.test/5/1/2.png'
  const go = 'https://tile.test/5/9/9.png'
  await w.send({
    type: 'save-tiles',
    area: { id: 'a1', name: 'one', bounds: {}, minZoom: 5, maxZoom: 5, sources: [] },
    entries: [{ url: keep, key: keep }, { url: go, key: go }],
  })

  const { reply } = await w.send({ type: 'drop-area', id: 'a1', keys: [go] })
  assert.deepEqual(reply.areas, [])
  // The shared tile the page decided to keep is still there.
  assert.ok(await tileCache(w).match(keep))
  assert.ok(!await tileCache(w).match(go))
})

test('dropping one area leaves the other listed', async () => {
  const w = bootWorker()
  for (const id of ['a1', 'a2']) {
    await w.send({
      type: 'save-tiles',
      area: { id, name: id, bounds: {}, minZoom: 5, maxZoom: 5, sources: [] },
      entries: [{ url: `https://tile.test/5/1/${id}.png`, key: `https://tile.test/5/1/${id}.png` }],
    })
  }
  const { reply } = await w.send({ type: 'drop-area', id: 'a1', keys: [] })
  assert.deepEqual(reply.areas.map((a) => a.id), ['a2'])
})

test('re-saving an area replaces its record rather than listing it twice', async () => {
  const w = bootWorker()
  const area = { id: 'a1', name: 'North ridge', bounds: {}, minZoom: 5, maxZoom: 5, sources: [] }
  await w.send({ type: 'save-tiles', area, entries: [{ url: 'https://t/5/1/1.png', key: 'https://t/5/1/1.png' }] })
  await w.send({ type: 'put-area', area: { ...area, name: 'Renamed' } })
  const { reply } = await w.send({ type: 'areas' })
  assert.equal(reply.areas.length, 1)
  assert.equal(reply.areas[0].name, 'Renamed')
})

test('clearing the tiles also clears the areas that claimed them', async () => {
  // Otherwise the portal lists places as saved whose tiles are gone, which is
  // the one thing an offline list must never do.
  const w = bootWorker()
  await w.send({
    type: 'save-tiles',
    area: { id: 'a1', name: 'one', bounds: {}, minZoom: 5, maxZoom: 5, sources: [] },
    entries: [{ url: 'https://t/5/1/1.png', key: 'https://t/5/1/1.png' }],
  })
  await w.send({ type: 'clear', which: 'tiles' })
  const { reply } = await w.send({ type: 'areas' })
  assert.deepEqual(reply.areas, [])
})

test('clearing only the dataset leaves saved areas alone', async () => {
  const w = bootWorker()
  await w.send({
    type: 'save-tiles',
    area: { id: 'a1', name: 'one', bounds: {}, minZoom: 5, maxZoom: 5, sources: [] },
    entries: [{ url: 'https://t/5/1/1.png', key: 'https://t/5/1/1.png' }],
  })
  await w.send({ type: 'clear', which: 'data' })
  const { reply } = await w.send({ type: 'areas' })
  assert.equal(reply.areas.length, 1)
})

// ── Usage ────────────────────────────────────────────────────────────────────

test('usage reports how much of the count it could actually measure', async () => {
  // Opaque tile responses carry no length, so a byte total built from them is a
  // floor, not a size. Saying which is the difference between an honest number
  // and "~2 MB" printed under four hundred tiles.
  const w = bootWorker()
  await w.send({
    type: 'save-tiles',
    entries: [{ url: 'https://t/5/1/1.png', key: 'https://t/5/1/1.png' }],
  })
  const { reply } = await w.send({ type: 'usage' })
  assert.equal(reply.usage.tiles.count, 1)
  assert.equal(reply.usage.tiles.sized, 0, 'an opaque response cannot be measured')
  // The browser's own figure is the one worth showing.
  assert.equal(reply.usage.quota.usage, 1234)
})

test('usage counts the saved areas', async () => {
  const w = bootWorker()
  await w.send({ type: 'put-area', area: { id: 'a1', name: 'one' } })
  const { reply } = await w.send({ type: 'usage' })
  assert.equal(reply.usage.areas, 1)
})

// ── Ordinary caching is unchanged ────────────────────────────────────────────

test('a cross-origin tile that was never saved goes to the network untouched', async () => {
  const w = bootWorker()
  const before = w.fetched.length
  await w.doFetch('https://tile.test/5/1/2.png')
  assert.ok(w.fetched.length > before)
  // And it is NOT added to the cache: a normal pan must not fill the disk.
  assert.equal(tileCache(w)?.entries.size ?? 0, 0)
})

test('a non-tile cross-origin request is left entirely alone', async () => {
  const w = bootWorker()
  const answered = await w.doFetch('https://api.test/some/endpoint')
  assert.equal(answered, null, 'the worker should not respond to this at all')
})

test('the area is in the list by the time the save says it finished', async () => {
  // The reply used to be posted from inside the copy loop, before the record
  // was written. The page refreshes its list as soon as the save resolves, so
  // it raced the record it had just asked for and the area was missing from
  // the list until something else happened to reload it.
  const w = bootWorker()
  // Resolving on the reply rather than on the worker's own bookkeeping is the
  // point: it is the page's view of when the save finished.
  const reply = await w.sendUntilReply({
    type: 'save-tiles',
    area: { id: 'a1', name: 'North ridge', bounds: {}, minZoom: 5, maxZoom: 5, sources: [] },
    entries: [{ url: 'https://t/5/1/1.png', key: 'https://t/5/1/1.png' }],
  })
  assert.equal(reply.type, 'done')
  const listedAtDone = await w.sendUntilReply({ type: 'areas' })
  assert.equal(listedAtDone.areas.length, 1, 'the area should exist the moment done is reported')
})

test('two saves in a row both land', async () => {
  const w = bootWorker()
  for (const id of ['a1', 'a2']) {
    await w.send({
      type: 'save-tiles',
      area: { id, name: id, bounds: {}, minZoom: 5, maxZoom: 5, sources: [] },
      entries: [{ url: `https://t/5/1/${id}.png`, key: `https://t/5/1/${id}.png` }],
    })
  }
  const { reply } = await w.send({ type: 'areas' })
  assert.equal(reply.areas.length, 2)
})

// ── A tile host that goes down ───────────────────────────────────────────────

test('a failed cross-origin tile resolves rather than rejecting', async () => {
  // Terrascope began answering ERR_HTTP2_PROTOCOL_ERROR and took ESA WorldCover
  // with it. The worker had no catch on that path, so every tile on screen
  // became an "Uncaught (in promise) TypeError: Failed to fetch" — the console
  // filled with our noise while the actual outage scrolled past.
  //
  // The outcome is the same either way: Leaflet sees a failed tile. What must
  // not happen is the promise inside respondWith rejecting.
  const w = bootWorker({
    fetchImpl: async () => { throw new TypeError('Failed to fetch') },
  })

  const answered = await w.doFetch('https://tiles.example/9/1/2.png')
  assert.ok(answered, 'the worker should still answer the request')

  // The assertion that matters: awaiting it does not throw.
  const res = await answered
  assert.ok(res, 'a failed fetch should resolve to a response, not reject')
  assert.equal(res.type, 'error')
})

test('a dataset request with nothing cached and no network does not reject either', async () => {
  const w = bootWorker({
    fetchImpl: async () => { throw new TypeError('Failed to fetch') },
  })
  const answered = await w.doFetch(`${ORIGIN}/data/observations.geojson`)
  assert.ok(answered)
  const res = await answered
  assert.ok(res, 'an offline dataset read should resolve to a response, not reject')
})
