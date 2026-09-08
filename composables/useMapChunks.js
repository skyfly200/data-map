// Loading the map's observations by area, as the viewport reaches them.
//
// The map used to wait for all 48,233 observations, 49.7 MB of them, before it
// could draw one. On a slow connection it never appeared at all. This replaces
// that with two stages:
//
//   1. The overview lands first. Every observation, thinned and carrying only
//      the fields the map draws with, so the whole shape of the data is on
//      screen in a few hundred kilobytes rather than seven megabytes.
//   2. Cells load as the viewport touches them, at full detail. Panning into
//      new ground fetches only the new ground, and a cell already held is never
//      fetched twice.
//
// The two are merged by identity: a full record replaces its thinned twin, so a
// point that was drawn from the overview gains its enrichment when its cell
// arrives without moving or flickering.
//
// Written by scripts/mapChunks.mjs at build time. When that has not run, or the
// files are missing, `available` stays false and the caller falls back to the
// single combined file, so this is an optimisation rather than a dependency.

import { computed, ref, shallowRef } from 'vue'

const MAP_DIR = '/data/map'

/** Below this the viewport spans too much ground to fetch at full detail. */
export const DETAIL_ZOOM = 8

/** Never fetch more than this many cells for one viewport change. */
export const MAX_CELLS_PER_MOVE = 12

/** Must match cellKey in scripts/mapChunks.mjs, or nothing will ever be found. */
export function cellKeyFor(lat, lon, size) {
  return `${Math.floor(lat / size)}_${Math.floor(lon / size)}`
}

/** Every cell key covering a bounding box. */
export function cellsForBounds({ north, south, east, west }, size) {
  const out = []
  const y0 = Math.floor(south / size); const y1 = Math.floor(north / size)
  const x0 = Math.floor(west / size); const x1 = Math.floor(east / size)
  for (let y = y0; y <= y1; y += 1) for (let x = x0; x <= x1; x += 1) out.push(`${y}_${x}`)
  return out
}

/**
 * Which cells to fetch next, nearest the middle of the view first.
 *
 * A pan that reveals thirty cells should not fire thirty requests: the ones
 * under the viewer's eye matter and the corners can wait for the next move.
 * Cells the index says are empty are never requested at all.
 */
export function pickCells(keys, { loaded, index, centre, size, limit = MAX_CELLS_PER_MOVE }) {
  const wanted = keys.filter((k) => !loaded.has(k) && index?.cells?.[k]?.n)
  if (!centre) return wanted.slice(0, limit)
  const [clat, clon] = centre
  const distance = (key) => {
    const [y, x] = key.split('_').map(Number)
    return ((y + 0.5) * size - clat) ** 2 + ((x + 0.5) * size - clon) ** 2
  }
  return wanted.sort((a, b) => distance(a) - distance(b)).slice(0, limit)
}

// Module-level so every consumer of the map shares one set of loaded cells
// rather than each refetching the same ground.
const index = shallowRef(null)
const available = ref(false)
const overviewLoaded = ref(false)
const loadedCells = ref(new Set())
const pending = ref(0)
const failed = ref(0)

// Keyed by whatever identifies an observation, so a full record can replace the
// thinned one drawn from the overview. A Map preserves insertion order, which
// keeps the draw order stable as cells arrive.
const byId = shallowRef(new Map())
const version = ref(0)

let indexPromise = null
const inFlight = new Map()

function idOf(feature) {
  const p = feature?.properties || {}
  if (p.inat_id !== undefined && p.inat_id !== null) return `i${p.inat_id}`
  if (p.uuid) return `u${p.uuid}`
  const co = feature?.geometry?.coordinates || []
  return `${p.species || ''}|${p.date || ''}|${co[0]}|${co[1]}`
}

/** Merge features in, letting a fuller record win over a thinner one. */
function absorb(features, { full }) {
  const map = byId.value
  let added = 0
  for (const f of features) {
    const key = idOf(f)
    const existing = map.get(key)
    // A full record always replaces a thinned one; a thinned one never
    // replaces a full one, or panning back over loaded ground would strip the
    // enrichment off points that already had it.
    if (!existing || (full && !existing.__full)) {
      if (full) f.__full = true
      map.set(key, f)
      added += 1
    }
  }
  if (added) version.value += 1
  return added
}

export function useMapChunks() {
  async function loadIndex() {
    if (indexPromise) return indexPromise
    indexPromise = (async () => {
      try {
        const res = await fetch(`${MAP_DIR}/index.json`)
        if (!res.ok) throw new Error(String(res.status))
        index.value = await res.json()
        available.value = Boolean(index.value?.cells && index.value?.cellSize)
      } catch {
        // No chunks built. The caller falls back to the combined file.
        available.value = false
      }
      return index.value
    })()
    return indexPromise
  }

  /** The thinned everything, for an immediate first paint. */
  async function loadOverview() {
    await loadIndex()
    if (!available.value || overviewLoaded.value) return
    pending.value += 1
    try {
      const res = await fetch(`${MAP_DIR}/overview.json`)
      if (!res.ok) throw new Error(String(res.status))
      const { features = [] } = await res.json()
      absorb(features, { full: false })
      overviewLoaded.value = true
    } catch {
      failed.value += 1
    } finally {
      pending.value -= 1
    }
  }

  async function loadCell(key) {
    if (loadedCells.value.has(key) || inFlight.has(key)) return inFlight.get(key)
    pending.value += 1
    const p = (async () => {
      try {
        const res = await fetch(`${MAP_DIR}/${key}.json`)
        if (!res.ok) throw new Error(String(res.status))
        const { features = [] } = await res.json()
        absorb(features, { full: true })
        loadedCells.value = new Set([...loadedCells.value, key])
      } catch {
        failed.value += 1
      } finally {
        pending.value -= 1
        inFlight.delete(key)
      }
    })()
    inFlight.set(key, p)
    return p
  }

  /**
   * Load whatever the current view needs.
   *
   * Zoomed out past DETAIL_ZOOM the viewport spans more ground than is worth
   * fetching at full detail, and the overview is already showing all of it, so
   * nothing is requested. That is the whole reason the overview exists.
   */
  async function loadForView({ bounds, zoom, centre }) {
    await loadIndex()
    if (!available.value || !bounds) return
    if (zoom < DETAIL_ZOOM) return
    const size = index.value.cellSize
    const keys = cellsForBounds(bounds, size)
    const next = pickCells(keys, { loaded: loadedCells.value, index: index.value, centre, size })
    await Promise.all(next.map(loadCell))
  }

  const features = computed(() => {
    version.value                       // re-read when a chunk lands
    return [...byId.value.values()]
  })

  const stats = computed(() => ({
    loaded: byId.value.size,
    total: index.value?.total ?? 0,
    cells: loadedCells.value.size,
    cellsTotal: Object.keys(index.value?.cells || {}).length,
    thinned: index.value?.overview?.stride ?? 1,
  }))

  return {
    available, pending, failed, overviewLoaded, features, stats, index, version,
    loadIndex, loadOverview, loadCell, loadForView,
    busy: computed(() => pending.value > 0),
  }
}
