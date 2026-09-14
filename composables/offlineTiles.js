// Which tiles cover an area, so a viewer can save a place to read it with no
// signal. Pure module — no framework, no Leaflet — so the arithmetic can be
// tested without a map.
//
// An "area" is a named, re-savable place: bounds, a zoom range, and the layers
// that were on when it was saved. Named because the thing being managed is a
// place — "the north ridge", "camp" — and a list of anonymous tile counts is
// not something anyone can decide to delete.

/** Slippy-map tile x/y for a coordinate at one zoom. */
export function tileFor(lat, lon, z) {
  const n = 2 ** z
  const x = Math.floor(((lon + 180) / 360) * n)
  const rad = (Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI) / 180
  const y = Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * n)
  return [Math.max(0, Math.min(n - 1, x)), Math.max(0, Math.min(n - 1, y))]
}

/**
 * Every tile covering a bounding box across a span of zooms.
 *
 * The count roughly quadruples per extra zoom level, which is why the caller is
 * told the number before anything is downloaded: three levels deeper than what
 * is on screen is a few hundred tiles, six is tens of thousands.
 */
export function tilesInBounds({ north, south, east, west }, minZoom, maxZoom) {
  const out = []
  for (let z = minZoom; z <= maxZoom; z += 1) {
    const [x0, y0] = tileFor(north, west, z)
    const [x1, y1] = tileFor(south, east, z)
    for (let x = Math.min(x0, x1); x <= Math.max(x0, x1); x += 1) {
      for (let y = Math.min(y0, y1); y <= Math.max(y0, y1); y += 1) out.push({ x, y, z })
    }
  }
  return out
}

/**
 * How many tiles cover a box, without building the list.
 *
 * The control recomputes this on every drag of the detail slider, and at six
 * zoom levels the list is tens of thousands of objects — allocated, counted and
 * thrown away, on each frame. The count is the product of two spans, so it does
 * not need the list to exist.
 */
export function countTilesInBounds({ north, south, east, west }, minZoom, maxZoom) {
  let n = 0
  for (let z = minZoom; z <= maxZoom; z += 1) {
    const [x0, y0] = tileFor(north, west, z)
    const [x1, y1] = tileFor(south, east, z)
    n += (Math.abs(x1 - x0) + 1) * (Math.abs(y1 - y0) + 1)
  }
  return n
}

/**
 * Fill a Leaflet-style URL template for one tile.
 *
 * Subdomain templates are resolved to the first subdomain rather than being
 * spread across them: Leaflet picks a subdomain per tile from the same list, so
 * a saved tile would sit under one host and be requested from another, and the
 * cache would miss on everything it had.
 */
export function tileUrl(template, { x, y, z }, subdomains = 'abc') {
  return template
    .replace('{s}', typeof subdomains === 'string' ? subdomains[0] : (subdomains[0] ?? 'a'))
    .replace('{z}', String(z))
    .replace('{x}', String(x))
    .replace('{y}', String(y))
    .replace('{r}', '')
}

/**
 * How many tiles a save would fetch, and roughly how many bytes.
 *
 * A basemap tile averages about 15 KB. The estimate is deliberately reported as
 * approximate — the point is to tell someone on mobile data whether they are
 * about to spend 2 MB or 200, not to be exact.
 */
export const AVG_TILE_BYTES = 15 * 1024

export function estimateSave(tileCount, layerCount = 1) {
  const tiles = tileCount * layerCount
  return { tiles, bytes: tiles * AVG_TILE_BYTES }
}

/** Human-readable byte count, for a control that has to fit on a phone. */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB'
  const mb = bytes / (1024 * 1024)
  if (mb < 1) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  if (mb < 1024) return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`
  return `${(mb / 1024).toFixed(1)} GB`
}

// How far past the current view to offer saving. Beyond three levels the tile
// count runs into the tens of thousands, which is not a "save this area" any
// more — it is a download nobody meant to start.
export const MAX_EXTRA_ZOOM = 3

// A ceiling per area, checked before anything is fetched. The slider cannot
// reach this on a normal view, but a world-sized viewport at four levels can,
// and the failure mode is someone's phone plan.
export const MAX_AREA_TILES = 20000

// ─────────────────────────────────────────────────────────────────────────────
// Layers as save sources
// ─────────────────────────────────────────────────────────────────────────────
//
// A source is a tile template plus a STABLE identity for the layer it came
// from. The identity matters because Earth Engine templates do not survive:
// getMapId() returns a URL carrying a short-lived token, so the same layer has
// a different URL every hour or so.
//
// Cached by URL, an Earth Engine tile is therefore unreachable the moment the
// token rotates — not an error, just a miss, and the layer draws blank over
// ground it has perfectly good tiles for. So a volatile source is stored under
// a synthetic key built from the layer's own id, and the worker maps a live
// request back onto it.

/** Where a volatile layer's tiles are filed. Same origin, so it cannot collide
 *  with a real request, and unmistakable in a cache listing. */
export const TILE_KEY_PREFIX = '/__tile/'

/** Earth Engine's tile endpoint, whose map id is the part that expires. */
const EE_TILE_RE = /^https:\/\/earthengine\.googleapis\.com\/.*\/maps\/([^/]+)\/tiles\//

/** True for a template whose URL will not be valid for long. */
export function isVolatileTemplate(template) {
  return typeof template === 'string' && EE_TILE_RE.test(template)
}

/** The expiring map id inside an Earth Engine tile URL, or ''. */
export function eeMapId(url) {
  return (typeof url === 'string' && url.match(EE_TILE_RE)?.[1]) || ''
}

/**
 * Accept either a bare template string or a { template, id } pair.
 *
 * Bare strings are what the catalogue layers have always passed and what an
 * older saved area holds, and for them the URL is its own identity.
 */
export function normaliseSource(source) {
  if (typeof source === 'string') {
    return { template: source, id: source, volatile: isVolatileTemplate(source) }
  }
  const template = source?.template || ''
  return {
    template,
    id: source?.id || template,
    volatile: source?.volatile ?? isVolatileTemplate(template),
  }
}

/**
 * Where one tile of one source is filed in the cache.
 *
 * For an ordinary layer this is simply the tile's own URL, so the worker serves
 * it by matching the request unchanged — no indirection for the common case.
 */
export function cacheKeyFor(source, tile, origin = '') {
  const s = normaliseSource(source)
  if (!s.volatile) return tileUrl(s.template, tile)
  return `${origin}${TILE_KEY_PREFIX}${encodeURIComponent(s.id)}/${tile.z}/${tile.x}/${tile.y}`
}

/** What a save has to fetch and where each result belongs. */
export function saveTargets(area, origin = '') {
  const sources = (area.sources || []).map(normaliseSource).filter((s) => s.template)
  const tiles = tilesInBounds(area.bounds, area.minZoom, area.maxZoom)
  const out = []
  for (const tile of tiles) {
    for (const source of sources) {
      out.push({ url: tileUrl(source.template, tile), key: cacheKeyFor(source, tile, origin) })
    }
  }
  return out
}

/** Every cache key an area occupies, as a Set. */
export function areaKeys(area, origin = '') {
  return new Set(saveTargets(area, origin).map((t) => t.key))
}

/**
 * The keys that would become unreferenced if this area went.
 *
 * Two areas that overlap share tiles, and so do two areas saved over the same
 * place at different zooms. Deleting one must not blank the other, so what
 * actually gets removed is the difference, not the area's own list.
 */
export function keysToDrop(area, others, origin = '') {
  const doomed = areaKeys(area, origin)
  for (const other of others) {
    if (other.id === area.id) continue
    for (const key of areaKeys(other, origin)) doomed.delete(key)
  }
  return [...doomed]
}

// ─────────────────────────────────────────────────────────────────────────────
// Areas
// ─────────────────────────────────────────────────────────────────────────────

/** Tiles an area occupies, counted rather than listed. */
export function areaTileCount(area) {
  const layers = Math.max(1, (area.sources || []).length)
  return countTilesInBounds(area.bounds, area.minZoom, area.maxZoom) * layers
}

/** The rough size of an area on disk.
 *
 *  Estimated, and labelled as such wherever it is shown. Tiles are fetched
 *  no-cors from hosts that do not all send CORS headers, and an opaque response
 *  reports neither a length nor a readable body — so the only honest exact
 *  number available is the browser's own total, which the portal shows
 *  separately. */
export const estimateAreaBytes = (area) => areaTileCount(area) * AVG_TILE_BYTES

export const boundsCentre = ({ north, south, east, west }) => ({
  lat: (north + south) / 2,
  lon: (east + west) / 2,
})

/** Whether a coordinate falls inside a saved area, so the map can say which of
 *  them you are standing in. */
export function areaContains(area, lat, lon) {
  const b = area?.bounds
  if (!b) return false
  return lat <= b.north && lat >= b.south && lon <= b.east && lon >= b.west
}

/** A degree-minute style label, for naming a place that has no name. */
function coordLabel(value, [pos, neg]) {
  const hemisphere = value >= 0 ? pos : neg
  return `${Math.abs(value).toFixed(2)}°${hemisphere}`
}

/**
 * A name for an area the viewer did not name.
 *
 * Coordinates rather than "Area 3": a list of saved places is something you
 * come back to weeks later, and a number tells you nothing about which one is
 * the ridge you meant.
 */
export function suggestAreaName(bounds) {
  if (!bounds) return 'Saved area'
  const { lat, lon } = boundsCentre(bounds)
  return `${coordLabel(lat, ['N', 'S'])}, ${coordLabel(lon, ['E', 'W'])}`
}

/** A short, sortable, collision-resistant id. */
export function newAreaId() {
  return `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`
}

/**
 * Build a storable area record.
 *
 * Bounds are rounded to about a metre. They come from a map viewport as full
 * doubles, and the extra digits only make two saves of the same place look
 * different in a list.
 */
export function makeArea({ id, name, bounds, minZoom, maxZoom, sources = [], savedAt } = {}) {
  const round = (v) => Math.round(v * 1e5) / 1e5
  const lo = Math.max(0, Math.round(minZoom))
  return {
    id: id || newAreaId(),
    name: (name || '').trim() || suggestAreaName(bounds),
    bounds: {
      north: round(bounds.north), south: round(bounds.south),
      east: round(bounds.east), west: round(bounds.west),
    },
    minZoom: lo,
    maxZoom: Math.max(lo, Math.round(maxZoom)),
    sources: sources.map(normaliseSource).filter((s) => s.template)
      // The template of a volatile source is not worth storing: it will have
      // expired by the time the area is next opened, and re-saving mints a new
      // one. The id is what survives and what the cache is keyed on.
      .map((s) => (s.volatile ? { id: s.id, volatile: true, template: s.template } : s)),
    savedAt: savedAt || new Date().toISOString(),
  }
}

/** How an area reads in a list: what it covers and how deep it goes. */
export function describeArea(area) {
  const tiles = areaTileCount(area)
  const { north, south, east, west } = area.bounds
  // Rough, and only ever shown as such — a degree of longitude is not a fixed
  // distance, so this is scaled by latitude to stop a polar box reading as
  // enormous.
  const midLat = ((north + south) / 2) * (Math.PI / 180)
  const kmNS = Math.abs(north - south) * 111
  const kmEW = Math.abs(east - west) * 111 * Math.cos(midLat)
  const span = (km) => (km < 10 ? `${km.toFixed(1)} km` : `${Math.round(km)} km`)
  return {
    tiles,
    bytes: estimateAreaBytes(area),
    zooms: area.minZoom === area.maxZoom
      ? `zoom ${area.minZoom}`
      : `zoom ${area.minZoom}–${area.maxZoom}`,
    extent: `${span(kmEW)} × ${span(kmNS)}`,
    layers: (area.sources || []).length,
  }
}
