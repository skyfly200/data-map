// Which tiles cover an area, so a viewer can save a place to read it with no
// signal. Pure module — no framework, no Leaflet — so the arithmetic can be
// tested without a map.

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
