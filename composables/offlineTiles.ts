// Which tiles cover an area, so a viewer can save a place to read it with no
// signal. Pure module — no framework, no Leaflet — so the arithmetic can be
// tested without a map.
//
// An "area" is a named, re-savable place: bounds, a zoom range, and the layers
// that were on when it was saved. Named because the thing being managed is a
// place — "the north ridge", "camp" — and a list of anonymous tile counts is
// not something anyone can decide to delete.

export interface AreaBounds {
  north: number
  south: number
  east: number
  west: number
}

export interface TileCoord {
  x: number
  y: number
  z: number
}

export interface AreaSource {
  template: string
  id: string
  volatile: boolean
  name?: string
  maxZoom?: number | null
}

export interface SavedArea {
  id: string
  name: string
  bounds: AreaBounds
  minZoom: number
  maxZoom: number
  sources: AreaSource[]
  savedAt: string
}

/** Slippy-map tile x/y for a coordinate at one zoom. */
export function tileFor(lat: number, lon: number, z: number): [number, number] {
  const n = 2 ** z
  const x = Math.floor(((lon + 180) / 360) * n)
  const rad = (Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI) / 180
  const y = Math.floor(((1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2) * n)
  return [Math.max(0, Math.min(n - 1, x)), Math.max(0, Math.min(n - 1, y))]
}

/**
 * Every tile covering a bounding box across a span of zooms.
 */
export function tilesInBounds({ north, south, east, west }: AreaBounds, minZoom: number, maxZoom: number): TileCoord[] {
  const out: TileCoord[] = []
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
 */
export function countTilesInBounds({ north, south, east, west }: AreaBounds, minZoom: number, maxZoom: number): number {
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
 */
export function tileUrl(template: string, { x, y, z }: TileCoord, subdomains = 'abc'): string {
  return template
    .replace('{s}', typeof subdomains === 'string' ? subdomains[0] : (subdomains[0] ?? 'a'))
    .replace('{z}', String(z))
    .replace('{x}', String(x))
    .replace('{y}', String(y))
    .replace('{r}', '')
}

/**
 * How many tiles a save would fetch, and roughly how many bytes.
 */
export const AVG_TILE_BYTES = 15 * 1024

export function estimateSave(tileCount: number, layerCount = 1) {
  const tiles = tileCount * layerCount
  return { tiles, bytes: tiles * AVG_TILE_BYTES }
}

/**
 * A readable name for a tile source that did not bring one.
 */
export function sourceLabel(source: Partial<AreaSource> = {}): string {
  if (source.name) return source.name
  const template = source.template || source.id || ''
  try {
    return new URL(template).host.replace(/^www\./, '')
  } catch {
    return 'Map tiles'
  }
}

/**
 * The same save, broken down by layer.
 */
export function estimatePerSource(tilesPerLayer: number, sources: Partial<AreaSource>[] = [], { bounds = null, minZoom = 0, maxZoom = 0 } = {}): any[] {
  const list = sources.length ? sources : [{ id: 'map', name: 'Map tiles' }]
  return list.map((s, i) => {
    const capped = bounds && Number.isFinite(s.maxZoom!) && s.maxZoom! < maxZoom
      ? countTilesInBounds(bounds, minZoom, Math.max(minZoom - 1, s.maxZoom!))
      : tilesPerLayer
    return {
      id: s.id || `source-${i}`,
      name: sourceLabel(s),
      tiles: capped,
      bytes: capped * AVG_TILE_BYTES,
      capped: capped !== tilesPerLayer,
    }
  })
}

/** Human-readable byte count, for a control that has to fit on a phone. */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 MB'
  const mb = bytes / (1024 * 1024)
  if (mb < 1) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  if (mb < 1024) return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`
  return `${(mb / 1024).toFixed(1)} GB`
}

export const MAX_EXTRA_ZOOM = 3
export const MAX_AREA_TILES = 20000

export const TILE_KEY_PREFIX = '/__tile/'
const EE_TILE_RE = /^https:\/\/earthengine\.googleapis\.com\/.*\/maps\/([^/]+)\/tiles\//

export function isVolatileTemplate(template: any): boolean {
  return typeof template === 'string' && EE_TILE_RE.test(template)
}

export function eeMapId(url: any): string {
  return (typeof url === 'string' && url.match(EE_TILE_RE)?.[1]) || ''
}

export function normaliseSource(source: string | Partial<AreaSource>): AreaSource {
  if (typeof source === 'string') {
    return { template: source, id: source, volatile: isVolatileTemplate(source) }
  }
  const template = source?.template || ''
  return {
    template,
    id: source?.id || template,
    volatile: source?.volatile ?? isVolatileTemplate(template),
    name: source?.name || '',
    maxZoom: Number.isFinite(source?.maxZoom) ? source.maxZoom : null,
  }
}

export function cacheKeyFor(source: any, tile: TileCoord, origin = ''): string {
  const s = normaliseSource(source)
  if (!s.volatile) return tileUrl(s.template, tile)
  return `${origin}${TILE_KEY_PREFIX}${encodeURIComponent(s.id)}/${tile.z}/${tile.x}/${tile.y}`
}

export function saveTargets(area: SavedArea, origin = ''): { url: string, key: string }[] {
  const sources = (area.sources || []).map(normaliseSource).filter((s) => s.template)
  const tiles = tilesInBounds(area.bounds, area.minZoom, area.maxZoom)
  const out: { url: string, key: string }[] = []
  for (const tile of tiles) {
    for (const source of sources) {
      if (Number.isFinite(source.maxZoom!) && tile.z > source.maxZoom!) continue
      out.push({ url: tileUrl(source.template, tile), key: cacheKeyFor(source, tile, origin) })
    }
  }
  return out
}

export function areaKeys(area: SavedArea, origin = ''): Set<string> {
  return new Set(saveTargets(area, origin).map((t) => t.key))
}

export function keysToDrop(area: SavedArea, others: SavedArea[], origin = ''): string[] {
  const doomed = areaKeys(area, origin)
  for (const other of others) {
    if (other.id === area.id) continue
    for (const key of areaKeys(other, origin)) doomed.delete(key)
  }
  return [...doomed]
}

export function areaTileCount(area: SavedArea): number {
  const layers = Math.max(1, (area.sources || []).length)
  return countTilesInBounds(area.bounds, area.minZoom, area.maxZoom) * layers
}

export const estimateAreaBytes = (area: SavedArea) => areaTileCount(area) * AVG_TILE_BYTES

export function boundsCentre({ north, south, east, west }: AreaBounds) {
  return {
    lat: (north + south) / 2,
    lon: (east + west) / 2,
  }
}

export function areaContains(area: SavedArea | null | undefined, lat: number, lon: number): boolean {
  const b = area?.bounds
  if (!b) return false
  return lat <= b.north && lat >= b.south && lon <= b.east && lon >= b.west
}

function coordLabel(value: number, [pos, neg]: [string, string]) {
  const hemisphere = value >= 0 ? pos : neg
  return `${Math.abs(value).toFixed(2)}°${hemisphere}`
}

export function suggestAreaName(bounds: AreaBounds | null | undefined): string {
  if (!bounds) return 'Saved area'
  const { lat, lon } = boundsCentre(bounds)
  return `${coordLabel(lat, ['N', 'S'])}, ${coordLabel(lon, ['E', 'W'])}`
}

export function newAreaId(): string {
  return `a${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`
}

export function makeArea({ id, name, bounds, minZoom, maxZoom, sources = [], savedAt }: any = {}): SavedArea {
  const round = (v: number) => Math.round(v * 1e5) / 1e5
  const lo = Math.max(0, Math.round(minZoom))
  
  const normSources = (sources as any[]).map(normaliseSource).filter((s) => s.template)
  const finalizedSources = normSources.map((s) => 
    s.volatile ? { ...s, template: s.template } : s
  )

  return {
    id: id || newAreaId(),
    name: (name || '').trim() || suggestAreaName(bounds),
    bounds: {
      north: round(bounds.north), south: round(bounds.south),
      east: round(bounds.east), west: round(bounds.west),
    },
    minZoom: lo,
    maxZoom: Math.max(lo, Math.round(maxZoom)),
    sources: finalizedSources,
    savedAt: savedAt || new Date().toISOString(),
  }
}
