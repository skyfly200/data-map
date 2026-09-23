// The reference layers stacked over the basemap: imagery and modelled rasters
// from public services, for context the observations cannot supply themselves.
//
// Distinct from the heatmaps in composables/useMapHeatmaps.js. A heatmap is our
// own numbers binned into a grid; a layer here is somebody else's raster, drawn
// everywhere rather than only where somebody recorded a find. That difference is
// the point of having both: the heatmap tells you what the finds say about the
// ground, the layer tells you what the ground is.
//
// Pure module — no framework imports — so the catalogue and the URL builders can
// be unit-tested without a map.

export interface LayerLegendClass {
  color: string
  label: string
  aliases?: string[]
}

export interface LayerLegendRamp {
  type: 'ramp'
  unit: string
  min: string
  max: string
  stops: string[]
}

export interface LayerLegendClasses {
  type: 'classes'
  items: LayerLegendClass[]
}

export type LayerLegend = LayerLegendRamp | LayerLegendClasses

export interface TileLayer {
  name: string
  group: string
  url?: string
  arcgis?: string
  layers?: string
  attribution: string
  maxZoom: number
  opacity?: number
  // Suggested CSS mix-blend-mode when drawn over a basemap. Applied automatically
  // on first toggle; the user can override it via the layer manager.
  blend?: string
  note?: string
  time?: boolean
  lag?: number
  legend?: LayerLegend
}

/** Web Mercator half-extent, in metres. */
const MERC_R = 20037508.342789244

/**
 * EPSG:3857 bounds of an XYZ tile, as [xmin, ymin, xmax, ymax].
 *
 * ArcGIS MapServer services render on demand from a bbox rather than serving a
 * pre-cut tile pyramid, so a tile has to be asked for by its extent.
 */
export function tileBounds(x: number, y: number, z: number): [number, number, number, number] {
  const span = (2 * MERC_R) / 2 ** z
  const xmin = -MERC_R + x * span
  const ymax = MERC_R - y * span
  return [xmin, ymax - span, xmin + span, ymax]
}

/** One tile's worth of an ArcGIS MapServer `export` request. */
export function arcgisExportUrl(service: string, x: number, y: number, z: number, options: { size?: number, layers?: string } = { size: 256, layers: '' }): string {
  const { size = 256, layers = '' } = options
  const [xmin, ymin, xmax, ymax] = tileBounds(x, y, z)
  const q = new URLSearchParams({
    bbox: `${xmin},${ymin},${xmax},${ymax}`,
    bboxSR: '3857', imageSR: '3857',
    size: `${size},${size}`,
    format: 'png32', transparent: 'true', f: 'image',
  })
  if (layers) q.set('layers', layers)
  return `${service}/export?${q}`
}

/**
 * A NASA GIBS tile URL.
 *
 * GIBS publishes a REST WMTS endpoint whose path is fixed except for the layer,
 * the date and the matrix set — every layer is reachable by filling those in,
 * which is what makes a catalogue of them worth having rather than a hand-built
 * URL each. `date` is ISO yyyy-mm-dd; layers that do not vary in time still take
 * one and ignore it.
 */
export function gibsUrl(layer: string, options: { level?: number, format?: string, date?: string } = { level: 6, format: 'png', date: '{date}' }): string {
  const { level = 6, format = 'png', date = '{date}' } = options
  return 'https://gibs.earthdata.nasa.gov/wmts/epsg3857/best'
    + `/${layer}/default/${date}/GoogleMapsCompatible_Level${level}/{z}/{y}/{x}.${format}`
}

/**
 * A GIBS layer's URL and its tile ceiling, from one number.
 *
 * The matrix set in the URL *is* the ceiling: `GoogleMapsCompatible_Level7`
 * contains zooms 0–7 and nothing above, and GIBS answers a request past the top
 * with a 400 rather than a blank tile. The catalogue used to carry the level and
 * the ceiling as two independent numbers, and they had drifted — every GIBS
 * layer here declared a ceiling two levels above its own matrix, so panning past
 * it produced a screenful of 400s and a layer that silently stopped drawing.
 *
 * Derived rather than written twice, so the two cannot disagree again. The
 * ceiling is where GIBS stops having tiles; MushroomMap passes it as
 * maxNativeZoom and lets Leaflet upscale beyond it, so the layer stays on screen
 * rather than vanishing when you zoom in.
 */
export function gibs(layer: string, level: number, options: { format?: string } = { format: 'png' }) {
  const { format = 'png' } = options
  return { url: gibsUrl(layer, { level, format }), maxZoom: level }
}

/**
 * The date to ask a satellite product for.
 *
 * Every one of these has latency — an 8-day NDVI composite for today does not
 * exist yet, and asking for it returns blank tiles that read as "no vegetation"
// rather than "not processed". Backing off by the product's own lag is what
// keeps a layer from lying about the present.
 */
export function layerDate(lagDays: number, now = new Date()): string {
  const d = new Date(now.getTime() - lagDays * 86400000)
  return d.toISOString().slice(0, 10)
}


/**
 * The catalogue.
 *
 * `legend` is what the key on the map draws: a `ramp` of color stops with its
 * end labels, or a list of `classes`. A layer without one is a layer a reader
 * cannot interpret, so every raster here has one — the imagery and label layers,
 * which are pictures rather than measurements, do not need one.
 *
 * `lag` is how many days back to ask for, `time: true` marks a layer whose date
 * the viewer can move.
 */
export const TILE_LAYERS: TileLayer[] = [
  // ── Terrain ───────────────────────────────────────────────────────────────
  {
    name: 'Hillshade', group: 'Terrain',
    url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Esri', maxZoom: 16, opacity: 0.65, blend: 'multiply',
  },
  {
    name: 'USGS topo', group: 'Terrain',
    url: 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}',
    attribution: 'USGS The National Map', maxZoom: 16, opacity: 0.75, blend: 'normal',
  },
  {
    name: 'USGS imagery', group: 'Terrain',
    url: 'https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}',
    attribution: 'USGS The National Map', maxZoom: 16, opacity: 0.80, blend: 'normal',
  },
  {
    name: 'OpenTopoMap relief', group: 'Terrain',
    url: 'https://tile.opentopomap.org/{z}/{x}/{y}.png',
    attribution: 'OpenTopoMap (CC-BY-SA)', maxZoom: 17, opacity: 0.55, blend: 'multiply',
  },

  // ── Weather ───────────────────────────────────────────────────────────────
  {
    name: 'Radar (US, now)', group: 'Weather',
    arcgis: 'https://mapservices.weather.noaa.gov/eventdriven/rest/services/radar/radar_base_reflectivity/MapServer',
    attribution: 'NOAA / NWS', maxZoom: 12, opacity: 0.70, blend: 'screen',
    note: 'Live NEXRAD base reflectivity over the US. Reflectivity is not rainfall: hail, bright banding and ground clutter all show up as returns.',
    legend: {
      type: 'ramp', unit: 'dBZ', min: '5', max: '75',
      stops: ['#04e9e7', '#019ff4', '#02fd02', '#fdf802', '#fd9500', '#fd0000', '#bc0000', '#f800fd'],
    },
  },
  {
    name: 'Rain past 24h (US)', group: 'Weather',
    arcgis: 'https://mapservices.weather.noaa.gov/raster/rest/services/obs/rfc_qpe/MapServer',
    layers: 'show:25',
    attribution: 'NOAA / NWS River Forecast Centers', maxZoom: 12, opacity: 0.65, blend: 'screen',
    note: 'Quantitative precipitation estimate from gauge-corrected radar, so it is an estimate of what fell, not a gauge reading. US only.',
    legend: {
      type: 'ramp', unit: 'in', min: '0.01', max: '8+',
      stops: ['#c9e8c0', '#7fc97f', '#2b8cbe', '#253494', '#7a0177', '#c51b8a', '#fd8d3c', '#bd0026'],
    },
  },
  {
    name: 'Rain past 7d (US)', group: 'Weather',
    arcgis: 'https://mapservices.weather.noaa.gov/raster/rest/services/obs/rfc_qpe/MapServer',
    layers: 'show:53',
    attribution: 'NOAA / NWS River Forecast Centers', maxZoom: 12, opacity: 0.65, blend: 'screen',
    note: 'Seven-day gauge-corrected QPE accumulation. Useful for reading how wet the ground has been through a fruiting window rather than a single storm. US only.',
    legend: {
      type: 'ramp', unit: 'in', min: '0.01', max: '16+',
      stops: ['#c9e8c0', '#7fc97f', '#2b8cbe', '#253494', '#7a0177', '#c51b8a', '#fd8d3c', '#bd0026'],
    },
  },
  {
    name: 'Rain past 30d (US)', group: 'Weather',
    arcgis: 'https://mapservices.weather.noaa.gov/raster/rest/services/obs/rfc_qpe/MapServer',
    layers: 'show:65',
    attribution: 'NOAA / NWS River Forecast Centers', maxZoom: 12, opacity: 0.65, blend: 'screen',
    note: 'Thirty-day QPE accumulation, a proxy for seasonal soil wetting ahead of a flush. US only.',
    legend: {
      type: 'ramp', unit: 'in', min: '0.01', max: '30+',
      stops: ['#c9e8c0', '#7fc97f', '#2b8cbe', '#253494', '#7a0177', '#c51b8a', '#fd8d3c', '#bd0026'],
    },
  },
  {
    name: 'Rainfall (global)', group: 'Weather',
    ...gibs('IMERG_Precipitation_Rate', 6),
    attribution: 'NASA GIBS / GPM IMERG', opacity: 0.70, blend: 'screen', time: true, lag: 1,
    note: 'Satellite precipitation rate at ~10 km. Global, but coarse: a cell is bigger than most of the places on this map, so read it as weather, not as a shower.',
    legend: {
      type: 'ramp', unit: 'mm/hr', min: '0.1', max: '30',
      stops: ['#a0d3f5', '#3d8fd1', '#2fb457', '#e8e337', '#ea9c2c', '#d43d20'],
    },
  },
  {
    name: 'Land surface temp', group: 'Weather',
    ...gibs('MODIS_Terra_Land_Surface_Temp_Day', 7),
    attribution: 'NASA GIBS / MODIS Terra', opacity: 0.65, blend: 'screen', time: true, lag: 3,
    note: 'Daytime skin temperature of the ground itself, not air temperature: bare rock in sun reads far hotter than the air above it. Cloudy days are gaps.',
    legend: {
      type: 'ramp', unit: '°C', min: '−25', max: '45',
      stops: ['#3b1a8c', '#2a6fb0', '#48b3a8', '#c8dd52', '#e8a020', '#b81414'],
    },
  },

  // ── Vegetation ────────────────────────────────────────────────────────────
  {
    name: 'NDVI (greenness)', group: 'Vegetation',
    ...gibs('MODIS_Terra_NDVI_8Day', 9),
    attribution: 'NASA GIBS / MODIS Terra', opacity: 0.65, blend: 'multiply', time: true, lag: 10,
    note: 'An 8-day composite ending on the chosen date, at 250 m. Dense conifer and dense broadleaf both saturate near the top, so it separates bare from green far better than it separates forest types.',
    legend: {
      type: 'ramp', unit: 'NDVI', min: '−0.2', max: '1.0',
      stops: ['#bfa06a', '#dfd39a', '#c3d17a', '#7fb04a', '#3d8228', '#14520f'],
    },
  },

  // ── Context ───────────────────────────────────────────────────────────────
  {
    name: 'Place labels', group: 'Context',
    url: 'https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Esri', maxZoom: 16, opacity: 0.80, blend: 'normal',
  },
  {
    name: 'Hiking trails', group: 'Context',
    url: 'https://tile.waymarkedtrails.org/hiking/{z}/{x}/{y}.png',
    attribution: 'waymarkedtrails.org · OpenStreetMap (CC-BY-SA)', maxZoom: 18, opacity: 0.80, blend: 'normal',
    note: 'Waymarked hiking routes from OpenStreetMap. Not a complete trail map: an unmapped path is missing, not absent.',
  },
  {
    name: 'Land ownership (US)', group: 'Context',
    url: 'https://gis.blm.gov/arcgis/rest/services/lands/BLM_Natl_SMA_Cached_without_PriUnk/MapServer/tile/{z}/{y}/{x}',
    attribution: 'BLM Surface Management Agency', maxZoom: 16, opacity: 0.50, blend: 'multiply',
    note: 'US federal and state land, by managing agency. Unpainted means private or unrecorded, not necessarily open. Always confirm access before relying on it.',
    legend: {
      type: 'classes',
      items: [
        { color: '#f5d76e', label: 'BLM' },
        { color: '#2e7d32', label: 'Forest Service' },
        { color: '#7cb342', label: 'National Park Service' },
        { color: '#4fc3c3', label: 'Fish & Wildlife Service' },
        { color: '#9e9e9e', label: 'Dept. of Defense' },
        { color: '#c48ad4', label: 'State' },
        { color: '#d98880', label: 'Tribal' },
      ],
    },
  },
]

/** Layers whose date the viewer can move. */
export const TIME_LAYERS = TILE_LAYERS.filter((l) => l.time).map((l) => l.name)

/**
 * Filter grouped layers for the manager's search box.
 */
export function filterLayerGroups(groups: any[], query: string | null | undefined) {
  const q = (query || '').trim().toLowerCase()
  if (!q) return groups
  return groups
    .map((g) => {
      const label = (g.label || '').toLowerCase()
      if (label.startsWith(q)) return g
      return { ...g, items: g.items.filter((o: any) => (o.name || '').toLowerCase().includes(q)) }
    })
    .filter((g: any) => g.items.length)
}

/**
 * The provider a layer's data comes from, as a short label to group by.
 */
export function layerSource(attribution = ''): string {
  const a = String(attribution)
  const rules: [RegExp, string][] = [
    [/Sentinel/i, 'Copernicus Sentinel-2'],
    [/TreeMap/i, 'USFS TreeMap'],
    [/MTBS/i, 'USFS MTBS'],
    [/FIRMS/i, 'NASA FIRMS'],
    [/MODIS/i, 'NASA MODIS'],
    [/SMAP/i, 'NASA SMAP'],
    [/SRTM/i, 'NASA SRTM'],
    [/MERIT/i, 'MERIT Hydro'],
    [/Hansen/i, 'Hansen GFC'],
    [/SOLUS/i, 'USDA SOLUS100'],
    [/OpenLandMap/i, 'OpenLandMap'],
    [/\bGAP\b|LANDFIRE/i, 'USGS GAP'],
    [/WorldCover/i, 'ESA WorldCover'],
    [/GIBS|GPM|IMERG/i, 'NASA GIBS'],
    [/NOAA|NWS/i, 'NOAA / NWS'],
    [/National Map|\bUSGS\b/i, 'USGS'],
    [/OpenTopoMap/i, 'OpenTopoMap'],
    [/waymarked|OpenStreetMap/i, 'OpenStreetMap'],
    [/\bBLM\b/i, 'BLM'],
    [/Esri/i, 'Esri'],
  ]
  for (const [re, label] of rules) if (re.test(a)) return label
  return a.split(/[/(]/)[0].trim() || 'Other'
}

/**
 * What kind of raster a layer is, as a label to group by.
 */
export function layerDataType(legend: LayerLegend | null | undefined): string {
  if (!legend) return 'Basemap & imagery'
  if (legend.type === 'ramp') return 'Continuous raster'
  if (legend.type === 'classes') return 'Categorical raster'
  return 'Other raster'
}

/** Catalogue entries grouped for the layers control, in declaration order. */
export function layerGroups() {
  const groups: { name: string, layers: TileLayer[] }[] = []
  for (const l of TILE_LAYERS) {
    let g = groups.find((x) => x.name === l.group)
    if (!g) { g = { name: l.group, layers: [] }; groups.push(g) }
    g.layers.push(l)
  }
  return groups
}
