// Heatmaps for the map: instead of one mark per observation, bin the
// observations into a grid and shade each cell by a summary statistic.
//
// "Heatmap" rather than "overlay", which the map also uses for the reference
// tile services (hillshade, land ownership, rainfall radar). Those come from
// somebody else's server and sit under the data; these are computed here from
// the observations themselves. Calling both "overlay" made two unrelated
// controls read as the same one.
//
// This answers questions the point layer cannot. 48k overlapping dots show where
// the data is dense; a grid shows how many species live in an area, when in the
// year each area fruits, and what the terrain and weather are like across it.
//
// A caveat runs through all of this and is surfaced in the UI: iNaturalist
// records are *observations*, not surveys. Dense cells are partly dense because
// people walk there. The `season` and `hotspots` modes normalise by each cell's
// own total, which cancels most of that bias — a cell's seasonal shape does not
// depend on how many people visited it, only on when they found things.

import { computed, ref } from 'vue'
import { mix, normaliseStops, rampColor } from './ramps'
import { categoryColor, hasValue } from '~/composables/useObservations'
import { cellAt, cellKeyAt, CELL_SHAPES } from '~/composables/gridCells'
import { ALL_NUMERIC } from '~/composables/useChartFields'
import { fieldValue } from '~/composables/statistics'
import type { ObservationFeature } from '~/composables/useObservations'

// CELL_SHAPES is deliberately not re-exported: Nuxt auto-imports every
// composables/ export by name, and a second export of the same symbol makes
// which module wins depend on scan order. It is handed out on the returned
// object instead, and gridCells is the one place it is declared.

export interface CellSize {
  value: number
  label: string
}

// Grid resolutions in degrees of latitude, with a rough ground distance.
export const CELL_SIZES: CellSize[] = [
  { value: 0.001, label: '~100 m' },
  { value: 0.0025, label: '~250 m' },
  { value: 0.005, label: '~500 m' },
  { value: 0.01, label: '~1 km' },
  { value: 0.02, label: '~2 km' },
  { value: 0.05, label: '~5 km' },
  { value: 0.1, label: '~11 km' },
  { value: 0.25, label: '~28 km' },
]

/** Today as a day of the year, which is where the seasonal heatmaps start. */
export function todayOfYear(now = new Date()): number {
  const start = new Date(Date.UTC(now.getUTCFullYear(), 0, 0))
  return Math.min(365, Math.max(1, Math.floor((now - start) / 86400000)))
}

export interface FieldMode {
  key: string
  group: string
  ramp: string[] | null
  note: string
  circular?: boolean
}

// The enriched per-observation fields worth averaging across a cell. Each one
// becomes a heatmap of its own: the map already carries the value at every
// point, and a grid of cell means is what turns 48k scattered readings into a
// surface you can actually read.
export const FIELD_MODES: FieldMode[] = [
  { key: 'rain7', group: 'Weather', ramp: ['#e8f4fb', '#01579b'],
    note: 'Mean rain over the 7 days before each find in the cell. Sampled at the observations, so cells with no finds are blank rather than dry.' },
  { key: 'tavg', group: 'Weather', ramp: ['#e3f2fd', '#b71c1c'],
    note: 'Mean daily average temperature at the finds in this cell.' },
  { key: 'soil_moisture', group: 'Ground', ramp: ['#fbf3e4', '#00695c'],
    note: 'Mean modelled soil moisture at the finds in this cell.' },
  { key: 'water_retention', group: 'Terrain', ramp: ['#f1f8e9', '#1a237e'],
    note: 'Topographic wetness index, how much upslope area drains through here. High means water collects.' },
  { key: 'slope', group: 'Terrain', ramp: ['#f5f5f5', '#4e342e'],
    note: 'Mean ground steepness at the finds in this cell.' },
  { key: 'aspect', group: 'Terrain', ramp: null, circular: true,
    note: 'Mean compass direction the ground faces, averaged as vectors so north does not average to south. Color is the direction itself, not a magnitude.' },
  { key: 'solar_exposure', group: 'Exposure', ramp: ['#fffde7', '#e65100'],
    note: 'Modelled sun the ground receives, from slope and aspect. High is an open south face; low is a shaded draw.' },
  { key: 'wind_exposure', group: 'Exposure', ramp: ['#eceff1', '#263238'],
    note: 'How exposed the ground is to wind, from terrain shape. High is a ridge; low is sheltered.' },
  { key: 'ndvi', group: 'Vegetation', ramp: ['#f4e9d8', '#1b5e20'],
    note: 'Greenness from satellite imagery near each find. High is dense living vegetation.' },
  { key: 'ndmi', group: 'Vegetation', ramp: ['#fdf3e7', '#004d61'],
    note: 'Moisture in the vegetation canopy from satellite imagery. High is a wet canopy.' },
  { key: 'elevation', group: 'Terrain', ramp: ['#f0f4c3', '#3e2723'],
    note: 'Mean elevation of the finds in this cell.' },
]

const FIELD_LABELS: Record<string, string> = Object.fromEntries(ALL_NUMERIC.map((f) => [f.key, f.label]))
const FIELD_MODE_KEYS = new Set(FIELD_MODES.map((f) => f.key))

export interface HeatmapMode {
  key: string
  label: string
  kind: 'none' | 'sequential' | 'categorical' | 'vector' | 'field'
  group?: string
  note: string
  field?: string
  circular?: boolean
  fieldRamp?: string[] | null
  windNote?: string
}

export const HEATMAP_MODES: HeatmapMode[] = [
  { key: '', label: 'None', kind: 'none', note: '' },
  {
    key: 'maxent', label: 'MaxEnt Suitability', kind: 'sequential', group: 'MaxEnt',
    note: 'Predicted habitat suitability from a trained MaxEnt model. Select a model run and configure the visualization below.',
  },
  {
    key: 'density', label: 'Observation density', kind: 'sequential', group: 'Observations',
    note: 'Observations per cell. Reflects where people look as much as where mushrooms are.',
  },
  {
    key: 'richness', label: 'Species richness', kind: 'sequential', group: 'Observations',
    note: 'Distinct species recorded in each cell.',
  },
  {
    key: 'season', label: 'Seasonal activity', kind: 'sequential', group: 'Observations',
    note: "Share of the cell's own finds that fall in the selected window, effort-neutral, so it shows when an area fruits.",
  },
  {
    key: 'hotspots', label: 'In-season hotspots', kind: 'sequential', group: 'Observations',
    note: 'Where finds have actually concentrated in this window, weighted by how well-sampled the cell is. A record of past finds, not a forecast.',
  },
  {
    key: 'common', label: 'Most common species', kind: 'categorical', group: 'Observations',
    note: 'The most-recorded species in each cell, colored to match the points.',
  },
  {
    key: 'land_cover', label: 'Land cover', kind: 'categorical', group: 'Observations',
    note: 'The most common land-cover class recorded across the finds in each cell.',
  },
  {
    key: 'wind', label: 'Wind / aspect vectors', kind: 'vector', group: 'Terrain',
    note: 'Arrows point the way slopes face; length is how consistent the aspect is, color is wind exposure.',
    windNote: 'Arrows point downwind (ERA5 10 m mean); length is wind speed.',
  },
  // Cell means of the enriched fields, generated so a new enrichment column
  // becomes a readable layer by being named once above.
  ...FIELD_MODES.map((f) => ({
    key: `f:${f.key}`,
    label: FIELD_LABELS[f.key] || f.key,
    kind: 'field' as const, group: f.group,
    field: f.key, circular: !!f.circular, fieldRamp: f.ramp, note: f.note,
  })),
]

/** Numeric field behind a mode key, or null for the built-in modes. */
export function fieldOf(mode: string): string | null {
  return typeof mode === 'string' && mode.startsWith('f:') ? mode.slice(2) : null
}

const MIN_VECTOR_CELL = 0.1

export interface RampPreset {
  key: string
  label: string
  ramp: string[] | null
}

export const DEFAULT_RAMPS: Record<string, string[]> = {
  density: ['#e8f1fb', '#0b3d91'],
  richness: ['#eef7ec', '#1b5e20'],
  season: ['#fff3e0', '#bf360c'],
  hotspots: ['#f3e9fb', '#4a148c'],
  wind: ['#9ecae1', '#08306b'],
  ...Object.fromEntries(FIELD_MODES.filter((f) => f.ramp).map((f) => [`f:${f.key}`, f.ramp!])),
}

export const RAMP_PRESETS: RampPreset[] = [
  { key: 'default', label: 'Per heatmap (default)', ramp: null },
  { key: 'blue', label: 'Blue', ramp: ['#e8f1fb', '#0b3d91'] },
  { key: 'green', label: 'Green', ramp: ['#eef7ec', '#1b5e20'] },
  { key: 'warm', label: 'Warm', ramp: ['#fff3e0', '#bf360c'] },
  { key: 'purple', label: 'Purple', ramp: ['#f3e9fb', '#4a148c'] },
  { key: 'viridis', label: 'Viridis-ish', ramp: ['#fde725', '#440154'] },
  { key: 'mono', label: 'Greyscale', ramp: ['#f2f2f2', '#1a1a1a'] },
]

export const heatmapRampKey = ref('default')
export const heatmapRampCustom = ref<string[] | null>(null)

/** The ramp actually used for a mode, after any override. */
export function rampFor(mode: string): string[] {
  if (heatmapRampKey.value === 'custom') {
    const custom = normaliseStops(heatmapRampCustom.value)
    if (custom) return custom
  }
  const preset = RAMP_PRESETS.find((p) => p.key === heatmapRampKey.value)
  if (preset?.ramp) return preset.ramp
  return DEFAULT_RAMPS[mode] || DEFAULT_RAMPS.density
}

const RAMPS = new Proxy({}, {
  get: (_t, mode) => rampFor(String(mode)),
  has: (_t, mode) => String(mode) in DEFAULT_RAMPS,
})

export const hexLerp = mix

export function dayDistance(a: number, b: number): number {
  const d = Math.abs(a - b) % 365
  return Math.min(d, 365 - d)
}

export function bearingColor(deg: number): string {
  const h = ((deg % 360) + 360) % 360
  return `hsl(${h.toFixed(0)}, 68%, 48%)`
}

interface HeatmapCell {
  key: string
  lat: number
  lon: number
  polygon: [number, number][]
  lat0: number
  lon0: number
  lat1: number
  lon1: number
  n: number
  inWindow: number
  species: Map<string, number>
  cover: Map<string, number>
  ax: number
  ay: number
  aspectN: number
  wu: number
  wv: number
  windN: number
  expSum: number
  expN: number
  fields: Map<string, { sum: number, x: number, y: number, n: number }>
  // Result fields
  raw?: number
  t?: number
  color?: string
  value?: number
  label?: string
  samples?: number
  dx?: number
  dy?: number
  magnitude?: number
  exposure?: number
}

export function useMapHeatmaps() {
  const cloud = safeCloudSync()
  const mode = useState('map-overlay-mode', () => '')
  const cellSize = useState('map-overlay-cell', () => 0.05)
  const cellShape = useState('map-overlay-shape', () => 'hex')
  const seasonDay = useState('map-overlay-day', () => todayOfYear())
  const seasonWindow = useState('map-overlay-window', () => 14)
  const heatmapOpacity = useState('map-heatmap-opacity', () => 0.55)
  const tileOpacity = useState('map-tile-opacity', () => 1)

  // MaxEnt suitability display state (HEAT-2, HEAT-3, HEAT-4)
  const maxentVizMode = useState<'probability' | 'binary'>('maxent-viz-mode', () => 'probability')
  const maxentThreshold = useState<number>('maxent-threshold', () => 0.5)
  const maxentShowCI = useState<boolean>('maxent-show-ci', () => false)
  const maxentModelId = useState<string>('maxent-model-id', () => '')

  const activeMode = computed(() => HEATMAP_MODES.find((m) => m.key === mode.value) || HEATMAP_MODES[0])

  const groupedModes = computed(() => {
    const groups: { label: string, modes: HeatmapMode[] }[] = []
    for (const m of HEATMAP_MODES) {
      if (!m.group) continue
      let g = groups.find((x) => x.label === m.group)
      if (!g) { g = { label: m.group, modes: [] }; groups.push(g) }
      g.modes.push(m)
    }
    return groups
  })

  function persist() {
    if (!import.meta.client) return
    try {
      localStorage.setItem('map-overlay', JSON.stringify({
        mode: mode.value, cellSize: cellSize.value, cellShape: cellShape.value,
        seasonDay: seasonDay.value, seasonWindow: seasonWindow.value,
        rampKey: heatmapRampKey.value, rampCustom: heatmapRampCustom.value,
        heatmapOpacity: heatmapOpacity.value, tileOpacity: tileOpacity.value,
        maxentVizMode: maxentVizMode.value, maxentThreshold: maxentThreshold.value,
        maxentShowCI: maxentShowCI.value, maxentModelId: maxentModelId.value,
      }))
      cloud?.schedulePush()
    } catch { /* ignore */ }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem('map-overlay') || 'null')
      if (!saved) return
      const RENAMED = { fruiting: 'hotspots', dominant: 'common' }
      const savedMode = RENAMED[saved.mode] || saved.mode
      if (HEATMAP_MODES.some((m) => m.key === savedMode)) mode.value = savedMode
      if (CELL_SIZES.some((c) => c.value === saved.cellSize)) cellSize.value = saved.cellSize
      if (CELL_SHAPES.some((s) => s.value === saved.cellShape)) cellShape.value = saved.cellShape
      if (Number.isFinite(saved.seasonDay)) seasonDay.value = saved.seasonDay
      if (Number.isFinite(saved.seasonWindow)) seasonWindow.value = saved.seasonWindow
      if (Number.isFinite(saved.heatmapOpacity)) {
        heatmapOpacity.value = Math.min(1, Math.max(0.05, saved.heatmapOpacity))
      }
      if (Number.isFinite(saved.tileOpacity)) {
        tileOpacity.value = Math.min(1, Math.max(0.05, saved.tileOpacity))
      }
      if (saved.rampKey === 'custom' || RAMP_PRESETS.some((r) => r.key === saved.rampKey)) {
        heatmapRampKey.value = saved.rampKey
      }
      if (Array.isArray(saved.rampCustom) && saved.rampCustom.length === 2
        && saved.rampCustom.every((c) => /^#[0-9a-f]{6}$/i.test(c))) {
        heatmapRampCustom.value = normaliseStops(saved.rampCustom)
      }
      if (saved.maxentVizMode === 'probability' || saved.maxentVizMode === 'binary') {
        maxentVizMode.value = saved.maxentVizMode
      }
      if (Number.isFinite(saved.maxentThreshold) && saved.maxentThreshold >= 0.05 && saved.maxentThreshold <= 0.95) {
        maxentThreshold.value = saved.maxentThreshold
      }
      if (typeof saved.maxentShowCI === 'boolean') maxentShowCI.value = saved.maxentShowCI
      if (typeof saved.maxentModelId === 'string') maxentModelId.value = saved.maxentModelId
    } catch { /* keep defaults */ }
  }

  function buildCells(features: ObservationFeature[], size: number, day: number, window: number, shape = cellShape.value, fields: string[] = []) {
    const cells = new Map<string, HeatmapCell>()
    const wanted = fields.filter((f) => FIELD_MODE_KEYS.has(f))
    for (const f of features) {
      const co = f.geometry?.coordinates
      if (!co) continue
      const lon = Number(co[0]), lat = Number(co[1])
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue

      const geom = cellAt(lon, lat, size, shape)
      let cell = cells.get(geom.key)
      if (!cell) {
        cell = {
          ...geom, n: 0, inWindow: 0, species: new Map(), cover: new Map(),
          ax: 0, ay: 0, aspectN: 0,
          wu: 0, wv: 0, windN: 0,
          expSum: 0, expN: 0,
          fields: new Map(),
        }
        cells.set(geom.key, cell)
      }
      cell.n += 1

      const p = f.properties || {}
      if (hasValue(p.species)) cell.species.set(p.species, (cell.species.get(p.species) || 0) + 1)
      if (hasValue(p.land_cover_label)) {
        cell.cover.set(p.land_cover_label, (cell.cover.get(p.land_cover_label) || 0) + 1)
      }
      const doy = Number(p.day_of_year)
      if (Number.isFinite(doy) && dayDistance(doy, day) <= window) cell.inWindow += 1

      const wu = Number(p.wind_u), wv = Number(p.wind_v)
      if (Number.isFinite(wu) && Number.isFinite(wv)) {
        cell.wu += wu; cell.wv += wv; cell.windN += 1
      }
      const aspect = Number(p.aspect)
      if (Number.isFinite(aspect)) {
        const rad = (aspect * Math.PI) / 180
        cell.ax += Math.sin(rad)
        cell.ay += Math.cos(rad)
        cell.aspectN += 1
      }
      const exp = Number(p.wind_exposure)
      if (Number.isFinite(exp)) { cell.expSum += exp; cell.expN += 1 }

      for (const key of wanted) {
        const v = fieldValue(p, key)
        if (!Number.isFinite(v)) continue
        let acc = cell.fields.get(key)
        if (!acc) { acc = { sum: 0, x: 0, y: 0, n: 0 }; cell.fields.set(key, acc) }
        acc.sum += v
        const rad = (v * Math.PI) / 180
        acc.x += Math.sin(rad); acc.y += Math.cos(rad)
        acc.n += 1
      }
    }
    return [...cells.values()]
  }

  function windField(cells: HeatmapCell[], meta: HeatmapMode) {
    const hasWind = cells.some((c) => c.windN > 0)
    const out: HeatmapCell[] = []
    for (const c of cells) {
      let dx, dy, magnitude
      if (hasWind) {
        if (!c.windN) continue
        const u = c.wu / c.windN, v = c.wv / c.windN
        magnitude = Math.hypot(u, v)
        if (magnitude < 1e-6) continue
        dx = u / magnitude; dy = v / magnitude
      } else {
        if (!c.aspectN) continue
        const mx = c.ax / c.aspectN, my = c.ay / c.aspectN
        magnitude = Math.hypot(mx, my)
        if (magnitude < 1e-6) continue
        dx = mx / magnitude; dy = my / magnitude
      }
      out.push({ ...c, dx, dy, magnitude, exposure: c.expN ? c.expSum / c.expN : null })
    }
    if (!out.length) return { cells: [], legend: null }

    const mags = out.map((c) => c.magnitude)
    const lo = Math.min(...mags), hi = Math.max(...mags)
    const ramp = RAMPS.wind
    for (const c of out) {
      c.t = hi === lo ? 0.5 : (c.magnitude - lo) / (hi - lo)
      const shade = c.exposure ?? c.t
      c.color = rampColor(ramp, shade)
    }
    const fmt = hasWind ? (v: number) => `${v.toFixed(1)} m/s` : (v: number) => `${Math.round(v * 100)}% aligned`
    return {
      cells: out,
      legend: {
        type: 'vector', ramp, source: hasWind ? 'ERA5 10 m wind' : 'Terrain aspect',
        min: fmt(lo), max: fmt(hi), cells: out.length,
        note: meta.windNote,
        colorBy: out.some((c) => c.exposure !== null) ? 'Wind exposure' : 'Magnitude',
      },
    }
  }

  function modalField(cells: HeatmapCell[], meta: HeatmapMode, pick: (c: HeatmapCell) => Map<string, number>, colorKey: string) {
    for (const c of cells) {
      let best = null, bestN = 0
      for (const [v, n] of pick(c)) if (n > bestN) { best = v; bestN = n }
      c.label = best
      c.value = bestN
      c.color = best ? categoryColor(colorKey, best) : '#888'
    }
    const wins = new Map<string, number>()
    for (const c of cells) if (c.label) wins.set(c.label, (wins.get(c.label) || 0) + 1)
    const items = [...wins.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
      .map(([label, n]) => ({ label, color: categoryColor(colorKey, label), n }))
    return { cells, legend: { type: 'categorical', items, total: wins.size, note: meta.note } }
  }

  function fieldMeans(cells: HeatmapCell[], meta: HeatmapMode) {
    const key = meta.field
    const shown: HeatmapCell[] = []
    for (const c of cells) {
      const acc = c.fields.get(key)
      if (!acc || !acc.n) continue
      c.value = meta.circular
        ? ((Math.atan2(acc.x / acc.n, acc.y / acc.n) * 180) / Math.PI + 360) % 360
        : acc.sum / acc.n
      c.samples = acc.n
      shown.push(c)
    }
    if (!shown.length) return { cells: [], legend: null }

    if (meta.circular) {
      for (const c of shown) c.color = bearingColor(c.value)
      return {
        cells: shown,
        legend: {
          type: 'compass', title: meta.label, note: meta.note, cells: shown.length,
          items: ['N', 'E', 'S', 'W'].map((label, i) => ({ label, color: bearingColor(i * 90) })),
        },
      }
    }

    const vals = shown.map((c) => c.value)
    const lo = Math.min(...vals)
    const hi = Math.max(...vals)
    const ramp = RAMPS[meta.key]
    for (const c of shown) {
      c.t = hi === lo ? 0.5 : (c.value - lo) / (hi - lo)
      c.color = rampColor(ramp, c.t)
    }
    const fmt = (v: number) => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2))
    return {
      cells: shown,
      legend: {
        type: 'sequential', ramp, min: fmt(lo), max: fmt(hi),
        title: meta.label, note: meta.note, cells: shown.length,
      },
    }
  }

  function computeHeatmap(features: ObservationFeature[], m = mode.value) {
    const meta = HEATMAP_MODES.find((x) => x.key === m)
    if (!meta || meta.kind === 'none' || !features?.length) return { cells: [], legend: null }

    const size = m === 'wind' ? Math.max(cellSize.value, MIN_VECTOR_CELL) : cellSize.value
    const cells = buildCells(features, size, seasonDay.value, seasonWindow.value,
      cellShape.value, meta.field ? [meta.field] : [])
    if (!cells.length) return { cells: [], legend: null }

    if (m === 'wind') return windField(cells, meta)
    if (meta.kind === 'field') return fieldMeans(cells, meta)
    if (m === 'common') return modalField(cells, meta, (c) => c.species, 'species')
    if (m === 'land_cover') return modalField(cells, meta, (c) => c.cover, 'land_cover_label')

    const MIN_SAMPLE = 3
    for (const c of cells) {
      if (m === 'density') c.raw = Math.log1p(c.n)
      else if (m === 'richness') c.raw = c.species.size
      else if (m === 'season') c.raw = c.n >= MIN_SAMPLE ? c.inWindow / c.n : null
      else if (m === 'hotspots') {
        c.raw = c.n >= MIN_SAMPLE ? (c.inWindow / c.n) * Math.log1p(c.n) : null
      }
    }

    const vals = cells.map((c) => c.raw).filter((v) => Number.isFinite(v as number))
    if (!vals.length) return { cells: [], legend: null }
    const lo = Math.min(...vals as number[])
    const hi = Math.max(...vals as number[])
    const ramp = RAMPS[m] || RAMPS.density
    const shown: HeatmapCell[] = []
    for (const c of cells) {
      if (!Number.isFinite(c.raw as number)) continue
      c.t = hi === lo ? 0.5 : ((c.raw as number) - lo) / (hi - lo)
      c.color = rampColor(ramp, c.t)
      c.value = m === 'density' ? c.n
        : m === 'richness' ? c.species.size
        : c.n ? c.inWindow / c.n : 0
      shown.push(c)
    }

    const fmt = m === 'season' || m === 'hotspots'
      ? (c: HeatmapCell) => `${Math.round((c.n ? c.inWindow / c.n : 0) * 100)}%`
      : (c: HeatmapCell) => String(c.value)
    const loCell = shown.reduce((a, b) => (a.raw as number <= b.raw as number ? a : b))
    const hiCell = shown.reduce((a, b) => (a.raw as number >= b.raw as number ? a : b))

    return {
      cells: shown,
      legend: {
        type: 'sequential', ramp,
        min: fmt(loCell), max: fmt(hiCell),
        title: meta.label, note: meta.note, cells: shown.length,
      },
    }
  }

  function keyAt(lat: number, lon: number, size = cellSize.value) {
    return cellKeyAt(lon, lat, size, cellShape.value)
  }

  const HEATMAP_DOCS: Record<string, string> = {
    density: 'map-heatmap-density',
    richness: 'map-heatmap-richness',
    season: 'map-heatmap-season',
    hotspots: 'map-heatmap-hotspots',
    common: 'map-heatmap-common',
    land_cover: 'map-heatmap-land-cover',
    wind: 'map-heatmap-wind',
  }

  const dayLabel = (day: number) => {
    const d = new Date(Date.UTC(2001, 0, 1))
    d.setUTCDate(((Math.round(day) - 1 + 365) % 365) + 1)
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })
  }

  const seasonLabel = computed(() => dayLabel(seasonDay.value))
  const windowSpan = computed(() =>
    `Counting finds from ${dayLabel(seasonDay.value - seasonWindow.value)} `
    + `to ${dayLabel(seasonDay.value + seasonWindow.value)}`)
  const todayDay = computed(() => todayOfYear())
  const docId = computed(() => (activeMode.value?.kind === 'field'
    ? 'map-heatmap-field'
    : HEATMAP_DOCS[mode.value] || 'map-heatmap'))

  return {
    mode, cellSize, cellShape, seasonDay, seasonWindow, activeMode, groupedModes,
    heatmapOpacity, tileOpacity, todayOfYear, fieldOf,
    seasonLabel, windowSpan, todayDay, docId,
    HEATMAP_MODES, CELL_SIZES, CELL_SHAPES,
    computeHeatmap, buildCells, keyAt, persist, loadFromStorage,
    RAMP_PRESETS, DEFAULT_RAMPS, heatmapRampKey, heatmapRampCustom, rampFor,
    maxentVizMode, maxentThreshold, maxentShowCI, maxentModelId,
  }
}
