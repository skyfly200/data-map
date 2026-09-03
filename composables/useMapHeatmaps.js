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
import { categoryColor, hasValue } from '~/composables/useObservations'
import { cellAt, cellKeyAt, CELL_SHAPES } from '~/composables/gridCells'
import { ALL_NUMERIC } from '~/composables/useChartFields'

// CELL_SHAPES is deliberately not re-exported: Nuxt auto-imports every
// composables/ export by name, and a second export of the same symbol makes
// which module wins depend on scan order. It is handed out on the returned
// object instead, and gridCells is the one place it is declared.

// Grid resolutions in degrees of latitude, with a rough ground distance.
export const CELL_SIZES = [
  // Below about a kilometre the grid stops summarising and starts drawing one
  // cell per observation, which is the point layer with square markers — so
  // these are offered, but the legend reports the cell count so you can see
  // when that has happened.
  { value: 0.005, label: '~500 m' },
  { value: 0.01, label: '~1 km' },
  { value: 0.02, label: '~2 km' },
  { value: 0.05, label: '~5 km' },
  { value: 0.1, label: '~11 km' },
  { value: 0.25, label: '~28 km' },
]

/** Today as a day of the year, which is where the seasonal heatmaps start. */
export function todayOfYear(now = new Date()) {
  const start = new Date(Date.UTC(now.getUTCFullYear(), 0, 0))
  return Math.min(365, Math.max(1, Math.floor((now - start) / 86400000)))
}

// The enriched per-observation fields worth averaging across a cell. Each one
// becomes a heatmap of its own: the map already carries the value at every
// point, and a grid of cell means is what turns 48k scattered readings into a
// surface you can actually read.
//
// These are NOT global rasters — a cell has a value only where somebody has
// recorded a find. That is the honest version: an empty cell means nobody has
// looked there, and the note on each one says so.
export const FIELD_MODES = [
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
    note: 'Mean compass direction the ground faces, averaged as vectors so north does not average to south. Colour is the direction itself, not a magnitude.' },
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

const FIELD_LABELS = Object.fromEntries(ALL_NUMERIC.map((f) => [f.key, f.label]))
// Fields the running dataset does not carry are dropped from the picker rather
// than offered and then drawing nothing.
const FIELD_MODE_KEYS = new Set(FIELD_MODES.map((f) => f.key))

export const HEATMAP_MODES = [
  { key: '', label: 'None', kind: 'none' },
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
    note: 'The most-recorded species in each cell, coloured to match the points.',
  },
  {
    key: 'land_cover', label: 'Land cover', kind: 'categorical', group: 'Observations',
    note: 'The most common land-cover class recorded across the finds in each cell.',
  },
  {
    key: 'wind', label: 'Wind / aspect vectors', kind: 'vector', group: 'Terrain',
    note: 'Arrows point the way slopes face; length is how consistent the aspect is, colour is wind exposure.',
    windNote: 'Arrows point downwind (ERA5 10 m mean); length is wind speed.',
  },
  // Cell means of the enriched fields, generated so a new enrichment column
  // becomes a readable layer by being named once above.
  ...FIELD_MODES.map((f) => ({
    key: `f:${f.key}`,
    label: FIELD_LABELS[f.key] || f.key,
    kind: 'field', group: f.group,
    field: f.key, circular: !!f.circular, fieldRamp: f.ramp, note: f.note,
  })),
]

/** Numeric field behind a mode key, or null for the built-in modes. */
export function fieldOf(mode) {
  return typeof mode === 'string' && mode.startsWith('f:') ? mode.slice(2) : null
}

// Cells smaller than this make the arrow field unreadable — arrows overlap
// before they can be followed — so the vector heatmap snaps to at least this.
const MIN_VECTOR_CELL = 0.1

// Light → saturated ramps, one per sequential mode so heatmaps stay tellable
// apart when a reader switches between them.
export const DEFAULT_RAMPS = {
  density: ['#e8f1fb', '#0b3d91'],
  richness: ['#eef7ec', '#1b5e20'],
  season: ['#fff3e0', '#bf360c'],
  hotspots: ['#f3e9fb', '#4a148c'],
  wind: ['#9ecae1', '#08306b'],
  ...Object.fromEntries(FIELD_MODES.filter((f) => f.ramp).map((f) => [`f:${f.key}`, f.ramp])),
}

// A named set to pick from, plus whatever the viewer sets by hand. Colour on a
// map is not decoration — a ramp that a reader cannot separate at the light end
// hides exactly the low values a density map is meant to show — so the presets
// are all light-to-dark in luminance, and a custom pair is theirs to get wrong.
export const RAMP_PRESETS = [
  { key: 'default', label: 'Per heatmap (default)', ramp: null },
  { key: 'blue', label: 'Blue', ramp: ['#e8f1fb', '#0b3d91'] },
  { key: 'green', label: 'Green', ramp: ['#eef7ec', '#1b5e20'] },
  { key: 'warm', label: 'Warm', ramp: ['#fff3e0', '#bf360c'] },
  { key: 'purple', label: 'Purple', ramp: ['#f3e9fb', '#4a148c'] },
  { key: 'viridis', label: 'Viridis-ish', ramp: ['#fde725', '#440154'] },
  { key: 'mono', label: 'Greyscale', ramp: ['#f2f2f2', '#1a1a1a'] },
]

// Overridden per viewer, from the style panel. Module-level so the colour
// helpers below track it the same way the point palette does.
export const heatmapRampKey = ref('default')
export const heatmapRampCustom = ref(null)   // [from, to] hex, when set by hand

/** The ramp actually used for a mode, after any override. */
export function rampFor(mode) {
  const custom = heatmapRampCustom.value
  if (heatmapRampKey.value === 'custom' && Array.isArray(custom) && custom.length === 2) {
    return custom
  }
  const preset = RAMP_PRESETS.find((p) => p.key === heatmapRampKey.value)
  if (preset?.ramp) return preset.ramp
  return DEFAULT_RAMPS[mode] || DEFAULT_RAMPS.density
}

// Read through the override wherever the ramps were read directly before.
const RAMPS = new Proxy({}, {
  get: (_t, mode) => rampFor(String(mode)),
  has: (_t, mode) => String(mode) in DEFAULT_RAMPS,
})

export function hexLerp(a, b, t) {
  const k = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0))
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  const mix = pa.map((v, i) => Math.round(v + (pb[i] - v) * k))
  return `#${mix.map((v) => v.toString(16).padStart(2, '0')).join('')}`
}

// Circular distance between two days of the year, so a window around 1 Jan
// reaches back into December rather than falling off the end.
export function dayDistance(a, b) {
  const d = Math.abs(a - b) % 365
  return Math.min(d, 365 - d)
}

/**
 * Colour for a compass bearing.
 *
 * Direction is circular, so it needs a circular ramp: a light-to-dark one would
 * paint 359° and 1° at opposite ends of the scale. This walks the hue wheel,
 * with north fixed at the top so the legend can be read as a compass.
 */
export function bearingColor(deg) {
  const h = ((deg % 360) + 360) % 360
  return `hsl(${h.toFixed(0)}, 68%, 48%)`
}

export function useMapHeatmaps() {
  const cloud = safeCloudSync()
  const mode = useState('map-overlay-mode', () => '')
  const cellSize = useState('map-overlay-cell', () => 0.05)
  const cellShape = useState('map-overlay-shape', () => 'hex')
  // Day-of-year the seasonal modes centre on, and how wide the window is.
  const seasonDay = useState('map-overlay-day', () => todayOfYear())
  const seasonWindow = useState('map-overlay-window', () => 14)
  // How strongly the cells sit over the basemap. Low enough to read the ground
  // through, high enough that the ramp's light end is still distinguishable.
  const heatmapOpacity = useState('map-heatmap-opacity', () => 0.55)
  // The reference tile layers, dimmed together. Separate from the above: they
  // are different things stacked at different depths, and wanting a faint
  // hillshade under a solid heatmap is the normal case, not an odd one.
  const tileOpacity = useState('map-tile-opacity', () => 1)

  const activeMode = computed(() => HEATMAP_MODES.find((m) => m.key === mode.value) || HEATMAP_MODES[0])

  /** Modes grouped for an <optgroup>-based picker, in declaration order. */
  const groupedModes = computed(() => {
    const groups = []
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
      }))
      cloud?.schedulePush()
    } catch { /* ignore */ }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem('map-overlay') || 'null')
      if (!saved) return
      // Retired mode keys, carried over rather than silently dropping the
      // viewer back to no heatmap: 'fruiting' became 'hotspots', and
      // 'dominant' became 'common' when it was relabelled "Most common
      // species". A key is not just a label — it is in every shared link and
      // every viewer's saved settings.
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
        heatmapRampCustom.value = saved.rampCustom
      }
    } catch { /* keep defaults */ }
  }

  /**
   * Bin features into grid cells in a single pass.
   *
   * Returns [{ key, lat, lon, polygon, lat0..lon1, n, inWindow, species: Map }].
   * Callers turn those counts into whatever the active mode needs.
   */
  function buildCells(features, size, day, window, shape = cellShape.value, fields = []) {
    const cells = new Map()
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
          ax: 0, ay: 0, aspectN: 0,      // aspect as summed unit vectors
          wu: 0, wv: 0, windN: 0,        // ERA5 wind components, when present
          expSum: 0, expN: 0,            // wind exposure index
          fields: new Map(),             // key → { sum, n } or { x, y, n } if circular
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

      // Directions are circular, so they are averaged as unit vectors — the
      // arithmetic mean of 350° and 10° is 180°, which points exactly backwards.
      // ERA5 wind components are already vectors and sum directly.
      const wu = Number(p.wind_u), wv = Number(p.wind_v)
      if (Number.isFinite(wu) && Number.isFinite(wv)) {
        cell.wu += wu; cell.wv += wv; cell.windN += 1
      }
      const aspect = Number(p.aspect)
      if (Number.isFinite(aspect)) {
        const rad = (aspect * Math.PI) / 180
        cell.ax += Math.sin(rad)   // compass: 0° = north, increasing clockwise
        cell.ay += Math.cos(rad)
        cell.aspectN += 1
      }
      const exp = Number(p.wind_exposure)
      if (Number.isFinite(exp)) { cell.expSum += exp; cell.expN += 1 }

      // Only the field the active heatmap needs is accumulated; summing all
      // eleven on every one of 48k features would cost eleven times as much for
      // ten results nobody asked for.
      for (const key of wanted) {
        const v = Number(p[key])
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

  /**
   * Arrow per cell: real ERA5 wind where the pipeline has sampled it, otherwise
   * mean terrain aspect.
   *
   * The two mean different things and the legend says which is in use. Wind
   * arrows point downwind at a length set by speed. Aspect arrows point the way
   * the ground faces, at a length set by how consistently it faces that way —
   * a short arrow is a cell of mixed terrain, not a calm one.
   */
  function windField(cells, meta) {
    const hasWind = cells.some((c) => c.windN > 0)
    const out = []
    for (const c of cells) {
      let dx, dy, magnitude
      if (hasWind) {
        if (!c.windN) continue
        const u = c.wu / c.windN, v = c.wv / c.windN
        magnitude = Math.hypot(u, v)          // mean wind speed, m/s
        if (magnitude < 1e-6) continue
        dx = u / magnitude; dy = v / magnitude
      } else {
        if (!c.aspectN) continue
        const mx = c.ax / c.aspectN, my = c.ay / c.aspectN
        // Resultant length of the mean unit vector: 1 = every slope faces the
        // same way, 0 = aspects cancel out.
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
      // Colour by exposure when it is populated (that is the forager-relevant
      // signal); fall back to the vector's own magnitude.
      const shade = c.exposure ?? c.t
      c.color = hexLerp(ramp[0], ramp[1], shade)
    }
    const fmt = hasWind ? (v) => `${v.toFixed(1)} m/s` : (v) => `${Math.round(v * 100)}% aligned`
    return {
      cells: out,
      legend: {
        type: 'vector', ramp, source: hasWind ? 'ERA5 10 m wind' : 'Terrain aspect',
        min: fmt(lo), max: fmt(hi), cells: out.length,
        note: hasWind ? meta.windNote : meta.note,
        colorBy: out.some((c) => c.exposure !== null) ? 'Wind exposure' : 'Magnitude',
      },
    }
  }

  /** Most common value of a per-cell tally, as a categorical heatmap. */
  function modalField(cells, meta, pick, colorKey) {
    for (const c of cells) {
      let best = null, bestN = 0
      for (const [v, n] of pick(c)) if (n > bestN) { best = v; bestN = n }
      c.label = best
      c.value = bestN
      c.color = best ? categoryColor(colorKey, best) : '#888'
    }
    const wins = new Map()
    for (const c of cells) if (c.label) wins.set(c.label, (wins.get(c.label) || 0) + 1)
    const items = [...wins.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
      .map(([label, n]) => ({ label, color: categoryColor(colorKey, label), n }))
    return { cells, legend: { type: 'categorical', items, total: wins.size, note: meta.note } }
  }

  /**
   * Cell means of one enriched field.
   *
   * Aspect is circular and gets a vector mean plus a hue-wheel key; everything
   * else is an arithmetic mean on the mode's own ramp. Cells with no reading for
   * the field are dropped rather than drawn as zero — no data is not a low value.
   */
  function fieldMeans(cells, meta) {
    const key = meta.field
    const shown = []
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
    const lo = Math.min(...vals), hi = Math.max(...vals)
    const ramp = RAMPS[meta.key]
    for (const c of shown) {
      c.t = hi === lo ? 0.5 : (c.value - lo) / (hi - lo)
      c.color = hexLerp(ramp[0], ramp[1], c.t)
    }
    const fmt = (v) => (Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2))
    return {
      cells: shown,
      legend: {
        type: 'sequential', ramp, min: fmt(lo), max: fmt(hi),
        title: meta.label, note: meta.note, cells: shown.length,
      },
    }
  }

  /**
   * Cells with a colour and a display value for the active mode.
   * `features` is the already-filtered FeatureCollection's features.
   */
  function computeHeatmap(features, m = mode.value) {
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

    // Sequential modes: compute a raw value per cell, then scale to the range
    // actually present so the ramp uses its full contrast.
    const MIN_SAMPLE = 3   // a cell with one find has no meaningful seasonal shape
    for (const c of cells) {
      if (m === 'density') c.raw = Math.log1p(c.n)
      else if (m === 'richness') c.raw = c.species.size
      else if (m === 'season') c.raw = c.n >= MIN_SAMPLE ? c.inWindow / c.n : null
      else if (m === 'hotspots') {
        // Seasonal share carries the "when"; log density carries confidence that
        // the cell is worth trusting at all. Cells nobody has sampled stay dark
        // rather than scoring high off one lucky find.
        c.raw = c.n >= MIN_SAMPLE ? (c.inWindow / c.n) * Math.log1p(c.n) : null
      }
    }

    const vals = cells.map((c) => c.raw).filter((v) => Number.isFinite(v))
    if (!vals.length) return { cells: [], legend: null }
    const lo = Math.min(...vals), hi = Math.max(...vals)
    const ramp = RAMPS[m] || RAMPS.density
    const shown = []
    for (const c of cells) {
      if (!Number.isFinite(c.raw)) continue
      c.t = hi === lo ? 0.5 : (c.raw - lo) / (hi - lo)
      c.color = hexLerp(ramp[0], ramp[1], c.t)
      c.value = m === 'density' ? c.n
        : m === 'richness' ? c.species.size
          : c.n ? c.inWindow / c.n : 0
      shown.push(c)
    }

    // Legend endpoints read in the metric's own units, not the scaled 0–1.
    const fmt = m === 'season' || m === 'hotspots'
      ? (c) => `${Math.round((c.n ? c.inWindow / c.n : 0) * 100)}%`
      : (c) => String(c.value)
    const loCell = shown.reduce((a, b) => (a.raw <= b.raw ? a : b))
    const hiCell = shown.reduce((a, b) => (a.raw >= b.raw ? a : b))
    return {
      cells: shown,
      legend: {
        type: 'sequential', ramp,
        min: fmt(loCell), max: fmt(hiCell),
        title: meta.label, note: meta.note, cells: shown.length,
      },
    }
  }

  /** Bin key for a coordinate under the current grid, for hover lookups. */
  function keyAt(lat, lon, size = cellSize.value) {
    return cellKeyAt(lon, lat, size, cellShape.value)
  }

  return {
    mode, cellSize, cellShape, seasonDay, seasonWindow, activeMode, groupedModes,
    heatmapOpacity, tileOpacity, todayOfYear, fieldOf,
    HEATMAP_MODES, CELL_SIZES, CELL_SHAPES,
    computeHeatmap, buildCells, keyAt, persist, loadFromStorage,
    RAMP_PRESETS, DEFAULT_RAMPS, heatmapRampKey, heatmapRampCustom, rampFor,
  }
}
