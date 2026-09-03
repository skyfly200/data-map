// Aggregate overlays for the map: instead of one mark per observation, bin the
// observations into a lat/lon grid and shade each cell by a summary statistic.
//
// This answers questions the point layer cannot. 48k overlapping dots show where
// the data is dense; a grid shows how many species live in an area, when in the
// year each area fruits, and which species dominates it.
//
// A caveat runs through all of this and is surfaced in the UI: iNaturalist
// records are *observations*, not surveys. Dense cells are partly dense because
// people walk there. The `season` and `hotspots` modes normalise by each cell's
// own total, which cancels most of that bias — a cell's seasonal shape does not
// depend on how many people visited it, only on when they found things.

import { computed, ref } from 'vue'
import { categoryColor, hasValue } from '~/composables/useObservations'

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

/** Today as a day of the year, which is where the seasonal overlays start. */
export function todayOfYear(now = new Date()) {
  const start = new Date(Date.UTC(now.getUTCFullYear(), 0, 0))
  return Math.min(365, Math.max(1, Math.floor((now - start) / 86400000)))
}

export const OVERLAY_MODES = [
  { key: '', label: 'None', kind: 'none' },
  {
    key: 'density', label: 'Observation density', kind: 'sequential',
    note: 'Observations per cell. Reflects where people look as much as where mushrooms are.',
  },
  {
    key: 'richness', label: 'Species richness', kind: 'sequential',
    note: 'Distinct species recorded in each cell.',
  },
  {
    key: 'season', label: 'Seasonal activity', kind: 'sequential',
    note: "Share of the cell's own finds that fall in the selected window — effort-neutral, so it shows when an area fruits.",
  },
  {
    key: 'hotspots', label: 'In-season hotspots', kind: 'sequential',
    note: 'Where finds have actually concentrated in this window, weighted by how well-sampled the cell is. A record of past finds, not a forecast.',
  },
  {
    key: 'common', label: 'Most common species', kind: 'categorical',
    note: 'The most-recorded species in each cell, coloured to match the points.',
  },
  {
    key: 'wind', label: 'Wind / aspect vectors', kind: 'vector',
    note: 'Arrows point the way slopes face; length is how consistent the aspect is, colour is wind exposure.',
    windNote: 'Arrows point downwind (ERA5 10 m mean); length is wind speed.',
  },
]

// Cells smaller than this make the arrow field unreadable — arrows overlap
// before they can be followed — so the vector overlay snaps to at least this.
const MIN_VECTOR_CELL = 0.1

// Light → saturated ramps, one per sequential mode so overlays stay tellable
// apart when a reader switches between them.
export const DEFAULT_RAMPS = {
  density: ['#e8f1fb', '#0b3d91'],
  richness: ['#eef7ec', '#1b5e20'],
  season: ['#fff3e0', '#bf360c'],
  hotspots: ['#f3e9fb', '#4a148c'],
  wind: ['#9ecae1', '#08306b'],
}

// A named set to pick from, plus whatever the viewer sets by hand. Colour on a
// map is not decoration — a ramp that a reader cannot separate at the light end
// hides exactly the low values a density map is meant to show — so the presets
// are all light-to-dark in luminance, and a custom pair is theirs to get wrong.
export const RAMP_PRESETS = [
  { key: 'default', label: 'Per overlay (default)', ramp: null },
  { key: 'blue', label: 'Blue', ramp: ['#e8f1fb', '#0b3d91'] },
  { key: 'green', label: 'Green', ramp: ['#eef7ec', '#1b5e20'] },
  { key: 'warm', label: 'Warm', ramp: ['#fff3e0', '#bf360c'] },
  { key: 'purple', label: 'Purple', ramp: ['#f3e9fb', '#4a148c'] },
  { key: 'viridis', label: 'Viridis-ish', ramp: ['#fde725', '#440154'] },
  { key: 'mono', label: 'Greyscale', ramp: ['#f2f2f2', '#1a1a1a'] },
]

// Overridden per viewer, from the appearance panel. Module-level so the colour
// helpers below track it the same way the point palette does.
export const overlayRampKey = ref('default')
export const overlayRampCustom = ref(null)   // [from, to] hex, when set by hand

/** The ramp actually used for a mode, after any override. */
export function rampFor(mode) {
  const custom = overlayRampCustom.value
  if (overlayRampKey.value === 'custom' && Array.isArray(custom) && custom.length === 2) {
    return custom
  }
  const preset = RAMP_PRESETS.find((p) => p.key === overlayRampKey.value)
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

export function useMapOverlays() {
  const cloud = safeCloudSync()
  const mode = useState('map-overlay-mode', () => '')
  const cellSize = useState('map-overlay-cell', () => 0.05)
  // Day-of-year the seasonal modes centre on, and how wide the window is.
  const seasonDay = useState('map-overlay-day', () => todayOfYear())
  const seasonWindow = useState('map-overlay-window', () => 14)

  const activeMode = computed(() => OVERLAY_MODES.find((m) => m.key === mode.value) || OVERLAY_MODES[0])

  function persist() {
    if (!import.meta.client) return
    try {
      localStorage.setItem('map-overlay', JSON.stringify({
        mode: mode.value, cellSize: cellSize.value,
        seasonDay: seasonDay.value, seasonWindow: seasonWindow.value,
        rampKey: overlayRampKey.value, rampCustom: overlayRampCustom.value,
      }))
      cloud?.schedulePush()
    } catch { /* ignore */ }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem('map-overlay') || 'null')
      if (!saved) return
      // 'fruiting' was the old name for 'hotspots'; carry the saved choice over
      // rather than silently dropping the viewer back to no overlay.
      const savedMode = saved.mode === 'fruiting' ? 'hotspots' : saved.mode
      if (OVERLAY_MODES.some((m) => m.key === savedMode)) mode.value = savedMode
      if (CELL_SIZES.some((c) => c.value === saved.cellSize)) cellSize.value = saved.cellSize
      if (Number.isFinite(saved.seasonDay)) seasonDay.value = saved.seasonDay
      if (Number.isFinite(saved.seasonWindow)) seasonWindow.value = saved.seasonWindow
      if (saved.rampKey === 'custom' || RAMP_PRESETS.some((r) => r.key === saved.rampKey)) {
        overlayRampKey.value = saved.rampKey
      }
      if (Array.isArray(saved.rampCustom) && saved.rampCustom.length === 2
        && saved.rampCustom.every((c) => /^#[0-9a-f]{6}$/i.test(c))) {
        overlayRampCustom.value = saved.rampCustom
      }
    } catch { /* keep defaults */ }
  }

  /**
   * Bin features into grid cells in a single pass.
   *
   * Returns [{ key, lat0, lon0, lat1, lon1, n, inWindow, species: Map }].
   * Callers turn those counts into whatever the active mode needs.
   */
  function buildCells(features, size, day, window) {
    const cells = new Map()
    for (const f of features) {
      const co = f.geometry?.coordinates
      if (!co) continue
      const lon = Number(co[0]), lat = Number(co[1])
      if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue

      const gy = Math.floor(lat / size)
      const gx = Math.floor(lon / size)
      const key = `${gy}:${gx}`
      let cell = cells.get(key)
      if (!cell) {
        cell = {
          key, n: 0, inWindow: 0, species: new Map(),
          ax: 0, ay: 0, aspectN: 0,      // aspect as summed unit vectors
          wu: 0, wv: 0, windN: 0,        // ERA5 wind components, when present
          expSum: 0, expN: 0,            // wind exposure index
          lat0: gy * size, lon0: gx * size,
          lat1: (gy + 1) * size, lon1: (gx + 1) * size,
        }
        cells.set(key, cell)
      }
      cell.n += 1

      const p = f.properties || {}
      if (hasValue(p.species)) cell.species.set(p.species, (cell.species.get(p.species) || 0) + 1)
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
      out.push({
        ...c, dx, dy, magnitude,
        exposure: c.expN ? c.expSum / c.expN : null,
        lat: (c.lat0 + c.lat1) / 2, lon: (c.lon0 + c.lon1) / 2,
      })
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

  /**
   * Cells with a colour and a display value for the active mode.
   * `features` is the already-filtered FeatureCollection's features.
   */
  function computeOverlay(features, m = mode.value) {
    const meta = OVERLAY_MODES.find((x) => x.key === m)
    if (!meta || meta.kind === 'none' || !features?.length) return { cells: [], legend: null }

    const size = m === 'wind' ? Math.max(cellSize.value, MIN_VECTOR_CELL) : cellSize.value
    const cells = buildCells(features, size, seasonDay.value, seasonWindow.value)
    if (!cells.length) return { cells: [], legend: null }

    if (m === 'wind') return windField(cells, meta)

    if (m === 'common') {
      for (const c of cells) {
        let best = null, bestN = 0
        for (const [sp, n] of c.species) if (n > bestN) { best = sp; bestN = n }
        c.label = best
        c.value = bestN
        c.color = best ? categoryColor('species', best) : '#888'
      }
      // Legend: the species that actually dominate the most cells.
      const wins = new Map()
      for (const c of cells) if (c.label) wins.set(c.label, (wins.get(c.label) || 0) + 1)
      const legend = [...wins.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10)
        .map(([label, n]) => ({ label, color: categoryColor('species', label), n }))
      return { cells, legend: { type: 'categorical', items: legend, total: wins.size } }
    }

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

  return {
    mode, cellSize, seasonDay, seasonWindow, activeMode, todayOfYear,
    OVERLAY_MODES, CELL_SIZES,
    computeOverlay, buildCells, persist, loadFromStorage,
    RAMP_PRESETS, DEFAULT_RAMPS, overlayRampKey, overlayRampCustom, rampFor,
  }
}
