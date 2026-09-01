// Per-viewer control over how marks look: which palette categories draw from,
// which shapes they rotate through, per-value colour and shape overrides, and
// point size/opacity.
//
// The colour functions live here rather than in useObservations because they
// have to READ this state, and they are called from inside computeds all over
// the app. Keeping the state in module-level refs means every one of those
// computeds tracks it automatically — change the palette and the map, the
// legend and every chart re-colour themselves without a single explicit watch.
//
// Nothing here is SSR state: it is a display preference, loaded from
// localStorage on the client. The charts have no data server-side, so no mark
// colours are serialised and there is nothing to mismatch on hydration.

import { computed, ref } from 'vue'

export const PALETTES = [
  {
    key: 'default', label: 'Default',
    colors: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100',
             '#e87ba4', '#008300', '#4a3aa7', '#e34948'],
  },
  {
    // Okabe–Ito: designed to stay distinguishable with the common forms of
    // colour-vision deficiency. Worth having, given the app leans on colour to
    // carry species and cluster identity.
    key: 'okabe', label: 'Colour-blind safe',
    colors: ['#0072b2', '#e69f00', '#009e73', '#cc79a7',
             '#56b4e9', '#d55e00', '#f0e442', '#000000'],
  },
  {
    key: 'vivid', label: 'Vivid',
    colors: ['#e6194b', '#3cb44b', '#4363d8', '#f58231',
             '#911eb4', '#42d4f4', '#f032e6', '#bfef45'],
  },
  {
    key: 'earth', label: 'Earth',
    colors: ['#8c6d31', '#637939', '#8c564b', '#bd9e39',
             '#7b4173', '#31696d', '#a55194', '#556b2f'],
  },
  {
    key: 'pastel', label: 'Pastel',
    colors: ['#8ecae6', '#ffb703', '#90be6d', '#f4978e',
             '#cdb4db', '#b5e2fa', '#ffc8dd', '#a2d2ff'],
  },
]

export const ALL_SHAPES = ['circle', 'square', 'triangle', 'diamond', 'cross', 'wye']

export const SHAPE_SETS = [
  { key: 'all', label: 'All shapes', shapes: ALL_SHAPES },
  { key: 'geometric', label: 'Geometric', shapes: ['circle', 'square', 'triangle', 'diamond'] },
  { key: 'round', label: 'Circles only', shapes: ['circle'] },
]

export const UNCLUSTERED = '#9aa0a6'

const DEFAULTS = {
  palette: 'default',
  shapeSet: 'all',
  pointRadius: 4,
  pointOpacity: 0.85,
  colorOverrides: {},   // "field:value" → hex
  shapeOverrides: {},   // "field:value" → shape name
}

const STORAGE_KEY = 'appearance'

// Module-level refs: read by the colour helpers below, so every computed that
// calls one tracks them.
const paletteKey = ref(DEFAULTS.palette)
const shapeSetKey = ref(DEFAULTS.shapeSet)
const pointRadius = ref(DEFAULTS.pointRadius)
const pointOpacity = ref(DEFAULTS.pointOpacity)
const colorOverrides = ref({ ...DEFAULTS.colorOverrides })
const shapeOverrides = ref({ ...DEFAULTS.shapeOverrides })

const activeColors = computed(() =>
  (PALETTES.find((p) => p.key === paletteKey.value) || PALETTES[0]).colors)
const activeShapes = computed(() =>
  (SHAPE_SETS.find((s) => s.key === shapeSetKey.value) || SHAPE_SETS[0]).shapes)

/** Backwards-compatible name: the palette currently in effect. */
export const PALETTE = new Proxy([], {
  get: (_t, prop) => Reflect.get(activeColors.value, prop),
  has: (_t, prop) => Reflect.has(activeColors.value, prop),
  ownKeys: () => Reflect.ownKeys(activeColors.value),
  getOwnPropertyDescriptor: (_t, prop) =>
    Reflect.getOwnPropertyDescriptor(activeColors.value, prop),
})

export const SERIES_1 = '#2a78d6'

export const overrideKey = (field, value) => `${field}:${value}`

/** Index-based colour, used by pipeline clusters (0, 1, 2 …). */
export function colorFor(cluster) {
  if (cluster === null || cluster === undefined || Number.isNaN(cluster)) return UNCLUSTERED
  const colors = activeColors.value
  return colors[cluster % colors.length]
}

// Deterministic colour for a category value, so the same value (a species, a
// year, a land-cover class) gets the SAME colour on the map and in every chart.
export function stableColor(value) {
  if (value === null || value === undefined || value === '') return UNCLUSTERED
  const s = String(value)
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
  const colors = activeColors.value
  return colors[h % colors.length]
}

// Colour for a (field, value) pair. An explicit override wins; clusters keep
// their index-based palette; everything else uses the stable hash.
export function categoryColor(field, value) {
  if (value === null || value === undefined || value === '') return UNCLUSTERED
  const override = colorOverrides.value[overrideKey(field, value)]
  if (override) return override
  if (field === 'cluster' || field === 'live_cluster') {
    const n = Number(String(value).replace(/^[CK]/, ''))
    return Number.isFinite(n) ? colorFor(n) : UNCLUSTERED
  }
  return stableColor(value)
}

/**
 * Shape for a (field, value) pair, given the value's position in the category
 * list. An override wins; otherwise values rotate through the active shape set.
 */
export function categoryShape(field, value, index = 0) {
  const override = shapeOverrides.value[overrideKey(field, value)]
  if (override) return override
  const shapes = activeShapes.value
  return shapes[index % shapes.length]
}

export function useAppearance() {
  // Captured during setup: persist() runs from DOM event handlers, where
  // useNuxtApp() (and therefore useCloudSync) is not reliably available.
  const cloud = safeCloudSync()

  function persist() {
    if (!import.meta.client) return
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        palette: paletteKey.value,
        shapeSet: shapeSetKey.value,
        pointRadius: pointRadius.value,
        pointOpacity: pointOpacity.value,
        colorOverrides: colorOverrides.value,
        shapeOverrides: shapeOverrides.value,
      }))
      // Mirror to the account when one is connected. Debounced there, so
      // dragging a slider is one request, not one per pixel.
      cloud?.schedulePush()
    } catch { /* ignore */ }
  }

  function loadFromStorage() {
    if (!import.meta.client) return
    try {
      const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
      if (!saved) return
      if (PALETTES.some((p) => p.key === saved.palette)) paletteKey.value = saved.palette
      if (SHAPE_SETS.some((s) => s.key === saved.shapeSet)) shapeSetKey.value = saved.shapeSet
      if (Number.isFinite(saved.pointRadius)) pointRadius.value = saved.pointRadius
      if (Number.isFinite(saved.pointOpacity)) pointOpacity.value = saved.pointOpacity
      if (saved.colorOverrides && typeof saved.colorOverrides === 'object') {
        colorOverrides.value = { ...saved.colorOverrides }
      }
      if (saved.shapeOverrides && typeof saved.shapeOverrides === 'object') {
        shapeOverrides.value = { ...saved.shapeOverrides }
      }
    } catch { /* keep defaults */ }
  }

  function setColor(field, value, hex) {
    // Replace the object rather than mutating it, so the ref actually fires.
    colorOverrides.value = { ...colorOverrides.value, [overrideKey(field, value)]: hex }
    persist()
  }
  function clearColor(field, value) {
    const next = { ...colorOverrides.value }
    delete next[overrideKey(field, value)]
    colorOverrides.value = next
    persist()
  }
  function setShape(field, value, shape) {
    shapeOverrides.value = { ...shapeOverrides.value, [overrideKey(field, value)]: shape }
    persist()
  }
  function clearShape(field, value) {
    const next = { ...shapeOverrides.value }
    delete next[overrideKey(field, value)]
    shapeOverrides.value = next
    persist()
  }
  function hasOverride(field, value) {
    const k = overrideKey(field, value)
    return Boolean(colorOverrides.value[k] || shapeOverrides.value[k])
  }

  function reset() {
    paletteKey.value = DEFAULTS.palette
    shapeSetKey.value = DEFAULTS.shapeSet
    pointRadius.value = DEFAULTS.pointRadius
    pointOpacity.value = DEFAULTS.pointOpacity
    colorOverrides.value = {}
    shapeOverrides.value = {}
    persist()
  }

  const overrideCount = computed(() =>
    Object.keys(colorOverrides.value).length + Object.keys(shapeOverrides.value).length)

  return {
    PALETTES, SHAPE_SETS, ALL_SHAPES,
    paletteKey, shapeSetKey, pointRadius, pointOpacity,
    activeColors, activeShapes, colorOverrides, shapeOverrides, overrideCount,
    persist, loadFromStorage, reset,
    setColor, clearColor, setShape, clearShape, hasOverride,
  }
}
