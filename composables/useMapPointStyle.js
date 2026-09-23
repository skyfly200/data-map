import { computed, ref, watch } from 'vue'
import { hasValue } from '~/composables/useObservations'
import { PALETTE, UNCLUSTERED, categoryColor, colorFor } from '~/composables/useAppearance'
import { classColorFor, fraction, matchNote, paletteFor } from '~/composables/fieldPalettes'
import { normaliseStops, rampColor } from '~/composables/ramps'
import { RAMP_PRESETS } from '~/composables/useMapHeatmaps'
import { ALL_CATEGORY, ALL_NUMERIC } from '~/composables/useChartFields'
import { coverageNote } from '~/composables/fieldCoverage'
import { useAppearance } from '~/composables/useAppearance'
import { useUnits } from '~/composables/useUnits'

const COLORBY_KEY = 'map-color-by'
const SIZEBY_KEY = 'map-size-by'
const RAMP = ['#e8f1fb', '#0b3d91']
const LEGEND_CAP = 12

export const CATEGORY_KEYS = new Set(ALL_CATEGORY.map((f) => f.key))
export const FIELD_LABEL = Object.fromEntries([...ALL_CATEGORY, ...ALL_NUMERIC].map((f) => [f.key, f.label]))

export function fmtNum(v) {
  return Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v).toFixed(2)
}

export function useMapPointStyle({ filteredData, live }) {
  const appearance = useAppearance()
  const { elevLabel, elevValue, tempValue, unit, tempUnit } = useUnits()
  const { pointRadius, pointOpacity, pointOutline, colorSeed, activeColors, colorOverrides, stackBlend } = appearance

  const chosenPointRamp = computed(() => {
    const key = appearance.pointRampKey.value
    if (key === 'auto') return null
    if (key === 'custom') return normaliseStops(appearance.pointRampCustom.value)
    return RAMP_PRESETS.find((p) => p.key === key)?.ramp || null
  })

  const colorBy = ref('cluster')
  const sizeBy = ref('')

  if (import.meta.client) {
    const savedColor = localStorage.getItem(COLORBY_KEY)
    if (savedColor) colorBy.value = savedColor
    const savedSize = localStorage.getItem(SIZEBY_KEY)
    if (savedSize !== null) sizeBy.value = savedSize
  }

  watch(colorBy, (v) => { if (import.meta.client) localStorage.setItem(COLORBY_KEY, v) })
  watch(sizeBy, (v) => { if (import.meta.client) localStorage.setItem(SIZEBY_KEY, v) })

  const hoverValue = ref(null)

  function hoverEnter(label, e) { if (e.pointerType === 'mouse') hoverValue.value = label }
  function hoverLeave(e) { if (!e || e.pointerType === 'mouse') hoverValue.value = null }
  function pickValue(label, e) {
    if (e && e.pointerType === 'mouse') return
    hoverValue.value = hoverValue.value === label ? null : label
  }

  const colorOptions = computed(() => {
    const feats = filteredData.value?.features || []
    const present = (list) => list.filter((f) => (
      f.key === 'live_cluster' ? live.active.value : feats.some((ft) => hasValue(ft.properties[f.key]))
    ))
    return { category: present(ALL_CATEGORY), numeric: present(ALL_NUMERIC) }
  })

  const colorCoverageNote = computed(() => coverageNote(filteredData.value?.features || [], colorBy.value))

  watch(colorOptions, (opts) => {
    const keys = [...opts.category, ...opts.numeric].map((o) => o.key)
    if (keys.length && !keys.includes(colorBy.value)) colorBy.value = keys[0]
    if (sizeBy.value && !opts.numeric.some((o) => o.key === sizeBy.value)) sizeBy.value = ''
  })

  const coloring = computed(() => {
    const feats = filteredData.value?.features || []
    const key = colorBy.value
    const title = FIELD_LABEL[key] || key

    if (key === 'live_cluster') {
      const seen = new Set()
      let hasNull = false
      for (const f of feats) {
        const lab = live.labelFor(f.properties)
        if (hasValue(lab)) seen.add(lab); else hasNull = true
      }
      const legend = [...seen].sort().map((lab) => ({ label: lab, color: categoryColor('live_cluster', lab) }))
      if (hasNull) legend.push({ label: 'Unclustered', color: UNCLUSTERED })
      return {
        type: 'categorical', title, legend,
        colorFn: (p) => categoryColor('live_cluster', live.labelFor(p)),
        labelOf: (p) => (hasValue(live.labelFor(p)) ? live.labelFor(p) : 'Unclustered'),
      }
    }

    if (key === 'cluster') {
      const seen = new Set()
      let hasNull = false
      for (const f of feats) {
        const c = f.properties.cluster
        if (hasValue(c)) seen.add(c); else hasNull = true
      }
      const legend = [...seen].sort((a, b) => a - b).map((c) => ({ label: `Cluster ${c}`, color: colorFor(c) }))
      if (hasNull) legend.push({ label: 'Unclustered', color: UNCLUSTERED })
      return {
        type: 'categorical', title, legend,
        colorFn: (p) => colorFor(p.cluster),
        labelOf: (p) => (hasValue(p.cluster) ? `Cluster ${p.cluster}` : 'Unclustered'),
      }
    }

    if (CATEGORY_KEYS.has(key)) {
      const counts = new Map()
      for (const f of feats) {
        const v = f.properties[key]
        if (hasValue(v)) counts.set(v, (counts.get(v) || 0) + 1)
      }
      const cats = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v)
      const palette = paletteFor(key)
      const classes = palette?.kind === 'classes' ? palette.items : null
      const colorOf = (v) => (classes && classColorFor(classes, v)) || categoryColor(key, v)
      const legend = cats.slice(0, LEGEND_CAP).map((v) => ({ label: String(v), color: colorOf(v) }))
      if (cats.length > LEGEND_CAP) legend.push({ label: `+${cats.length - LEGEND_CAP} more`, color: UNCLUSTERED })
      return {
        type: 'categorical', title, legend,
        match: palette ? matchNote(palette) : '',
        colorFn: (p) => (hasValue(p[key]) ? colorOf(p[key]) : UNCLUSTERED),
        labelOf: (p) => (hasValue(p[key]) ? String(p[key]) : null),
      }
    }

    const meta = ALL_NUMERIC.find((f) => f.key === key) || {}
    const conv = meta.unit === 'elev' ? elevValue : meta.unit === 'temp' ? tempValue : (v) => Number(v)
    const unitSuffix = meta.unit === 'elev' ? ` (${unit.value})` : meta.unit === 'temp' ? ` (°${tempUnit.value})` : ''
    const vals = feats.map((f) => f.properties[key]).filter(hasValue).map((v) => conv(Number(v)))
    const dataMin = vals.length ? Math.min(...vals) : 0
    const dataMax = vals.length ? Math.max(...vals) : 1
    const palette = paletteFor(key)
    const chosen = chosenPointRamp.value
    const stops = chosen || (palette?.kind === 'ramp' ? palette.stops : RAMP)
    const [min, max] = (!chosen && palette?.domain) || [dataMin, dataMax]
    return {
      type: 'sequential', title: title + unitSuffix, min, max, stops,
      match: !chosen && palette ? matchNote(palette) : '',
      colorFn: (p) => {
        const raw = p[key]
        if (!hasValue(raw)) return UNCLUSTERED
        return rampColor(stops, fraction(conv(Number(raw)), [min, max]))
      },
    }
  })

  const legendValues = computed(() => {
    if (coloring.value?.type !== 'categorical') return []
    const key = colorBy.value
    const counts = new Map()
    for (const f of filteredData.value?.features || []) {
      const v = key === 'live_cluster' ? live.labelFor(f.properties) : f.properties[key]
      if (hasValue(v)) counts.set(v, (counts.get(v) || 0) + 1)
    }
    return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v)
  })

  const sizeScale = computed(() => {
    if (!sizeBy.value) return null
    const feats = filteredData.value?.features || []
    const vals = feats.map((f) => f.properties[sizeBy.value]).filter(hasValue).map(Number)
    if (!vals.length) return null
    return { lo: Math.min(...vals), hi: Math.max(...vals) }
  })

  function radiusFor(props) {
    const base = pointRadius.value
    const s = sizeScale.value
    if (!s) return base * 1.5
    const v = props[sizeBy.value]
    if (!hasValue(v)) return base * 0.75
    return base + base * 2.25 * ((Number(v) - s.lo) / ((s.hi - s.lo) || 1))
  }

  function markerStyle(props) {
    const c = coloring.value
    const faded = hoverValue.value !== null && typeof c.labelOf === 'function'
      && c.labelOf(props) !== hoverValue.value
    const fill = faded ? pointOpacity.value * 0.12 : pointOpacity.value
    return {
      radius: radiusFor(props),
      fillColor: c.colorFn(props),
      fillOpacity: fill,
      stroke: pointOutline.value && !faded,
      weight: pointOutline.value && !faded ? 1 : 0,
      color: '#222',
      opacity: pointOutline.value && !faded ? fill : 0,
    }
  }

  return {
    colorBy, sizeBy, hoverValue,
    colorOptions, colorCoverageNote, coloring, legendValues, sizeScale,
    radiusFor, markerStyle,
    hoverEnter, hoverLeave, pickValue,
    // expose appearance refs needed by the watch in MushroomMap
    pointRadius, pointOpacity, pointOutline, colorSeed, activeColors, colorOverrides,
  }
}
