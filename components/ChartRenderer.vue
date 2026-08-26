<template>
  <ScatterChart v-if="config.type === 'scatter'" :title="title" :data="scatterData" :legend="coloring.legend"
    :xLabel="labelOf(config.xField)" :yLabel="labelOf(config.yField)" :xFormat="fmtOf(config.xField)" :yFormat="fmtOf(config.yField)"
    :todayX="todayX" :todayLabel="todayLabel" @select="$emit('select', $event)" />
  <BarChart v-else-if="config.type === 'bar'" :title="title" :data="barData" :horizontal="!!config.horizontal" :format="barFmt" />
  <BoxPlot v-else-if="config.type === 'box'" :title="title" :data="boxData" :xLabel="labelOf(config.valueField)" :format="fmtOf(config.valueField)" />
  <BarChart v-else-if="config.type === 'histogram'" :title="title" :data="histogramData" :format="(v) => String(v)" />
  <HeatmapChart v-else-if="config.type === 'heatmap'" :title="title" :rows="heatmap.rows" :cols="heatmap.cols"
    :matrix="heatmap.matrix" :format="heatFmt" />
  <LineChart v-else-if="config.type === 'line' || config.type === 'area'" :title="title" :data="lineSeries"
    :xLabel="labelOf(config.xField)" :yLabel="labelOf(config.yField)" :xFormat="fmtOf(config.xField)" :yFormat="fmtOf(config.yField)" />
  <PieChart v-else-if="config.type === 'donut'" :title="title" :data="donutData" :format="(v) => String(Math.round(v))" />
  <BarChart v-else-if="config.type === 'radar'" :title="title" :data="radarData" :format="(v) => String(v)" :horizontal="false" />
  <p v-if="isEmpty" class="cr-empty">No data for this combination.</p>
</template>

<script setup>
import { SERIES_1, UNCLUSTERED, categoryColor, hasValue, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'
import { ALL_NUMERIC, ALL_CATEGORY } from '~/composables/useChartFields'

const props = defineProps({ config: { type: Object, required: true } })
defineEmits(['select'])

const { rows } = useObservations()
const { unit, tempUnit, elevValue, tempValue } = useUnits()
const c = computed(() => props.config)

function currentDayOfYear() {
  const now = new Date()
  const start = new Date(now.getFullYear(), 0, 0)
  const diff = (now - start) / 86400000
  return Math.floor(diff) + 1
}

const todayX = computed(() => {
  if (!c.value.showToday || c.value.xField !== 'day_of_year') return null
  return currentDayOfYear()
})
const todayLabel = computed(() => 'Today')

function rawNum(r, key) {
  if (key === 'rain7') {
    const parts = [0, 1, 2, 3, 4, 5, 6].map((o) => r[`prcp_d${o}`]).filter(hasValue)
    return parts.length ? parts.reduce((s, v) => s + Number(v), 0) : null
  }
  return hasValue(r[key]) ? Number(r[key]) : null
}
function numFieldOf(key) { return ALL_NUMERIC.find((x) => x.key === key) || { key } }
function numVal(r, key) {
  const raw = rawNum(r, key)
  if (raw === null) return null
  const f = numFieldOf(key)
  if (f.unit === 'elev') return elevValue(raw)
  if (f.unit === 'temp') return tempValue(raw)
  return raw
}
function catVal(r, key) {
  if (key === 'cluster') return hasValue(r.cluster) ? `C${r.cluster}` : null
  return hasValue(r[key]) ? String(r[key]) : null
}
function labelOf(key) {
  const f = ALL_NUMERIC.find((x) => x.key === key)
  if (!f) return key
  if (f.unit === 'elev') return `${f.label} (${unit.value})`
  if (f.unit === 'temp') return `${f.label} (°${tempUnit.value})`
  return f.label
}
function catLabel(key) { return (ALL_CATEGORY.find((f) => f.key === key) || { label: key }).label }
function fmtOf(key) {
  const f = numFieldOf(key)
  if (['ndvi', 'soil_moisture', 'water_retention'].includes(f.key) || String(f.key).includes('exposure')) {
    return (v) => Number(v).toFixed(2)
  }
  return (v) => Math.round(v).toLocaleString()
}

// Colour a category value the same way the map does, so keys stay consistent.
function categoryColoring(field) {
  if (!field) return { colorOf: () => SERIES_1, legend: [] }
  const uniq = [...new Set(rows.value.map((r) => catVal(r, field)).filter((v) => v !== null))]
  if (field === 'cluster') uniq.sort()
  const colorOf = (v) => categoryColor(field, v)
  return { colorOf, legend: uniq.slice(0, 12).map((v) => ({ label: v, color: colorOf(v) })) }
}
const coloring = computed(() => categoryColoring(c.value.colorField))

// Colour a grouped mark (bar/box/donut/radar slice) by its category value, so
// the same category is the same colour here and on the map.
function groupColor(label) { return categoryColor(c.value.groupField, label) }

const scatterData = computed(() => rows.value.map((r) => {
  const x = numVal(r, c.value.xField), y = numVal(r, c.value.yField)
  if (x === null || y === null) return null
  return { x, y, color: c.value.colorField ? coloring.value.colorOf(catVal(r, c.value.colorField)) : SERIES_1, label: r.species, obs: r }
}).filter(Boolean))

function groupBy(field) {
  const m = new Map()
  for (const r of rows.value) {
    const k = catVal(r, field)
    if (k === null) continue
    if (!m.has(k)) m.set(k, [])
    m.get(k).push(r)
  }
  return m
}

const barData = computed(() => {
  const out = []
  for (const [label, rs] of groupBy(c.value.groupField)) {
    let value
    if (c.value.measure === 'count') value = rs.length
    else {
      const vals = rs.map((r) => numVal(r, c.value.measure)).filter((v) => v !== null)
      if (!vals.length) continue
      value = vals.reduce((s, v) => s + v, 0) / vals.length
    }
    out.push({ label, short: label, value, color: groupColor(label) })
  }
  return out.sort((a, b) => b.value - a.value).slice(0, 25)
})
const barFmt = computed(() => (c.value.measure === 'count' ? (v) => String(v) : (v) => Number(v).toFixed(1)))

// Real line/area series: mean of Y across bins of the ordered X axis, so a
// trend (e.g. mean elevation over day-of-year) reads as a connected curve.
const lineSeries = computed(() => {
  const xs = rows.value
    .map((r) => ({ x: numVal(r, c.value.xField), y: numVal(r, c.value.yField) }))
    .filter((d) => d.x !== null && d.y !== null)
  if (!xs.length) return []
  const lo = Math.min(...xs.map((d) => d.x)), hi = Math.max(...xs.map((d) => d.x))
  const n = 24
  const step = (hi - lo) / n || 1
  const bins = Array.from({ length: n }, () => [])
  for (const d of xs) bins[Math.min(n - 1, Math.floor((d.x - lo) / step))].push(d.y)
  const out = []
  bins.forEach((ys, i) => {
    if (ys.length) out.push({ x: lo + (i + 0.5) * step, y: ys.reduce((s, v) => s + v, 0) / ys.length })
  })
  return out
})

// Real donut: composition by category (top 8 + grey Other). Count, or the sum
// of a numeric measure — both are valid parts-of-a-whole.
const donutData = computed(() => {
  const entries = [...groupBy(c.value.groupField)].map(([label, rs]) => {
    let value
    if (c.value.measure === 'count') value = rs.length
    else value = rs.map((r) => numVal(r, c.value.measure)).filter((v) => v !== null).reduce((s, v) => s + v, 0)
    return { label, value }
  }).filter((e) => e.value > 0)
  entries.sort((a, b) => b.value - a.value)
  const top = entries.slice(0, 8)
  const rest = entries.slice(8)
  const hasOther = rest.length > 0
  if (hasOther) top.push({ label: `Other (${rest.length})`, value: rest.reduce((s, e) => s + e.value, 0) })
  return top.map((e, i) => ({
    ...e,
    color: (hasOther && i === top.length - 1) ? UNCLUSTERED : groupColor(e.label),
  }))
})

const radarData = computed(() => {
  const groups = groupBy(c.value.groupField)
  const out = []
  for (const [label, rs] of groups) {
    let value
    if (c.value.measure === 'count') value = rs.length
    else {
      const vals = rs.map((r) => numVal(r, c.value.measure)).filter((v) => v !== null)
      value = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
    }
    out.push({ label, short: label, value, color: groupColor(label) })
  }
  return out.sort((a, b) => b.value - a.value).slice(0, 12)
})

const histogramData = computed(() => {
  const vals = rows.value.map((r) => numVal(r, c.value.valueField)).filter((v) => v !== null)
  if (!vals.length) return []
  const lo = Math.min(...vals), hi = Math.max(...vals)
  const n = Math.max(4, Math.min(30, c.value.bins || 10))
  const step = (hi - lo) / n || 1
  const out = []
  for (let i = 0; i < n; i++) {
    const a = lo + i * step, b = a + step
    const count = vals.filter((v) => (i === n - 1 ? v >= a && v <= b : v >= a && v < b)).length
    out.push({ label: `${Math.round(a)}–${Math.round(b)}`, short: `${Math.round(a)}`, value: count })
  }
  return out
})

const boxData = computed(() => {
  const out = []
  for (const [label, rs] of groupBy(c.value.groupField)) {
    const values = rs.map((r) => numVal(r, c.value.valueField)).filter((v) => v !== null)
    if (values.length >= 3) {
      out.push({ label, values, color: groupColor(label) })
    }
  }
  // Cap to the most-sampled categories so a high-cardinality field (e.g. 200+
  // species) stays readable instead of running off-screen.
  return out.sort((a, b) => b.values.length - a.values.length).slice(0, 30)
})

const heatmap = computed(() => {
  const rowVals = [...new Set(rows.value.map((r) => catVal(r, c.value.rowField)).filter((v) => v !== null))].slice(0, 15)
  const colVals = [...new Set(rows.value.map((r) => catVal(r, c.value.colField)).filter((v) => v !== null))].slice(0, 15)
  const matrix = rowVals.map((rv) => colVals.map((cv) => {
    const cell = rows.value.filter((r) => catVal(r, c.value.rowField) === rv && catVal(r, c.value.colField) === cv)
    if (c.value.measure === 'count') return cell.length
    const vals = cell.map((r) => numVal(r, c.value.measure)).filter((v) => v !== null)
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
  }))
  return { rows: rowVals, cols: colVals, matrix }
})
const heatFmt = computed(() => (c.value.measure === 'count' ? (v) => `${Math.round(v)}` : (v) => Number(v).toFixed(1)))

const title = computed(() => {
  if (c.value.title) return c.value.title
  const t = c.value.type
  if (t === 'scatter') return `${labelOf(c.value.yField)} vs. ${labelOf(c.value.xField)}`
  if (t === 'bar') return c.value.measure === 'count' ? `Count by ${catLabel(c.value.groupField)}` : `Mean ${labelOf(c.value.measure)} by ${catLabel(c.value.groupField)}`
  if (t === 'line' || t === 'area') return `${labelOf(c.value.yField)} over ${labelOf(c.value.xField)}`
  if (t === 'box') return `${labelOf(c.value.valueField)} by ${catLabel(c.value.groupField)}`
  if (t === 'histogram') return `Distribution of ${labelOf(c.value.valueField)}`
  if (t === 'heatmap') return `${catLabel(c.value.rowField)} × ${catLabel(c.value.colField)}`
  if (t === 'radar' || t === 'donut') return `${c.value.measure === 'count' ? 'Count' : labelOf(c.value.measure)} by ${catLabel(c.value.groupField)}`
  return ''
})
defineExpose({ title })

const isEmpty = computed(() => {
  const t = c.value.type
  if (t === 'scatter') return scatterData.value.length === 0
  if (t === 'bar') return barData.value.length === 0
  if (t === 'line' || t === 'area') return lineSeries.value.length === 0
  if (t === 'donut') return donutData.value.length === 0
  if (t === 'radar') return radarData.value.length === 0
  if (t === 'box') return boxData.value.length === 0
  if (t === 'histogram') return histogramData.value.length === 0
  if (t === 'heatmap') return heatmap.value.rows.length === 0
  return false
})
</script>

<style scoped>
.cr-empty { color: var(--muted); text-align: center; padding: 20px; }
</style>
