<template>
  <ScatterChart v-if="config.type === 'scatter'" :title="title" :data="scatterData" :legend="coloring.legend"
    :xLabel="labelOf(config.xField)" :yLabel="labelOf(config.yField)" :xFormat="fmtOf(config.xField)" :yFormat="fmtOf(config.yField)"
    :todayX="todayX" :todayLabel="todayLabel" />
  <BarChart v-else-if="config.type === 'bar'" :title="title" :data="barData" :horizontal="!!config.horizontal" :format="barFmt" />
  <BoxPlot v-else-if="config.type === 'box'" :title="title" :data="boxData" :xLabel="labelOf(config.valueField)" :format="fmtOf(config.valueField)" />
  <BarChart v-else-if="config.type === 'histogram'" :title="title" :data="histogramData" :format="(v) => String(v)" />
  <HeatmapChart v-else-if="config.type === 'heatmap'" :title="title" :rows="heatmap.rows" :cols="heatmap.cols"
    :matrix="heatmap.matrix" :format="heatFmt" />
  <BarChart v-else-if="config.type === 'line' || config.type === 'area'" :title="title" :data="lineData" :format="lineFmt" :horizontal="false" />
  <BarChart v-else-if="config.type === 'radar' || config.type === 'donut'" :title="title" :data="radarData" :format="(v) => String(v)" :horizontal="false" />
  <p v-if="isEmpty" class="cr-empty">No data for this combination.</p>
</template>

<script setup>
import { PALETTE, SERIES_1, UNCLUSTERED, colorFor, hasValue, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'
import { ALL_NUMERIC, ALL_CATEGORY } from '~/composables/useChartFields'

const props = defineProps({ config: { type: Object, required: true } })

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

function clusterColor(label) {
  const n = typeof label === 'string' && label.startsWith('C') ? Number(label.slice(1)) : Number(label)
  return Number.isFinite(n) ? colorFor(n) : UNCLUSTERED
}
function categoryColoring(field) {
  if (!field) return { colorOf: () => SERIES_1, legend: [] }
  const uniq = [...new Set(rows.value.map((r) => catVal(r, field)).filter((v) => v !== null))]
  if (field === 'cluster') {
    uniq.sort()
    return { colorOf: clusterColor, legend: uniq.map((v) => ({ label: v, color: clusterColor(v) })) }
  }
  const map = new Map(uniq.map((v, i) => [v, PALETTE[i % PALETTE.length]]))
  return { colorOf: (v) => map.get(v) || UNCLUSTERED, legend: uniq.slice(0, 8).map((v) => ({ label: v, color: map.get(v) })) }
}
const coloring = computed(() => categoryColoring(c.value.colorField))

const scatterData = computed(() => rows.value.map((r) => {
  const x = numVal(r, c.value.xField), y = numVal(r, c.value.yField)
  if (x === null || y === null) return null
  return { x, y, color: c.value.colorField ? coloring.value.colorOf(catVal(r, c.value.colorField)) : SERIES_1, label: r.species }
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
    out.push({ label, short: label, value, color: c.value.groupField === 'cluster' ? clusterColor(label) : SERIES_1 })
  }
  return out.sort((a, b) => b.value - a.value).slice(0, 25)
})
const barFmt = computed(() => (c.value.measure === 'count' ? (v) => String(v) : (v) => Number(v).toFixed(1)))

const lineData = computed(() => {
  const grouped = new Map()
  for (const r of rows.value) {
    const x = numVal(r, c.value.xField)
    const y = numVal(r, c.value.yField)
    if (x === null || y === null) continue
    const key = c.value.groupField ? catVal(r, c.value.groupField) || 'Unassigned' : 'All'
    if (!grouped.has(key)) grouped.set(key, [])
    grouped.get(key).push({ x, y })
  }

  const out = []
  for (const [key, pts] of grouped.entries()) {
    const ordered = [...pts].sort((a, b) => a.x - b.x)
    const total = ordered.reduce((sum, p) => sum + p.y, 0)
    out.push({
      label: key,
      short: key,
      value: total / Math.max(1, ordered.length),
      color: c.value.groupField === 'cluster' ? clusterColor(key) : PALETTE[(out.length) % PALETTE.length],
    })
  }
  return out.slice(0, 12)
})
const lineFmt = computed(() => (v) => Number(v).toFixed(1))

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
    out.push({ label, short: label, value, color: c.value.groupField === 'cluster' ? clusterColor(label) : SERIES_1 })
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
  let i = 0
  for (const [label, rs] of groupBy(c.value.groupField)) {
    const values = rs.map((r) => numVal(r, c.value.valueField)).filter((v) => v !== null)
    if (values.length >= 3) {
      out.push({ label, values, color: c.value.groupField === 'cluster' ? clusterColor(label) : PALETTE[i % PALETTE.length] })
      i++
    }
  }
  return out.sort((a, b) => b.values.length - a.values.length)
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
  if (t === 'line' || t === 'area') return c.value.groupField ? `${labelOf(c.value.yField)} by ${catLabel(c.value.groupField)}` : `${labelOf(c.value.yField)} over ${labelOf(c.value.xField)}`
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
  if (t === 'bar' || t === 'line' || t === 'area' || t === 'radar' || t === 'donut') return (lineData.value.length === 0 && radarData.value.length === 0 && barData.value.length === 0)
  if (t === 'box') return boxData.value.length === 0
  if (t === 'histogram') return histogramData.value.length === 0
  if (t === 'heatmap') return heatmap.value.rows.length === 0
  return false
})
</script>

<style scoped>
.cr-empty { color: #9aa0a6; text-align: center; padding: 20px; }
</style>
