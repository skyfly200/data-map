<template>
  <div class="explore">
    <div class="panel">
      <label class="ctrl">
        <span>Chart</span>
        <select v-model="chartType">
          <option value="scatter">Scatter</option>
          <option value="bar">Bar (aggregate)</option>
          <option value="box">Box plot by category</option>
          <option value="histogram">Histogram</option>
          <option value="heatmap">Heatmap</option>
        </select>
      </label>

      <!-- Scatter -->
      <template v-if="chartType === 'scatter'">
        <label class="ctrl"><span>X</span><select v-model="xField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Y</span><select v-model="yField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Colour</span><select v-model="colorField"><option value="">— none —</option><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
      </template>

      <!-- Bar -->
      <template v-else-if="chartType === 'bar'">
        <label class="ctrl"><span>Group by</span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure</span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
        <label class="ctrl chk"><input type="checkbox" v-model="horizontal" /> Horizontal</label>
      </template>

      <!-- Box -->
      <template v-else-if="chartType === 'box'">
        <label class="ctrl"><span>Group by</span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Value</span><select v-model="valueField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
      </template>

      <!-- Histogram -->
      <template v-else-if="chartType === 'histogram'">
        <label class="ctrl"><span>Value</span><select v-model="valueField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Bins</span><input type="number" min="4" max="30" v-model.number="bins" /></label>
      </template>

      <!-- Heatmap -->
      <template v-else-if="chartType === 'heatmap'">
        <label class="ctrl"><span>Rows</span><select v-model="rowField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Columns</span><select v-model="colField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure</span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
      </template>
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <ChartCard v-else class="stage">
      <ScatterChart v-if="chartType === 'scatter'" :title="title" :data="scatterData" :legend="coloring.legend"
        :xLabel="labelOf(xField)" :yLabel="labelOf(yField)" :xFormat="fmtOf(xField)" :yFormat="fmtOf(yField)" />
      <BarChart v-else-if="chartType === 'bar'" :title="title" :data="barData" :horizontal="horizontal" :format="barFmt" />
      <BoxPlot v-else-if="chartType === 'box'" :title="title" :data="boxData" :xLabel="labelOf(valueField)" :format="fmtOf(valueField)" />
      <BarChart v-else-if="chartType === 'histogram'" :title="title" :data="histogramData" :format="(v) => String(v)" />
      <HeatmapChart v-else-if="chartType === 'heatmap'" :title="title" :rows="heatmap.rows" :cols="heatmap.cols"
        :matrix="heatmap.matrix" :format="heatFmt" />
      <p v-if="isEmpty" class="empty">No data for this combination — try different fields.</p>
    </ChartCard>
  </div>
</template>

<script setup>
import { PALETTE, SERIES_1, UNCLUSTERED, colorFor, hasValue, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const { rows, error, pending, load } = useObservations()
const { unit, tempUnit, elevValue, tempValue } = useUnits()
onMounted(load)

// ── Field registry ────────────────────────────────────────────────────────────
const ALL_NUMERIC = [
  { key: 'elevation', label: 'Elevation', unit: 'elev' },
  { key: 'day_of_year', label: 'Day of year' },
  { key: 'tmax', label: 'High temp', unit: 'temp' },
  { key: 'tmin', label: 'Low temp', unit: 'temp' },
  { key: 'tavg', label: 'Avg temp', unit: 'temp' },
  { key: 'rain7', label: '7-day rain total (mm)' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'solar_exposure', label: 'Solar exposure' },
  { key: 'wind_exposure', label: 'Wind exposure' },
  { key: 'water_retention', label: 'Water retention' },
  { key: 'slope', label: 'Slope (°)' },
  { key: 'aspect', label: 'Aspect (°)' },
  { key: 'num_identification_agreements', label: 'ID agreements' },
]
const ALL_CATEGORY = [
  { key: 'species', label: 'Species' },
  { key: 'land_cover_label', label: 'Land cover' },
  { key: 'cluster', label: 'Cluster' },
]

function rawNum(r, key) {
  if (key === 'rain7') {
    const parts = [0, 1, 2, 3, 4, 5, 6].map((o) => r[`prcp_d${o}`]).filter(hasValue)
    return parts.length ? parts.reduce((s, v) => s + Number(v), 0) : null
  }
  return hasValue(r[key]) ? Number(r[key]) : null
}
function numVal(r, f) {
  const raw = rawNum(r, f.key)
  if (raw === null) return null
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
function fmtOf(key) {
  const f = ALL_NUMERIC.find((x) => x.key === key)
  if (f && (f.key === 'ndvi' || String(f.key).includes('exposure') || f.key === 'water_retention' || f.key === 'soil_moisture')) {
    return (v) => Number(v).toFixed(2)
  }
  return (v) => Math.round(v).toLocaleString()
}

// Only offer fields that actually have data in the loaded dataset.
const numericFields = computed(() => ALL_NUMERIC.filter((f) => rows.value.some((r) => rawNum(r, f.key) !== null)))
const categoryFields = computed(() => ALL_CATEGORY.filter((f) => rows.value.some((r) => catVal(r, f.key) !== null)))

// ── Builder state ─────────────────────────────────────────────────────────────
const chartType = ref('scatter')
const xField = ref('day_of_year')
const yField = ref('elevation')
const colorField = ref('cluster')
const groupField = ref('species')
const valueField = ref('elevation')
const measure = ref('count')
const rowField = ref('species')
const colField = ref('land_cover_label')
const bins = ref(10)
const horizontal = ref(false)

// ── Colouring for categorical fields ──────────────────────────────────────────
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
const coloring = computed(() => categoryColoring(colorField.value))

// ── Chart data ────────────────────────────────────────────────────────────────
const fx = () => numericFields.value.find((f) => f.key === xField.value) || { key: xField.value }
const fy = () => numericFields.value.find((f) => f.key === yField.value) || { key: yField.value }
const fv = () => numericFields.value.find((f) => f.key === valueField.value) || { key: valueField.value }

const scatterData = computed(() => rows.value.map((r) => {
  const x = numVal(r, fx()), y = numVal(r, fy())
  if (x === null || y === null) return null
  return { x, y, color: colorField.value ? coloring.value.colorOf(catVal(r, colorField.value)) : SERIES_1, label: r.species }
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
  const groups = groupBy(groupField.value)
  const out = []
  for (const [label, rs] of groups) {
    let value
    if (measure.value === 'count') value = rs.length
    else {
      const f = numericFields.value.find((x) => x.key === measure.value) || { key: measure.value }
      const vals = rs.map((r) => numVal(r, f)).filter((v) => v !== null)
      if (!vals.length) continue
      value = vals.reduce((s, v) => s + v, 0) / vals.length
    }
    out.push({ label, short: label, value, color: groupField.value === 'cluster' ? clusterColor(label) : SERIES_1 })
  }
  return out.sort((a, b) => b.value - a.value).slice(0, 25)
})
const barFmt = computed(() => (measure.value === 'count' ? (v) => String(v) : (v) => Number(v).toFixed(1)))

const histogramData = computed(() => {
  const f = fv()
  const vals = rows.value.map((r) => numVal(r, f)).filter((v) => v !== null)
  if (!vals.length) return []
  const lo = Math.min(...vals), hi = Math.max(...vals)
  const n = Math.max(4, Math.min(30, bins.value || 10))
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
  const groups = groupBy(groupField.value)
  const out = []
  let i = 0
  for (const [label, rs] of groups) {
    const values = rs.map((r) => numVal(r, fv())).filter((v) => v !== null)
    if (values.length >= 3) {
      out.push({ label, values, color: groupField.value === 'cluster' ? clusterColor(label) : PALETTE[i % PALETTE.length] })
      i++
    }
  }
  return out.sort((a, b) => b.values.length - a.values.length)
})

const heatmap = computed(() => {
  const rowVals = [...new Set(rows.value.map((r) => catVal(r, rowField.value)).filter((v) => v !== null))].slice(0, 15)
  const colVals = [...new Set(rows.value.map((r) => catVal(r, colField.value)).filter((v) => v !== null))].slice(0, 15)
  const f = numericFields.value.find((x) => x.key === measure.value)
  const matrix = rowVals.map((rv) => colVals.map((cv) => {
    const cell = rows.value.filter((r) => catVal(r, rowField.value) === rv && catVal(r, colField.value) === cv)
    if (measure.value === 'count') return cell.length
    const vals = cell.map((r) => numVal(r, f)).filter((v) => v !== null)
    return vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : null
  }))
  return { rows: rowVals, cols: colVals, matrix }
})
const heatFmt = computed(() => (measure.value === 'count' ? (v) => `${Math.round(v)}` : (v) => Number(v).toFixed(1)))

// ── Title + empty state ───────────────────────────────────────────────────────
const title = computed(() => {
  if (chartType.value === 'scatter') return `${labelOf(yField.value)} vs. ${labelOf(xField.value)}`
  if (chartType.value === 'bar') return measure.value === 'count' ? `Count by ${catLabel(groupField.value)}` : `Mean ${labelOf(measure.value)} by ${catLabel(groupField.value)}`
  if (chartType.value === 'box') return `${labelOf(valueField.value)} by ${catLabel(groupField.value)}`
  if (chartType.value === 'histogram') return `Distribution of ${labelOf(valueField.value)}`
  if (chartType.value === 'heatmap') return `${catLabel(rowField.value)} × ${catLabel(colField.value)}`
  return ''
})
function catLabel(key) { return (ALL_CATEGORY.find((f) => f.key === key) || { label: key }).label }

const isEmpty = computed(() => {
  if (chartType.value === 'scatter') return scatterData.value.length === 0
  if (chartType.value === 'bar') return barData.value.length === 0
  if (chartType.value === 'box') return boxData.value.length === 0
  if (chartType.value === 'histogram') return histogramData.value.length === 0
  if (chartType.value === 'heatmap') return heatmap.value.rows.length === 0
  return false
})
</script>

<style scoped>
.explore { padding: 16px 18px; display: flex; flex-direction: column; gap: 14px; height: 100%; }
.panel {
  display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center;
  background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 16px;
}
.ctrl { display: inline-flex; align-items: center; gap: 7px; font-size: 0.85rem; color: #374151; }
.ctrl > span { color: #6b7280; font-weight: 600; }
.ctrl select, .ctrl input[type="number"] {
  border: 1px solid #cbd2d9; border-radius: 6px; padding: 4px 8px; font-size: 0.85rem; background: #fff;
}
.ctrl input[type="number"] { width: 60px; }
.ctrl.chk { gap: 5px; }

.stage { flex: 1 1 auto; min-height: 420px; display: flex; flex-direction: column; justify-content: center; }
.empty { color: #9aa0a6; text-align: center; padding: 20px; }
.msg { padding: 16px; color: #555; }
.msg.error { color: #b00020; }
</style>
