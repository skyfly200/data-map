<template>
  <div class="charts-page">
    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <div v-else class="grid">
      <ChartCard>
        <BarChart title="Observations per environmental cluster" :data="clusterData" :format="int" />
        <p class="note">Colours match the map. “Unclustered” = missing every clustering feature.</p>
      </ChartCard>

      <ChartCard>
        <BarChart title="Avg. rain in the 7 days before an observation" :data="rainLeadUp" :format="mm" />
        <p class="note">Mean daily precipitation (mm) across all observations, by days before the find.</p>
      </ChartCard>

      <ChartCard>
        <BarChart title="Enrichment coverage (values present)" :data="coverageData" :format="cov" horizontal />
        <p class="note">How many of the {{ rows.length }} observations carry each attribute. Gaps fill in as the full pipeline runs.</p>
      </ChartCard>

      <ChartCard>
        <BarChart title="Observations by month" :data="monthData" :format="int" />
      </ChartCard>

      <ChartCard>
        <BarChart title="Observations by week of year" :data="weekData" :format="int" />
        <p class="note">Seasonal timing across all years (ISO week 1–53), ignoring which year.</p>
      </ChartCard>

      <ChartCard v-if="hasTempHistory">
        <BarChart :title="`Avg. daily high in the 7 days before (°${tempUnit})`" :data="tempLeadUp" :format="deg" />
        <p class="note">Mean daily high temperature across observations, by days before the find.</p>
      </ChartCard>

      <ChartCard v-if="hasDayTemp">
        <BarChart :title="`Observation-day high temperature (°${tempUnit})`" :data="tempHighDist" :format="int" />
        <p class="note">Count of observations per high-temperature band.</p>
      </ChartCard>

      <ChartCard>
        <BarChart title="Elevation distribution" :data="elevationData" :format="int" />
        <p class="note">Count of observations per elevation band ({{ unit }}).</p>
      </ChartCard>

      <ChartCard>
        <BarChart title="Land cover" :data="landCoverData" :format="int" horizontal />
      </ChartCard>

      <ChartCard>
        <BarChart title="Top species" :data="speciesData" :format="int" horizontal />
      </ChartCard>
    </div>
  </div>
</template>

<script setup>
import { PALETTE, UNCLUSTERED, colorFor, hasValue, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const { rows, error, pending, load } = useObservations()
const { unit, elevValue, tempUnit, tempValue } = useUnits()
onMounted(load)

const int = (v) => String(v)
const cov = (v) => `${v}/${rows.value.length}`
const mm = (v) => `${v}`
const deg = (v) => `${v}°`

const hasDayTemp = computed(() => rows.value.some((r) => hasValue(r.tmax)))
const hasTempHistory = computed(() => rows.value.some((r) => hasValue(r.tmax_d0)))

// Seasonal timing regardless of year: bucket day_of_year into ISO-ish weeks.
const weekData = computed(() => {
  const counts = new Map()
  for (const r of rows.value) {
    if (!hasValue(r.day_of_year)) continue
    const wk = Math.min(53, Math.max(1, Math.ceil(Number(r.day_of_year) / 7)))
    counts.set(wk, (counts.get(wk) || 0) + 1)
  }
  return [...counts.entries()].sort((a, b) => a[0] - b[0])
    .map(([wk, n]) => ({ label: `Week ${wk}`, short: `${wk}`, value: n }))
})

// Avg daily high (converted to the display unit) for the 7 days before a find.
const tempLeadUp = computed(() => [6, 5, 4, 3, 2, 1, 0].map((o) => {
  const vals = rows.value.map((r) => r[`tmax_d${o}`]).filter(hasValue).map((c) => tempValue(c))
  const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
  return { label: o === 0 ? 'day of' : `${o}d before`, short: o === 0 ? '0' : `-${o}`, value: Math.round(mean) }
}))

// Distribution of the observation-day high temperature, in the display unit.
const tempHighDist = computed(() => {
  const vals = rows.value.map((r) => r.tmax).filter(hasValue).map((c) => tempValue(c))
  if (!vals.length) return []
  const step = tempUnit.value === 'F' ? 10 : 5
  const min = Math.floor(Math.min(...vals) / step) * step
  const max = Math.ceil(Math.max(...vals) / step) * step
  const bins = []
  for (let lo = min; lo < max; lo += step) {
    const n = vals.filter((v) => v >= lo && v < lo + step).length
    bins.push({ label: `${lo}–${lo + step}°${tempUnit.value}`, short: `${lo}°`, value: n })
  }
  return bins
})

const rainLeadUp = computed(() => [6, 5, 4, 3, 2, 1, 0].map((o) => {
  const vals = rows.value.map((r) => r[`prcp_d${o}`]).filter(hasValue).map(Number)
  const mean = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
  return { label: o === 0 ? 'day of' : `${o}d before`, short: o === 0 ? '0' : `-${o}`, value: Number(mean.toFixed(2)) }
}))

function countBy(list, keyFn) {
  const m = new Map()
  for (const item of list) {
    const k = keyFn(item)
    if (k === null || k === undefined || k === '') continue
    m.set(k, (m.get(k) || 0) + 1)
  }
  return m
}

const clusterData = computed(() => {
  const counts = new Map()
  let unclustered = 0
  for (const r of rows.value) {
    if (hasValue(r.cluster)) counts.set(r.cluster, (counts.get(r.cluster) || 0) + 1)
    else unclustered++
  }
  const out = [...counts.entries()].sort((a, b) => a[0] - b[0])
    .map(([c, n]) => ({ label: `Cluster ${c}`, short: `C${c}`, value: n, color: colorFor(c) }))
  if (unclustered) out.push({ label: 'Unclustered', short: '—', value: unclustered, color: UNCLUSTERED })
  return out
})

const coverageData = computed(() => {
  const attrs = [
    ['NDVI', 'ndvi'], ['Soil moisture', 'soil_moisture'],
    ['Solar exposure', 'solar_exposure'], ['Wind exposure', 'wind_exposure'],
    ['Water retention', 'water_retention'], ['Elevation', 'elevation'],
    ['Land cover', 'land_cover_label'], ['Cluster', 'cluster'],
  ]
  return attrs.map(([label, key]) => ({
    label, value: rows.value.filter((r) => hasValue(r[key])).length,
  }))
})

const monthData = computed(() => {
  const counts = countBy(rows.value, (r) => (r.date ? String(r.date).slice(0, 7) : null))
  return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]))
    .map(([ym, n]) => ({ label: ym, short: ym.slice(2), value: n }))
})

const elevationData = computed(() => {
  const vals = rows.value.map((r) => r.elevation).filter(hasValue).map((m) => elevValue(m))
  if (!vals.length) return []
  const step = unit.value === 'ft' ? 1000 : 500
  const min = Math.floor(Math.min(...vals) / step) * step
  const max = Math.ceil(Math.max(...vals) / step) * step
  const bins = []
  for (let lo = min; lo < max; lo += step) {
    const n = vals.filter((v) => v >= lo && v < lo + step).length
    bins.push({
      label: `${lo.toLocaleString()}–${(lo + step).toLocaleString()} ${unit.value}`,
      short: `${(lo / 1000)}k`,
      value: n,
    })
  }
  return bins
})

const landCoverData = computed(() => {
  const counts = countBy(rows.value, (r) => r.land_cover_label)
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
    .map(([label, n]) => ({ label, value: n }))
})

const speciesData = computed(() => {
  const counts = countBy(rows.value, (r) => r.species)
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([label, n]) => ({ label, short: label, value: n }))
})
</script>

<style scoped>
.charts-page { padding: 16px 18px; }
.grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 16px;
}
.note { margin: 8px 0 0; font-size: 0.78rem; color: #6b7280; }
.msg { padding: 16px; color: #555; }
.msg.error { color: #b00020; }
</style>
