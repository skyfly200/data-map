<template>
  <div class="charts-page">
    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <div v-else class="grid">
      <section class="card">
        <BarChart title="Observations per environmental cluster" :data="clusterData" :format="int" />
        <p class="note">Colours match the map. “Unclustered” = missing every clustering feature.</p>
      </section>

      <section class="card">
        <BarChart title="Enrichment coverage (values present)" :data="coverageData" :format="cov" horizontal />
        <p class="note">How many of the {{ rows.length }} observations carry each attribute. Gaps fill in as the full pipeline runs.</p>
      </section>

      <section class="card">
        <BarChart title="Observations by month" :data="monthData" :format="int" />
      </section>

      <section class="card">
        <BarChart title="Elevation distribution" :data="elevationData" :format="int" />
        <p class="note">Count of observations per elevation band (m).</p>
      </section>

      <section class="card">
        <BarChart title="Land cover" :data="landCoverData" :format="int" horizontal />
      </section>

      <section class="card">
        <BarChart title="Top species" :data="speciesData" :format="int" horizontal />
      </section>
    </div>
  </div>
</template>

<script setup>
import { PALETTE, UNCLUSTERED, colorFor, hasValue, useObservations } from '~/composables/useObservations'

const { rows, error, pending, load } = useObservations()
onMounted(load)

const int = (v) => String(v)
const cov = (v) => `${v}/${rows.value.length}`

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
  const vals = rows.value.map((r) => r.elevation).filter(hasValue)
  if (!vals.length) return []
  const min = Math.floor(Math.min(...vals) / 500) * 500
  const max = Math.ceil(Math.max(...vals) / 500) * 500
  const bins = []
  for (let lo = min; lo < max; lo += 500) {
    const n = vals.filter((v) => v >= lo && v < lo + 500).length
    bins.push({ label: `${lo}–${lo + 500} m`, short: `${lo / 1000}k`, value: n })
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
.card {
  background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px;
}
.note { margin: 8px 0 0; font-size: 0.78rem; color: #6b7280; }
.msg { padding: 16px; color: #555; }
.msg.error { color: #b00020; }
</style>
