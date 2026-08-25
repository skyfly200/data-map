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
        <BarChart :title="`Observation-day temperature (°${tempUnit})`" :data="tempHighLowDist" :format="int" />
        <p class="note">Count of observations per 2° band, split into low and high day temperatures.</p>
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

      <ChartCard v-if="elevVsDoy.length">
        <ScatterChart title="Elevation vs. day of year" :data="elevVsDoy" :legend="clusterLegend"
          xLabel="Day of year" :yLabel="`Elevation (${unit})`"
          :xFormat="(v) => Math.round(v)" :yFormat="(v) => Math.round(v).toLocaleString()" />
        <p class="note">Each point is one observation, coloured by cluster — seasonal timing across elevation.</p>
      </ChartCard>

      <ChartCard v-if="elevVsTemp.length">
        <ScatterChart title="Elevation vs. observation-day high temp" :data="elevVsTemp" :legend="clusterLegend"
          :xLabel="`High temp (°${tempUnit})`" :yLabel="`Elevation (${unit})`"
          :xFormat="(v) => `${Math.round(v)}°`" :yFormat="(v) => Math.round(v).toLocaleString()" />
        <p class="note">Higher sites tend to be cooler on the day of the find.</p>
      </ChartCard>

      <ChartCard v-if="rainVsDoy.length">
        <ScatterChart title="7-day rain total vs. day of year" :data="rainVsDoy" :legend="clusterLegend"
          xLabel="Day of year" yLabel="Rain total (mm)"
          :xFormat="(v) => Math.round(v)" :yFormat="(v) => Math.round(v)" />
        <p class="note">Total precipitation in the 7 days before each find.</p>
      </ChartCard>

      <ChartCard v-if="phenologyBySpecies.length">
        <BoxPlot title="Fruiting season by species" :data="phenologyBySpecies" xLabel="Day of year"
          :format="(v) => Math.round(v)" />
        <p class="note">When each species (≥3 obs) is found through the year — the forager's calendar.</p>
      </ChartCard>

      <ChartCard v-if="elevationBySpecies.length">
        <BoxPlot :title="`Elevation range by species (${unit})`" :data="elevationBySpecies" :xLabel="`Elevation (${unit})`"
          :format="(v) => Math.round(v).toLocaleString()" />
        <p class="note">Elevation band each species (≥3 obs) prefers.</p>
      </ChartCard>

      <ChartCard v-if="clusterProfile.rows.length">
        <HeatmapChart title="Environmental cluster profiles" :rows="clusterProfile.rows"
          :cols="clusterProfile.cols" :matrix="clusterProfile.matrix" :format="(v) => v.toFixed(2)" />
        <p class="note">Mean of each feature per cluster, scaled 0–1 across clusters — what defines each group.</p>
      </ChartCard>

      <ChartCard v-if="speciesLandcover.rows.length">
        <HeatmapChart title="Species × land cover" :rows="speciesLandcover.rows"
          :cols="speciesLandcover.cols" :matrix="speciesLandcover.matrix" :format="(v) => `${Math.round(v)}`" />
        <p class="note">How many observations of each species fall in each land-cover class.</p>
      </ChartCard>

      <ChartCard v-if="rainBeforeDist.length">
        <BarChart title="Antecedent rainfall (7-day total before finds)" :data="rainBeforeDist" :format="int" />
        <p class="note">Distribution of total precipitation (mm) in the week before each observation.</p>
      </ChartCard>

      <ChartCard>
        <WindRose title="Slope aspect of finds" :values="aspectValues" />
        <p class="note">Which compass direction the ground faces at each find (from the DEM).</p>
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

const hasDayTemp = computed(() => rows.value.some((r) => hasValue(r.tmax) || hasValue(r.tmin)))
const hasTempHistory = computed(() => rows.value.some((r) => hasValue(r.tmax_d0)))

// ── Scatter plots (per-observation granularity, coloured by cluster) ──────────
const ptColor = (r) => (hasValue(r.cluster) ? colorFor(r.cluster) : UNCLUSTERED)
const clusterLegend = computed(() => {
  const seen = new Set()
  let hasNull = false
  for (const r of rows.value) { if (hasValue(r.cluster)) seen.add(r.cluster); else hasNull = true }
  const out = [...seen].sort((a, b) => a - b).map((c) => ({ label: `C${c}`, color: colorFor(c) }))
  if (hasNull) out.push({ label: '—', color: UNCLUSTERED })
  return out
})

const elevVsDoy = computed(() => rows.value
  .filter((r) => hasValue(r.elevation) && hasValue(r.day_of_year))
  .map((r) => ({ x: Number(r.day_of_year), y: elevValue(r.elevation), color: ptColor(r), label: r.species })))

const elevVsTemp = computed(() => rows.value
  .filter((r) => hasValue(r.elevation) && hasValue(r.tmax))
  .map((r) => ({ x: tempValue(r.tmax), y: elevValue(r.elevation), color: ptColor(r), label: r.species })))

const rainVsDoy = computed(() => rows.value
  .filter((r) => hasValue(r.day_of_year) && [0, 1, 2, 3, 4, 5, 6].some((o) => hasValue(r[`prcp_d${o}`])))
  .map((r) => {
    const total = [0, 1, 2, 3, 4, 5, 6].reduce((s, o) => s + (hasValue(r[`prcp_d${o}`]) ? Number(r[`prcp_d${o}`]) : 0), 0)
    return { x: Number(r.day_of_year), y: total, color: ptColor(r), label: r.species }
  }))

// ── Distribution charts (box plots, heatmaps, wind-rose) ─────────────────────
const MIN_PER_SPECIES = 3

function speciesGroups(valueFn) {
  const groups = new Map()
  for (const r of rows.value) {
    const v = valueFn(r)
    if (!hasValue(r.species) || v === null) continue
    if (!groups.has(r.species)) groups.set(r.species, [])
    groups.get(r.species).push(v)
  }
  return [...groups.entries()]
    .filter(([, vals]) => vals.length >= MIN_PER_SPECIES)
    .sort((a, b) => b[1].length - a[1].length)
    .map(([label, values], i) => ({ label, values, color: PALETTE[i % PALETTE.length] }))
}

const phenologyBySpecies = computed(() =>
  speciesGroups((r) => (hasValue(r.day_of_year) ? Number(r.day_of_year) : null)))

const elevationBySpecies = computed(() =>
  speciesGroups((r) => (hasValue(r.elevation) ? elevValue(r.elevation) : null)))

// Cluster centroids across the populated features, min-max scaled per feature.
const clusterProfile = computed(() => {
  const clusters = [...new Set(rows.value.map((r) => r.cluster).filter(hasValue))].sort((a, b) => a - b)
  const feats = [
    ['Elevation', (r) => r.elevation],
    ['High temp', (r) => r.tmax],
    ['7-day rain', (r) => [0, 1, 2, 3, 4, 5, 6].reduce((s, o) => s + (hasValue(r[`prcp_d${o}`]) ? Number(r[`prcp_d${o}`]) : 0), 0)],
    ['Day of year', (r) => r.day_of_year],
    ['Soil moist.', (r) => r.soil_moisture],
    ['Water ret.', (r) => r.water_retention],
  ].filter(([, fn]) => rows.value.some((r) => hasValue(fn(r))))
  if (!clusters.length || !feats.length) return { rows: [], cols: [], matrix: [] }

  // mean per cluster per feature
  const means = feats.map(([, fn]) => clusters.map((c) => {
    const vals = rows.value.filter((r) => r.cluster === c).map(fn).filter((v) => Number.isFinite(Number(v))).map(Number)
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null
  }))
  // scale each feature (row) 0–1 across clusters
  const matrix = means.map((row) => {
    const finite = row.filter((v) => Number.isFinite(v))
    const lo = Math.min(...finite), hi = Math.max(...finite)
    return row.map((v) => (Number.isFinite(v) ? (hi === lo ? 0.5 : (v - lo) / (hi - lo)) : null))
  })
  return { rows: feats.map(([l]) => l), cols: clusters.map((c) => `C${c}`), matrix }
})

// Species (rows) × land cover (cols) observation counts.
const speciesLandcover = computed(() => {
  const sp = [...new Set(rows.value.map((r) => r.species).filter(hasValue))]
    .map((s) => [s, rows.value.filter((r) => r.species === s).length])
    .filter(([, n]) => n >= MIN_PER_SPECIES).sort((a, b) => b[1] - a[1]).map(([s]) => s)
  const lc = [...new Set(rows.value.map((r) => r.land_cover_label).filter(hasValue))]
  if (!sp.length || !lc.length) return { rows: [], cols: [], matrix: [] }
  const matrix = sp.map((s) => lc.map((l) =>
    rows.value.filter((r) => r.species === s && r.land_cover_label === l).length))
  return { rows: sp, cols: lc, matrix }
})

const rainBeforeDist = computed(() => {
  const totals = rows.value
    .filter((r) => [0, 1, 2, 3, 4, 5, 6].some((o) => hasValue(r[`prcp_d${o}`])))
    .map((r) => [0, 1, 2, 3, 4, 5, 6].reduce((s, o) => s + (hasValue(r[`prcp_d${o}`]) ? Number(r[`prcp_d${o}`]) : 0), 0))
  if (!totals.length) return []
  const step = 10
  const max = Math.ceil(Math.max(...totals) / step) * step
  const bins = []
  for (let lo = 0; lo < Math.max(step, max); lo += step) {
    bins.push({ label: `${lo}–${lo + step} mm`, short: `${lo}`, value: totals.filter((v) => v >= lo && v < lo + step).length })
  }
  return bins
})

const aspectValues = computed(() => rows.value.map((r) => r.aspect).filter(hasValue).map(Number))

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

// Distribution of the observation-day high and low temperatures, in 2° bands.
const tempHighLowDist = computed(() => {
  const highVals = rows.value.map((r) => r.tmax).filter(hasValue).map((c) => tempValue(c))
  const lowVals = rows.value.map((r) => r.tmin).filter(hasValue).map((c) => tempValue(c))
  const combined = [...highVals, ...lowVals]
  if (!combined.length) return []

  const step = 2
  const min = Math.floor(Math.min(...combined) / step) * step
  const max = Math.ceil(Math.max(...combined) / step) * step
  const bins = []

  for (let lo = min; lo < max; lo += step) {
    const highCount = highVals.filter((v) => v >= lo && v < lo + step).length
    const lowCount = lowVals.filter((v) => v >= lo && v < lo + step).length
    bins.push({
      label: `Low ${lo}–${lo + step}°${tempUnit.value}`,
      short: `L ${lo}`,
      value: lowCount,
      color: '#1baf7a',
    })
    bins.push({
      label: `High ${lo}–${lo + step}°${tempUnit.value}`,
      short: `H ${lo}`,
      value: highCount,
      color: '#2a78d6',
    })
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
