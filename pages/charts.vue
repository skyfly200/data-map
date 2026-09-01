<template>
  <div class="charts-page">
    <nav class="tabs">
      <button :class="{ on: tab === 'gallery' }" @click="tab = 'gallery'">Charts</button>
      <button :class="{ on: tab === 'build' }" @click="tab = 'build'">Build</button>
    </nav>

    <ChartBuilder v-if="tab === 'build'" class="build-pane" />

    <template v-else>
    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <template v-else>
      <!-- Layout controls: reorder / hide the preset charts -->
      <div class="layout-bar">
        <button class="lb-btn" :class="{ on: layout.editing.value }" @click="layout.editing.value = !layout.editing.value">
          {{ layout.editing.value ? '✓ Done arranging' : '⇅ Arrange charts' }}
        </button>
        <span v-if="layout.editing.value" class="lb-hint">Use ‹ › to reorder and ✕ to hide.</span>
        <!-- Cards register themselves as they render, which happens after this
             bar is serialised on the server. Render the tally on the client
             only, so SSR's "0 shown" never mismatches the real count. -->
        <ClientOnly>
          <span class="lb-count">{{ layout.visibleCount.value }} shown</span>
        </ClientOnly>
        <button v-if="layout.hiddenCharts.value.length" class="lb-btn ghost" @click="layout.showAll()">
          Show all ({{ layout.hiddenCharts.value.length }} hidden)
        </button>
        <button v-if="layout.editing.value" class="lb-btn ghost" @click="layout.reset()">Reset layout</button>
      </div>

      <div v-if="layout.editing.value && layout.hiddenCharts.value.length" class="hidden-bar">
        <span class="hb-label">Hidden:</span>
        <button v-for="h in layout.hiddenCharts.value" :key="h.id" class="hb-chip" :title="`Show “${h.title}”`"
                @click="layout.show(h.id)">
          {{ h.title }} <span class="plus">+</span>
        </button>
      </div>

      <!-- Saved custom charts (from the Build tab), reorderable -->
      <section v-if="saved.charts.value.length" class="saved">
        <h2 class="saved-title">My charts</h2>
        <div class="grid">
          <ChartCard v-for="(chart, i) in saved.charts.value" :key="chart.id">
            <div class="saved-tools">
              <button title="Move left" :disabled="i === 0" @click="saved.move(chart.id, -1)">‹</button>
              <button title="Move right" :disabled="i === saved.charts.value.length - 1" @click="saved.move(chart.id, 1)">›</button>
              <button title="Remove" class="rm" @click="saved.remove(chart.id)">✕</button>
            </div>
            <ChartRenderer :config="chart" @select="selected = $event" />
          </ChartCard>
        </div>
      </section>

    <div class="grid">
      <GalleryChart id="clusters">
        <BarChart title="Observations per environmental cluster" :data="clusterData" :format="int" />
        <p class="note">Colours match the map. “Unclustered” = missing every clustering feature.</p>
      </GalleryChart>

      <GalleryChart id="rain-leadup">
        <BarChart title="Avg. rain in the 7 days before an observation" :data="rainLeadUp" :format="mm" />
        <p class="note">Mean daily precipitation (mm) across all observations, by days before the find.</p>
      </GalleryChart>

      <GalleryChart id="coverage">
        <BarChart title="Enrichment coverage (values present)" :data="coverageData" :format="cov" horizontal />
        <p class="note">How many of the {{ rows.length }} observations carry each attribute. Gaps fill in as the full pipeline runs.</p>
      </GalleryChart>

      <GalleryChart id="by-month">
        <BarChart title="Observations by month" :data="monthData" :format="int" />
      </GalleryChart>

      <GalleryChart id="by-week">
        <BarChart title="Observations by week of year" :data="weekData" :format="int" />
        <p class="note">Seasonal timing across all years (ISO week 1–53), ignoring which year.</p>
      </GalleryChart>

      <GalleryChart id="temp-leadup" v-if="hasTempHistory">
        <BarChart :title="`Avg. daily high in the 7 days before (°${tempUnit})`" :data="tempLeadUp" :format="deg" />
        <p class="note">Mean daily high temperature across observations, by days before the find.</p>
      </GalleryChart>

      <GalleryChart id="temp-dist" v-if="hasDayTemp">
        <BarChart :title="`Observation-day temperature (°${tempUnit})`" :data="tempHighLowDist" :format="int" />
        <p class="note">Count of observations per 2° band, split into low and high day temperatures.</p>
      </GalleryChart>

      <GalleryChart id="elevation-dist">
        <BarChart title="Elevation distribution" :data="elevationData" :format="int" />
        <p class="note">Count of observations per elevation band ({{ unit }}).</p>
      </GalleryChart>

      <GalleryChart id="land-cover">
        <BarChart title="Land cover" :data="landCoverData" :format="int" horizontal />
      </GalleryChart>

      <GalleryChart id="top-species">
        <BarChart title="Top species" :data="speciesData" :format="int" horizontal />
      </GalleryChart>

      <GalleryChart id="elev-vs-doy" v-if="elevVsDoy.length">
        <ScatterChart title="Elevation vs. day of year" :data="elevVsDoy" :legend="clusterLegend"
          xLabel="Day of year" :yLabel="`Elevation (${unit})`"
          :xFormat="(v) => Math.round(v)" :yFormat="(v) => Math.round(v).toLocaleString()"
          @select="selected = $event" />
        <p class="note">Each point is one observation, coloured by cluster — seasonal timing across elevation.</p>
      </GalleryChart>

      <GalleryChart id="elev-vs-temp" v-if="elevVsTemp.length">
        <ScatterChart title="Elevation vs. observation-day high temp" :data="elevVsTemp" :legend="clusterLegend"
          :xLabel="`High temp (°${tempUnit})`" :yLabel="`Elevation (${unit})`"
          :xFormat="(v) => `${Math.round(v)}°`" :yFormat="(v) => Math.round(v).toLocaleString()"
          @select="selected = $event" />
        <p class="note">Higher sites tend to be cooler on the day of the find.</p>
      </GalleryChart>

      <GalleryChart id="rain-vs-doy" v-if="rainVsDoy.length">
        <ScatterChart title="7-day rain total vs. day of year" :data="rainVsDoy" :legend="clusterLegend"
          xLabel="Day of year" yLabel="Rain total (mm)"
          :xFormat="(v) => Math.round(v)" :yFormat="(v) => Math.round(v)"
          @select="selected = $event" />
        <p class="note">Total precipitation in the 7 days before each find.</p>
      </GalleryChart>

      <GalleryChart id="phenology" v-if="phenologyBySpecies.length">
        <BoxPlot title="Fruiting season by species" :data="phenologyBySpecies" xLabel="Day of year"
          :format="(v) => Math.round(v)" />
        <p class="note">When each species (≥3 obs) is found through the year — the forager's calendar.</p>
      </GalleryChart>

      <GalleryChart id="elevation-by-species" v-if="elevationBySpecies.length">
        <BoxPlot :title="`Elevation range by species (${unit})`" :data="elevationBySpecies" :xLabel="`Elevation (${unit})`"
          :format="(v) => Math.round(v).toLocaleString()" />
        <p class="note">Elevation band each species (≥3 obs) prefers.</p>
      </GalleryChart>

      <GalleryChart id="cluster-profile" v-if="clusterProfile.rows.length">
        <HeatmapChart title="Environmental cluster profiles" :rows="clusterProfile.rows"
          :cols="clusterProfile.cols" :matrix="clusterProfile.matrix" :format="(v) => v.toFixed(2)" />
        <p class="note">Mean of each feature per cluster, scaled 0–1 across clusters — what defines each group.</p>
      </GalleryChart>

      <GalleryChart id="species-landcover" v-if="speciesLandcover.rows.length">
        <HeatmapChart title="Species × land cover" :rows="speciesLandcover.rows"
          :cols="speciesLandcover.cols" :matrix="speciesLandcover.matrix" :format="(v) => `${Math.round(v)}`" />
        <p class="note">How many observations of each species fall in each land-cover class.</p>
      </GalleryChart>

      <GalleryChart id="antecedent-rain" v-if="rainBeforeDist.length">
        <BarChart title="Antecedent rainfall (7-day total before finds)" :data="rainBeforeDist" :format="int" />
        <p class="note">Distribution of total precipitation (mm) in the week before each observation.</p>
      </GalleryChart>

      <GalleryChart id="aspect">
        <WindRose title="Slope aspect of finds" :values="aspectValues" />
        <p class="note">Which compass direction the ground faces at each find (from the DEM).</p>
      </GalleryChart>
    </div>
    </template>

    <ObservationDrawer :selected="selected" @close="selected = null" />
    </template>
  </div>
</template>

<script setup>
import { PALETTE, UNCLUSTERED, colorFor, hasValue, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'
import { useSavedCharts } from '~/composables/useSavedCharts'

// Two tabs on this page: the preset chart gallery and the chart builder.
// Tab lives in the URL query so /charts?tab=build deep-links (and the old
// /explore route redirects here).
const route = useRoute()
const router = useRouter()
const tab = computed({
  get: () => (route.query.tab === 'build' ? 'build' : 'gallery'),
  set: (v) => router.replace({ query: { ...route.query, tab: v } }),
})

const saved = useSavedCharts()
const layout = useChartLayout()
const { rows, error, pending, load } = useObservations()
const { unit, elevValue, tempUnit, tempValue } = useUnits()
onMounted(() => { load(); saved.loadFromStorage(); layout.loadFromStorage() })

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
  .map((r) => ({ x: Number(r.day_of_year), y: elevValue(r.elevation), color: ptColor(r), label: r.species, obs: r })))

const elevVsTemp = computed(() => rows.value
  .filter((r) => hasValue(r.elevation) && hasValue(r.tmax))
  .map((r) => ({ x: tempValue(r.tmax), y: elevValue(r.elevation), color: ptColor(r), label: r.species, obs: r })))

const rainVsDoy = computed(() => rows.value
  .filter((r) => hasValue(r.day_of_year) && [0, 1, 2, 3, 4, 5, 6].some((o) => hasValue(r[`prcp_d${o}`])))
  .map((r) => {
    const total = [0, 1, 2, 3, 4, 5, 6].reduce((s, o) => s + (hasValue(r[`prcp_d${o}`]) ? Number(r[`prcp_d${o}`]) : 0), 0)
    return { x: Number(r.day_of_year), y: total, color: ptColor(r), label: r.species, obs: r }
  }))

// Click a scatter point to open its observation (iNat link + open on map).
const selected = ref(null)

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

// Day offsets of the 7-day precipitation history columns (prcp_d0..d6).
const PRCP_OFFSETS = [0, 1, 2, 3, 4, 5, 6]
const rain7 = (r) => PRCP_OFFSETS.reduce(
  (s, o) => s + (hasValue(r[`prcp_d${o}`]) ? Number(r[`prcp_d${o}`]) : 0), 0)

// Cluster centroids across the populated features, min-max scaled per feature.
//
// Accumulated in ONE pass over the rows. The earlier shape — a filter of the
// whole set per (feature, cluster) pair, plus one more per feature to test for
// presence — meant dozens of full scans of ~48k rows.
const CLUSTER_FEATURES = [
  ['Elevation', (r) => r.elevation],
  ['High temp', (r) => r.tmax],
  ['7-day rain', rain7],
  ['Day of year', (r) => r.day_of_year],
  ['Soil moist.', (r) => r.soil_moisture],
  ['Water ret.', (r) => r.water_retention],
]

const clusterProfile = computed(() => {
  const present = CLUSTER_FEATURES.map(() => false)
  // feature index → cluster → running { sum, n }
  const acc = CLUSTER_FEATURES.map(() => new Map())
  const clusterSet = new Set()

  for (const r of rows.value) {
    const clustered = hasValue(r.cluster)
    if (clustered) clusterSet.add(r.cluster)
    for (let i = 0; i < CLUSTER_FEATURES.length; i++) {
      const raw = CLUSTER_FEATURES[i][1](r)
      if (!hasValue(raw)) continue
      present[i] = true
      const v = Number(raw)
      if (!clustered || !Number.isFinite(v)) continue
      const byCluster = acc[i]
      const cur = byCluster.get(r.cluster)
      if (cur) { cur.sum += v; cur.n += 1 } else byCluster.set(r.cluster, { sum: v, n: 1 })
    }
  }

  const clusters = [...clusterSet].sort((a, b) => a - b)
  const keep = CLUSTER_FEATURES.map((f, i) => [f, i]).filter(([, i]) => present[i])
  if (!clusters.length || !keep.length) return { rows: [], cols: [], matrix: [] }

  const means = keep.map(([, i]) => clusters.map((c) => {
    const cell = acc[i].get(c)
    return cell && cell.n ? cell.sum / cell.n : null
  }))
  // scale each feature (row) 0–1 across clusters
  const matrix = means.map((row) => {
    const finite = row.filter((v) => Number.isFinite(v))
    const lo = Math.min(...finite), hi = Math.max(...finite)
    return row.map((v) => (Number.isFinite(v) ? (hi === lo ? 0.5 : (v - lo) / (hi - lo)) : null))
  })
  return { rows: keep.map(([[l]]) => l), cols: clusters.map((c) => `C${c}`), matrix }
})

// Species (rows) × land cover (cols) observation counts.
//
// Also one pass. This was the single most expensive thing on the page: it
// scanned all ~48k rows once per species to get the totals (~975 species), then
// again per species × land-cover cell to fill the matrix — tens of millions of
// row visits for a table of counts.
const speciesLandcover = computed(() => {
  const totals = new Map()            // species → observations
  const cells = new Map()             // species → (land cover → count)
  const covers = new Set()

  for (const r of rows.value) {
    if (hasValue(r.land_cover_label)) covers.add(r.land_cover_label)
    if (!hasValue(r.species)) continue
    totals.set(r.species, (totals.get(r.species) || 0) + 1)
    if (!hasValue(r.land_cover_label)) continue
    let byCover = cells.get(r.species)
    if (!byCover) cells.set(r.species, byCover = new Map())
    byCover.set(r.land_cover_label, (byCover.get(r.land_cover_label) || 0) + 1)
  }

  const sp = [...totals.entries()]
    .filter(([, n]) => n >= MIN_PER_SPECIES)
    .sort((a, b) => b[1] - a[1])
    .map(([s]) => s)
  const lc = [...covers]
  if (!sp.length || !lc.length) return { rows: [], cols: [], matrix: [] }

  const matrix = sp.map((s) => {
    const byCover = cells.get(s)
    return lc.map((l) => byCover?.get(l) || 0)
  })
  return { rows: sp, cols: lc, matrix }
})

const rainBeforeDist = computed(() => {
  const totals = rows.value
    .filter((r) => PRCP_OFFSETS.some((o) => hasValue(r[`prcp_d${o}`])))
    .map(rain7)
  if (!totals.length) return []
  const step = 10
  const max = Math.ceil(Math.max(...totals) / step) * step
  const binCount = Math.max(1, Math.ceil(Math.max(step, max) / step))
  // Bucket in one pass rather than re-filtering every total per bin.
  const counts = new Array(binCount).fill(0)
  for (const v of totals) {
    const i = Math.floor(v / step)
    if (i >= 0 && i < binCount) counts[i] += 1
  }
  const bins = counts.map((value, i) => ({
    label: `${i * step}–${(i + 1) * step} mm`, short: `${i * step}`, value,
  }))
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
.tabs { display: flex; gap: 4px; margin: -4px 0 14px; border-bottom: 1px solid var(--border); }
.tabs button {
  border: 0; background: transparent; color: var(--muted); cursor: pointer;
  padding: 8px 16px; font-size: 0.92rem; font-weight: 600; border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tabs button:hover { color: var(--text); }
.tabs button.on { color: var(--text); border-bottom-color: var(--accent); }
.build-pane { height: calc(100vh - 150px); min-height: 440px; }
.build-pane :deep(.explore) { padding: 0; }
.grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 16px;
}
.grid > * {
  min-width: 0;
}
.layout-bar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 12px; }
.lb-btn {
  border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer;
  border-radius: 6px; padding: 5px 12px; font-size: 0.82rem; font-weight: 600;
}
.lb-btn:hover { background: var(--surface-2); }
.lb-btn.on { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
.lb-btn.ghost { font-weight: 500; color: var(--muted); }
.lb-btn.ghost:hover { color: var(--text); }
.lb-hint, .lb-count { font-size: 0.8rem; color: var(--muted); }
.lb-count { margin-left: auto; }

.hidden-bar {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: -4px 0 14px;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px;
}
.hb-label { font-size: 0.8rem; font-weight: 600; color: var(--muted); }
.hb-chip {
  border: 1px solid var(--border); background: var(--surface); color: var(--text); cursor: pointer;
  border-radius: 999px; padding: 3px 10px; font-size: 0.78rem;
}
.hb-chip:hover { background: var(--surface-3); }
.hb-chip .plus { color: var(--accent); font-weight: 700; }

.note { margin: 8px 0 0; font-size: 0.78rem; color: var(--muted); }
.msg { padding: 16px; color: var(--muted); }
.msg.error { color: var(--danger); }

.saved { margin-bottom: 22px; }
.saved-title { margin: 0 0 10px; font-size: 1rem; color: var(--text); }
.saved-tools { position: absolute; top: 8px; right: 34px; display: flex; gap: 2px; z-index: 3; }
.saved-tools button {
  border: 1px solid var(--border); background: var(--surface); color: var(--muted); cursor: pointer;
  width: 22px; height: 22px; border-radius: 5px; font-size: 0.85rem; line-height: 1; padding: 0;
}
.saved-tools button:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.saved-tools button:disabled { opacity: 0.35; cursor: default; }
.saved-tools .rm:hover { background: #fdecec; color: #b00020; border-color: #f5c2c2; }
</style>
