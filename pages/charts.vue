<template>
  <div class="charts-page">
    <nav class="tabs">
      <button :class="{ on: tab === 'gallery' }" :title="tip('The preset chart gallery', '1')"
              @click="tab = 'gallery'">Charts</button>
      <button :class="{ on: tab === 'build' }" :title="tip('Compose your own chart', '2')"
              @click="tab = 'build'">Build</button>
      <button :class="{ on: tab === 'analysis' }" :title="tip('Statistical analysis', '3')"
              @click="tab = 'analysis'">Analysis</button>
    </nav>

    <!-- Lazy: the builder's chunk loads only when the Build tab is opened, not
         on the gallery a reader lands on first (V12-PERF-3). -->
    <LazyChartBuilder v-if="tab === 'build'" class="build-pane" />

    <!-- ── Analysis pane ────────────────────────────────────────────────── -->
    <template v-else-if="tab === 'analysis'">
      <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
      <p v-else-if="pending && !rows.length" class="msg">Loading…</p>
      <template v-else>
        <nav class="analysis-tabs">
          <button v-for="(t, i) in ANALYSIS_TABS" :key="t.key"
                  :class="{ on: analysisTab === t.key }"
                  :title="tip(t.hint, String(i + 1))"
                  @click="analysisTab = t.key">
            {{ t.label }}
          </button>
          <HelpLink :option="analysisTabDocId" />
          <ShareMenu class="tab-share" :title="analysisShareTitle" />
        </nav>

        <p class="scope">
          Analysing <strong>{{ rows.length.toLocaleString() }}</strong> observations, whatever the filters on the Data page currently select.
          <HelpLink option="analysis-scope" />
        </p>

        <section v-if="analysisTab === 'drivers'" class="an-pane">
          <ChartCard class="wide">
            <HeatmapChart title="How the environmental variables relate (Spearman ρ)"
              :rows="corrLabels" :cols="corrLabels" :matrix="corr.matrix" :format="rho" />
            <p class="note">
              Rank correlation, so a relationship counts even when it bends. Each cell uses
              only the rows where <em>both</em> variables are present, because coverage is
              very uneven: dropping rows missing any variable would compute the whole
              matrix on an unrepresentative remainder.
            </p>
          </ChartCard>
          <ChartCard class="wide">
            <h3 class="ct">Strongest relationships</h3>
            <table class="tbl">
              <thead><tr><th>Pair</th><th class="num">ρ</th><th class="num">n</th><th>Reading</th></tr></thead>
              <tbody>
                <tr v-for="p in corr.pairs.slice(0, 12)" :key="`${p.a}-${p.b}`">
                  <td>{{ p.a }} ↔ {{ p.b }}</td>
                  <td class="num" :style="{ color: rhoColor(p.rho) }">{{ p.rho.toFixed(2) }}</td>
                  <td class="num">{{ p.n.toLocaleString() }}</td>
                  <td class="muted">{{ strength(p.rho) }}</td>
                </tr>
              </tbody>
            </table>
            <p class="note">
              Correlation is not cause, and two confounds run through every row of this
              table. <strong>Season</strong> is the big one: high-elevation finds happen in
              summer and low ones in spring and autumn, which is why elevation and
              temperature appear to rise together, hold the month still and that
              relationship flattens to roughly zero. <strong>Effort</strong> is the other:
              people record where people go.
            </p>
          </ChartCard>
        </section>

        <section v-else-if="analysisTab === 'species'" class="an-pane">
          <ChartCard class="wide">
            <HeatmapChart title="How each species differs from average (standard deviations)"
              :rows="profileRows" :cols="profileCols" :matrix="profiles.matrix" :format="z" />
            <p class="note">
              Positive means this species is found higher, warmer, wetter or later than the
              dataset average. Standard deviations, not raw values, so columns in different
              units can be read side by side. Species with at least {{ MIN_SPECIES_OBS }} records.
            </p>
          </ChartCard>
          <ChartCard class="wide">
            <h3 class="ct">Species found together</h3>
            <table v-if="pairs.length" class="tbl">
              <thead><tr><th>Pair</th><th class="num">Shared places</th><th class="num">Lift</th></tr></thead>
              <tbody>
                <tr v-for="p in pairs" :key="`${p.a}|${p.b}`">
                  <td><em>{{ p.a }}</em> + <em>{{ p.b }}</em></td>
                  <td class="num">{{ p.cells }}</td>
                  <td class="num">{{ p.lift.toFixed(1) }}×</td>
                </tr>
              </tbody>
            </table>
            <p v-else class="note">Not enough co-located records in the current selection.</p>
            <p class="note">
              Same ~5 km cell, same month-long window. Scored by <strong>lift</strong>: how
              much more often a pair appears together than their individual frequencies
              predict, because a raw count would just rank the two commonest species first
              whether or not they have anything to do with each other.
            </p>
          </ChartCard>
        </section>

        <section v-else-if="analysisTab === 'seasons'" class="an-pane">
          <ChartCard>
            <LineChart title="Middle of the season, by year" :data="seasonTiming"
              xLabel="Year" yLabel="Median day of year"
              :xFormat="(v) => String(Math.round(v))" :yFormat="(v) => String(Math.round(v))" />
            <p class="note">
              Median, not mean: a handful of winter records would drag a mean badly.
              An upward slope means the season is arriving later.
            </p>
          </ChartCard>
          <ChartCard>
            <LineChart :title="`Median elevation of finds, by year (${unit})`" :data="elevationTrend"
              xLabel="Year" :yLabel="`Elevation (${unit})`"
              :xFormat="(v) => String(Math.round(v))" :yFormat="(v) => Math.round(v).toLocaleString()" />
          </ChartCard>
          <ChartCard>
            <BarChart title="Records per year" :data="effortByYear" :format="(v) => String(v)" />
            <p class="note">
              The control for both charts above. Recording effort has grown steeply, so a
              shift in either trend may be a shift in who is looking rather than in the
              mushrooms.
            </p>
          </ChartCard>
          <ChartCard>
            <BarChart title="Distinct species recorded per year" :data="speciesByYear" :format="(v) => String(v)" />
          </ChartCard>
        </section>

        <section v-else-if="analysisTab === 'phenology'" class="an-pane">
          <PhenologyPanel :features="filteredData?.features || []" />
        </section>

        <section v-else class="an-pane">
          <ChartCard class="wide">
            <h3 class="ct">Field coverage</h3>
            <table class="tbl">
              <thead><tr><th>Field</th><th class="num">Filled</th><th class="num">Coverage</th><th>&nbsp;</th></tr></thead>
              <tbody>
                <tr v-for="f in coverage.fields" :key="f.key">
                  <td>{{ f.label }}</td>
                  <td class="num">{{ f.filled.toLocaleString() }}</td>
                  <td class="num">{{ (f.pct * 100).toFixed(1) }}%</td>
                  <td class="barcell">
                    <span class="bar" :style="{ width: `${f.pct * 100}%`, background: covColor(f.pct) }"></span>
                  </td>
                </tr>
              </tbody>
            </table>
            <p class="note">
              A chart drawn from a 20%-covered column looks exactly as confident as one drawn
              from a full column. This is which is which. Anything thin here is thin because
              the enrichment pipeline has not reached those rows yet, re-running it fills them.
            </p>
          </ChartCard>
          <ChartCard class="wide">
            <BarChart title="Enrichment coverage by year (mean across fields)"
              :data="coverageByYear" :format="(v) => `${v}%`" />
            <p class="note">
              Uneven coverage across years makes year-over-year comparison unreliable for the
              thin fields: a trend can be a change in what was measured rather than in what
              was there.
            </p>
          </ChartCard>
        </section>
      </template>
    </template>

    <template v-else-if="tab === 'gallery'">
    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <template v-else>
      <!-- Layout controls: reorder / hide the preset charts -->
      <div class="layout-bar">
        <button class="lb-btn" :class="{ on: layout.editing.value }"
                :title="tip('Reorder or hide the preset charts', 'r')"
                @click="layout.editing.value = !layout.editing.value">
          {{ layout.editing.value ? '✓ Done arranging' : '⇅ Arrange charts' }}
        </button>
        <HelpLink option="chart-arrange" keys="r" />
        <span v-if="layout.editing.value" class="lb-hint">Drag a card to move it, or use ‹ › and ✕.</span>
        <!-- Cards register themselves as they render, which happens after this
             bar is serialised on the server. Render the tally on the client
             only, so SSR's "0 shown" never mismatches the real count. -->
        <ClientOnly>
          <span class="lb-count">{{ layout.visibleCount.value }} shown</span>
        </ClientOnly>
        <button v-if="layout.hiddenCharts.value.length" class="lb-btn ghost"
                title="Bring every hidden chart back" @click="layout.showAll()">
          Show all ({{ layout.hiddenCharts.value.length }} hidden)
        </button>
        <button v-if="layout.editing.value" class="lb-btn ghost"
                title="Restore the original order and unhide everything"
                @click="layout.reset()">Reset layout</button>
        <AppearanceControls field="species" field-label="Species" :values="speciesValues" />
        <HelpLink option="appearance-palette" />
        <ShareMenu :title="shareTitle" />
        <HelpLink option="share-link" />
      </div>

      <div v-if="layout.editing.value && layout.hiddenCharts.value.length" class="hidden-bar">
        <span class="hb-label">Hidden:</span>
        <button v-for="h in layout.hiddenCharts.value" :key="h.id" class="hb-chip" :title="`Show '${h.title}'`"
                @click="layout.show(h.id)">
          {{ h.title }} <span class="plus">+</span>
        </button>
      </div>

      <!-- Saved custom charts (from the Build tab), reorderable by dragging.
           Positioned with CSS `order` rather than by reordering the array, so a
           drag reflows the grid without re-mounting a chart — the array is
           written once, on drop. -->
      <section v-if="saved.charts.value.length" class="saved">
        <h2 class="saved-title">
          My charts <HelpLink option="chart-edit" />
          <span class="saved-hint">Drag a card to rearrange</span>
        </h2>
        <div class="grid">
          <ChartCard
            v-for="(chart, i) in saved.charts.value"
            :key="chart.id"
            class="saved-card"
            :class="{ dragging: savedDrag.dragging.value === chart.id,
                      'drop-target': savedDrag.over.value === chart.id
                        && savedDrag.dragging.value && savedDrag.over.value !== savedDrag.dragging.value }"
            :style="{ order: savedDrag.orderOf(chart.id) }"
            draggable="true"
            @dragstart="savedDrag.start(chart.id, $event)"
            @dragenter.prevent="savedDrag.enter(chart.id)"
            @dragover.prevent
            @drop.prevent="savedDrag.end()"
            @dragend="savedDrag.end()"
          >
            <div class="saved-tools">
              <span class="grip" title="Drag to rearrange" aria-hidden="true">⠿</span>
              <button title="Move left" :disabled="i === 0" @click="saved.move(chart.id, -1)">‹</button>
              <button title="Move right" :disabled="i === saved.charts.value.length - 1" @click="saved.move(chart.id, 1)">›</button>
              <button :title="`Open '${chartName(chart)}' in the chart builder`"
                      :aria-label="`Edit ${chartName(chart)}`" @click="editChart(chart.id)">✎</button>
              <ShareMenu compact :title="chartName(chart)" path="/charts"
                         :extra="{ tab: 'build', cfg: encodeChartConfig(chart) }"
                         note="This link opens this chart, over the same filtered data." />
              <button title="Remove" class="rm" @click="saved.remove(chart.id)">✕</button>
            </div>
            <!-- Off-screen charts are built as they scroll into view, so a
                 gallery of many does not render them all at once (V12-PERF-3). -->
            <LazyVisible min-height="260px">
              <ChartRenderer :config="chart" @select="selected = $event" />
            </LazyVisible>
          </ChartCard>
        </div>
      </section>

    <div class="grid">
      <GalleryChart id="clusters">
        <BarChart title="Observations per environmental cluster" :data="clusterData" :format="int" />
        <p class="note">Colors match the map. "Unclustered" = missing every clustering feature.</p>
      </GalleryChart>

      <GalleryChart id="rain-leadup" v-if="hasRainData">
        <BarChart title="Avg. rain in the 7 days before an observation" :data="rainLeadUp" :format="mm" />
        <p class="note">Mean daily precipitation (mm) across all observations, by days before the find.</p>
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
          xKey="day_of_year" yKey="elevation"
          xLabel="Day of year" :yLabel="`Elevation (${unit})`"
          :xFormat="(v) => Math.round(v)" :yFormat="(v) => Math.round(v).toLocaleString()"
          @select="selected = $event" />
        <p class="note">Each point is one observation, colored by cluster: seasonal timing across elevation.</p>
      </GalleryChart>

      <GalleryChart id="elev-vs-temp" v-if="elevVsTemp.length">
        <ScatterChart title="Elevation vs. observation-day high temp" :data="elevVsTemp" :legend="clusterLegend"
          xKey="tmax" yKey="elevation"
          :xLabel="`High temp (°${tempUnit})`" :yLabel="`Elevation (${unit})`"
          :xFormat="(v) => `${Math.round(v)}°`" :yFormat="(v) => Math.round(v).toLocaleString()"
          @select="selected = $event" />
        <p class="note">
          These rise together (ρ +0.43), which is not altitude warming anything: high
          finds happen in summer and low ones in spring and autumn. Hold the season
          still and the relationship flattens to about zero.
        </p>
      </GalleryChart>

      <GalleryChart id="rain-vs-doy" v-if="rainVsDoy.length">
        <ScatterChart title="7-day rain total vs. day of year" :data="rainVsDoy" :legend="clusterLegend"
          xKey="day_of_year" yKey="rain7"
          xLabel="Day of year" yLabel="Rain total (mm)"
          :xFormat="(v) => Math.round(v)" :yFormat="(v) => Math.round(v)"
          @select="selected = $event" />
        <p class="note">Total precipitation in the 7 days before each find.</p>
      </GalleryChart>

      <GalleryChart id="phenology" v-if="phenologyBySpecies.length">
        <BoxPlot title="Fruiting season by species" :data="phenologyBySpecies" xLabel="Day of year" valueKey="day_of_year"
          :format="(v) => Math.round(v)" />
        <p class="note">When each species is found through the year (top {{ TOP_SPECIES_BOX }} by count, ≥3 obs each).</p>
      </GalleryChart>

      <GalleryChart id="elevation-by-species" v-if="elevationBySpecies.length">
        <BoxPlot :title="`Elevation range by species (${unit})`" :data="elevationBySpecies" :xLabel="`Elevation (${unit})`" valueKey="elevation"
          :format="(v) => Math.round(v).toLocaleString()" />
        <p class="note">Elevation band each species prefers (top {{ TOP_SPECIES_BOX }} by count, ≥3 obs each).</p>
      </GalleryChart>

      <GalleryChart id="cluster-profile" v-if="clusterProfile.rows.length">
        <HeatmapChart title="Environmental cluster profiles" :rows="clusterProfile.rows"
          :cols="clusterProfile.cols" :matrix="clusterProfile.matrix" :format="(v) => v.toFixed(2)" />
        <p class="note">Mean of each feature per cluster, scaled 0–1 across clusters: what defines each group.</p>
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

      <GalleryChart id="aspect" v-if="aspectValues.length">
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
import { hasValue, useObservations } from '~/composables/useObservations'
import { useDatasets } from '~/composables/useDatasets'
import { PALETTE, UNCLUSTERED, categoryColor, colorFor } from '~/composables/useAppearance'
import { useUnits } from '~/composables/useUnits'
import { useSavedCharts } from '~/composables/useSavedCharts'
import { describeChart, encodeChartConfig } from '~/composables/chartConfig'
import { ALL_CATEGORY, ALL_NUMERIC } from '~/composables/useChartFields'

// ── Analysis ────────────────────────────────────────────────────────────────
const ANALYSIS_TABS = [
  { key: 'drivers', label: 'What relates to what', hint: 'Which environmental variables move together' },
  { key: 'species', label: 'Species', hint: 'What each species prefers, and which are found together' },
  { key: 'seasons', label: 'Year over year', hint: 'Season timing and elevation by year, against recording effort' },
  { key: 'phenology', label: 'What moves the season', hint: 'Whether a species is fruiting earlier or later, and whether rain or warmth explains it' },
  { key: 'quality', label: 'Data quality', hint: 'How much of each field is actually filled in' },
]
const MIN_SPECIES_OBS = 20
const analysisTab = ref('drivers')
const ANALYSIS_TAB_DOCS = {
  drivers: 'analysis-drivers', species: 'analysis-species',
  seasons: 'analysis-seasons', quality: 'analysis-quality',
  phenology: 'analysis-phenology',
}
const analysisTabDocId = computed(() => ANALYSIS_TAB_DOCS[analysisTab.value] || 'analysis-drivers')
const analysisShareTitle = computed(() =>
  `Analysis of ${rows.value.length.toLocaleString()} mushroom observations`)

// Two tabs on this page: the preset chart gallery and the chart builder.
// Tab lives in the URL query so /charts?tab=build deep-links (and the old
// /explore route redirects here).
const route = useRoute()
const router = useRouter()
const tab = computed({
  get: () => (['build', 'analysis'].includes(route.query.tab) ? route.query.tab : 'gallery'),
  set: (v) => router.replace({ query: { ...route.query, tab: v } }),
})

const saved = useSavedCharts()
const layout = useChartLayout()

// Dragging one saved card onto another. Redraws are suspended for the duration:
// changing CSS order reflows the grid, which resizes every chart container and
// fires every ResizeObserver, so a pointer move would otherwise recompute every
// chart on the page. See composables/useRenderPause.js.
const { pauseRendering, resumeRendering } = useRenderPause()
const savedDrag = useDragReorder({
  key: 'saved-charts',
  ids: () => saved.charts.value.map((c) => c.id),
  onCommit: (next) => saved.setOrder(next),
  onPause: pauseRendering,
  onResume: resumeRendering,
})
// Escape abandons the drag; without this the pause would outlive it and leave
// the charts frozen at the size they had when it started.
function onDragKey(e) { if (e.key === 'Escape' && savedDrag.dragging.value) savedDrag.cancel() }
onMounted(() => window.addEventListener('keydown', onDragKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onDragKey))
const { rows, error, pending, load, addInlineDataset } = useObservations()
const datasetsApi = useDatasets()
const { unit, elevValue, tempUnit, tempValue } = useUnits()
const appearance = useAppearance()
const share = useShareState()
const shortcuts = useShortcuts()
const tip = (text, keys) => shortcuts.withKey(text, keys)

shortcuts.register([
  { scope: 'Charts', keys: '1', label: 'Chart gallery', run: () => { tab.value = 'gallery' } },
  { scope: 'Charts', keys: '2', label: 'Chart builder', run: () => { tab.value = 'build' } },
  { scope: 'Charts', keys: '3', label: 'Analysis', run: () => { tab.value = 'analysis' } },
  { scope: 'Charts', keys: 'r', label: 'Arrange charts', run: () => { layout.editing.value = !layout.editing.value } },
  { scope: 'Charts', keys: 'escape', label: 'Close the observation drawer', run: () => { selected.value = null } },
])
const shareTitle = computed(() =>
  `${rows.value.length.toLocaleString()} mushroom observations, charts`)

// A saved chart has no title of its own, so name it from what it plots — the
// share text and the tooltips both need something better than "chart".
const fieldLabel = (key) => (
  [...ALL_NUMERIC, ...ALL_CATEGORY].find((f) => f.key === key)?.label || key
)
const chartName = (chart) => describeChart(chart, fieldLabel)

/** Open a saved chart in the builder, editing that chart rather than a copy. */
function editChart(id) {
  router.push({ path: '/charts', query: { ...route.query, tab: 'build', edit: id, cfg: undefined } })
}
// Restore filters/palette from a shared link before the charts compute.
onMounted(() => share.apply(useRoute().query))
onMounted(async () => {
  const slug = route.query.dataset
  if (slug && typeof slug === 'string') {
    try {
      const ds = await datasetsApi.activate(slug)
      if (ds) {
        const geojson = await datasetsApi.loadActiveGeojson()
        if (geojson) {
          addInlineDataset(
            { id: `dataset-${ds.id}`, label: ds.title, path: `mem:dataset-${ds.id}` },
            geojson,
          )
        }
      }
    } catch { /* fall through — charts still work with whatever was loaded */ }
  }
})
onMounted(() => {
  load(); saved.loadFromStorage(); layout.loadFromStorage(); appearance.loadFromStorage()
})

// Species present, most common first — what the appearance panel offers for
// per-value recoloring. Species is the dimension worth pinning: it carries the
// same color across the map and every chart.
const speciesValues = computed(() =>
  [...countBy(rows.value, (r) => r.species).entries()]
    .sort((a, b) => b[1] - a[1]).map(([v]) => v))

const int = (v) => String(v)
const mm = (v) => `${v}`
const deg = (v) => `${v}°`

const hasDayTemp = computed(() => rows.value.some((r) => hasValue(r.tmax) || hasValue(r.tmin)))
const hasTempHistory = computed(() => rows.value.some((r) => hasValue(r.tmax_d0)))
const hasRainData = computed(() => rows.value.some((r) => hasValue(r.prcp_d0)))

// ── Scatter plots (per-observation granularity, colored by cluster) ──────────
const ptColor = (r) => (hasValue(r.cluster) ? colorFor(r.cluster) : UNCLUSTERED)
const clusterLegend = computed(() => {
  const seen = new Set()
  let hasNull = false
  for (const r of rows.value) { if (hasValue(r.cluster)) seen.add(r.cluster); else hasNull = true }
  const out = [...seen].sort((a, b) => a - b).map((c) => ({ label: `C${c}`, color: colorFor(c) }))
  if (hasNull) out.push({ label: 'Unclustered', color: UNCLUSTERED })
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
const TOP_SPECIES_BOX = 25

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
    .slice(0, TOP_SPECIES_BOX)
    .map(([label, values]) => ({ label, values, color: categoryColor('species', label) }))
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
  const domMin = Math.floor(Math.min(...combined) / step) * step
  const domMax = Math.ceil(Math.max(...combined) / step) * step
  const binCount = Math.round((domMax - domMin) / step)
  const highCounts = new Array(binCount).fill(0)
  const lowCounts = new Array(binCount).fill(0)
  for (const v of highVals) {
    const i = Math.floor((v - domMin) / step)
    if (i >= 0 && i < binCount) highCounts[i] += 1
  }
  for (const v of lowVals) {
    const i = Math.floor((v - domMin) / step)
    if (i >= 0 && i < binCount) lowCounts[i] += 1
  }
  const bins = []
  for (let bi = 0; bi < binCount; bi++) {
    const lo = domMin + bi * step
    bins.push({ label: `Low ${lo}–${lo + step}°${tempUnit.value}`, short: `L ${lo}`, value: lowCounts[bi], color: '#1baf7a' })
    bins.push({ label: `High ${lo}–${lo + step}°${tempUnit.value}`, short: `H ${lo}`, value: highCounts[bi], color: '#2a78d6' })
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
  if (unclustered) out.push({ label: 'Unclustered', short: 'uncl.', value: unclustered, color: UNCLUSTERED })
  return out
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
  const domMin = Math.floor(Math.min(...vals) / step) * step
  const domMax = Math.ceil(Math.max(...vals) / step) * step
  const binCount = Math.round((domMax - domMin) / step)
  const counts = new Array(binCount).fill(0)
  for (const v of vals) {
    const i = Math.floor((v - domMin) / step)
    if (i >= 0 && i < binCount) counts[i] += 1
  }
  return counts.map((n, bi) => {
    const lo = domMin + bi * step
    return {
      label: `${lo.toLocaleString()}–${(lo + step).toLocaleString()} ${unit.value}`,
      short: `${(lo / 1000)}k`,
      value: n,
    }
  })
})

const landCoverData = computed(() => {
  const counts = countBy(rows.value, (r) => r.land_cover_label)
  return [...counts.entries()].sort((a, b) => b[1] - a[1])
    .map(([label, n]) => ({ label, value: n }))
})

const speciesData = computed(() => {
  const counts = countBy(rows.value, (r) => r.species)
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8)
    .map(([label, n]) => ({ label, short: label, value: n, color: categoryColor('species', label) }))
})

// ── Analysis computed ────────────────────────────────────────────────────────
const { filteredData } = useObservations()
const analysis = useAnalysis()

const corr = computed(() => analysis.correlationMatrix.value)
const corrLabels = computed(() => corr.value.fields.map((f) => f.label))
const rho = (v) => (v === null || v === undefined ? '' : Number(v).toFixed(2))
const rhoColor = (v) => (Math.abs(v) >= 0.5 ? 'var(--accent)' : 'var(--text)')

function strength(v) {
  const a = Math.abs(v)
  const dir = v > 0 ? 'rise together' : 'move opposite'
  if (a >= 0.7) return `strong, ${dir}`
  if (a >= 0.4) return `moderate, ${dir}`
  if (a >= 0.2) return `weak: ${dir}`
  return 'little or none'
}

const profiles = computed(() => {
  const p = analysis.speciesProfiles(MIN_SPECIES_OBS, 25)
  return { ...p, matrix: p.species.map((s) => s.z) }
})
const profileRows = computed(() => profiles.value.species.map((s) => `${s.species} (${s.n})`))
const profileCols = computed(() => profiles.value.fields.map((f) => f.label))
const z = (v) => (v === null || v === undefined ? '' : `${v > 0 ? '+' : ''}${Number(v).toFixed(1)}`)

const pairs = computed(() => analysis.coOccurrence({ minObs: MIN_SPECIES_OBS, limit: 15 }))

const years = computed(() => analysis.byYear.value)
const seasonTiming = computed(() =>
  years.value.filter((y) => y.medianDoy !== null).map((y) => ({ x: y.year, y: y.medianDoy })))
const elevationTrend = computed(() =>
  years.value.filter((y) => y.medianElevation !== null)
    .map((y) => ({ x: y.year, y: elevValue(y.medianElevation) })))
const effortByYear = computed(() =>
  years.value.map((y) => ({ label: String(y.year), short: String(y.year), value: y.n })))
const speciesByYear = computed(() =>
  years.value.map((y) => ({ label: String(y.year), short: String(y.year), value: y.species })))

const coverage = computed(() => analysis.coverage.value)
const covColor = (pct) => (pct >= 0.8 ? 'var(--accent)' : pct >= 0.4 ? '#eda100' : 'var(--danger)')
const coverageByYear = computed(() => coverage.value.years.map((y) => {
  const vals = Object.values(y.pct)
  const mean = vals.length ? vals.reduce((s, v) => s + v, 0) / vals.length : 0
  return { label: String(y.year), short: String(y.year), value: Math.round(mean * 100) }
}))
</script>

<style scoped>
/* ── Dragging saved cards ─────────────────────────────────────────────────
   The grip advertises the gesture; the ‹ › buttons stay because HTML
   drag-and-drop never fires on touch and a keyboard has no drag at all. */
.saved-hint { font-size: 0.74rem; font-weight: 400; color: var(--muted); margin-left: 8px; }
.saved-card { cursor: grab; }
.saved-card.dragging { opacity: 0.4; cursor: grabbing; }
.saved-card.drop-target { outline: 2px solid var(--accent); outline-offset: 2px; }
.saved-tools .grip {
  color: var(--muted); font-size: 0.9rem; line-height: 1; padding: 0 4px;
  cursor: grab; user-select: none;
}

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

/* ── Analysis tab ──────────────────────────────────────────────────────── */
.analysis-tabs {
  display: flex; align-items: center; gap: 6px; margin-bottom: 12px; flex-wrap: wrap;
  border-bottom: 1px solid var(--border); padding-bottom: 10px;
}
.analysis-tabs button {
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  border-radius: 6px; padding: 6px 13px; font-size: 0.86rem; font-weight: 600; cursor: pointer;
}
.analysis-tabs button:hover { background: var(--surface-2); }
.analysis-tabs button.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.tab-share { margin-left: auto; }
.scope { color: var(--muted); font-size: 0.82rem; margin: 0 0 12px; }
.an-pane { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.an-pane :deep(.wide) { grid-column: 1 / -1; }
.an-pane :deep(.card) { height: auto; min-height: 340px; }
.ct { font-size: 0.95rem; font-weight: 600; color: var(--text); margin: 0 0 8px; }
.tbl { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
.tbl th {
  text-align: left; color: var(--muted); font-weight: 600; padding: 4px 8px;
  border-bottom: 1px solid var(--border);
}
.tbl td { padding: 4px 8px; border-bottom: 1px solid var(--border-soft); }
.tbl .num { text-align: right; font-variant-numeric: tabular-nums; }
.tbl .muted { color: var(--muted); }
.barcell { width: 34%; }
.bar { display: block; height: 8px; border-radius: 4px; }

.saved { margin-bottom: 22px; }
.saved-title { margin: 0 0 10px; font-size: 1rem; color: var(--text); }
/* Above the neighbouring cards, not just above this one: the share panel drops
   out of the card and would otherwise slide behind the card to its right. */
.saved-tools { position: absolute; top: 8px; right: 34px; display: flex; gap: 2px; z-index: 400; }
.saved-tools :deep(.share) { display: flex; }
.saved-tools button {
  border: 1px solid var(--border); background: var(--surface); color: var(--muted); cursor: pointer;
  width: 22px; height: 22px; border-radius: 5px; font-size: 0.85rem; line-height: 1; padding: 0;
}
.saved-tools button:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.saved-tools button:disabled { opacity: 0.35; cursor: default; }
.saved-tools .rm:hover { background: #fdecec; color: #b00020; border-color: #f5c2c2; }
</style>
