<template>
  <div class="explore">
    <div class="panel">
      <label class="ctrl">
        <span>Chart <HelpLink :option="typeDocId" /></span>
        <select v-model="chartType">
          <option value="scatter">Scatter</option>
          <option value="bar">Bar (aggregate)</option>
          <option value="stacked">Stacked bar</option>
          <option value="line">Line</option>
          <option value="area">Area</option>
          <option value="box">Box plot by category</option>
          <option value="histogram">Histogram</option>
          <option value="heatmap">Heatmap</option>
          <option value="radar">Radar</option>
          <option value="donut">Donut</option>
        </select>
      </label>

      <template v-if="chartType === 'scatter'">
        <label class="ctrl"><span>X <HelpLink option="chart-x" /></span><select v-model="xField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Y <HelpLink option="chart-y" /></span><select v-model="yField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Colour <HelpLink option="chart-color-field" /></span><select v-model="colorField"><option value="">(none)</option><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Shape <HelpLink option="chart-shape-field" /></span><select v-model="shapeField"><option value="">(none)</option><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Size <HelpLink option="chart-size-field" /></span><select v-model="sizeField"><option value="">(none)</option><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label v-if="xField === 'day_of_year'" class="ctrl chk"><input type="checkbox" v-model="showToday" /> Today line <HelpLink option="chart-today" /></label>
      </template>

      <template v-else-if="chartType === 'bar'">
        <label class="ctrl"><span>Group by <HelpLink option="chart-group-by" /></span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure <HelpLink option="chart-measure" /></span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
        <label class="ctrl chk"><input type="checkbox" v-model="horizontal" /> Horizontal <HelpLink option="chart-horizontal" /></label>
      </template>

      <template v-else-if="chartType === 'line' || chartType === 'area'">
        <label class="ctrl"><span>X <HelpLink option="chart-x" /></span><select v-model="xField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Y (mean) <HelpLink option="chart-y" /></span><select v-model="yField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Series <HelpLink option="chart-series" /></span><select v-model="seriesField"><option value="">(one line)</option><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Granularity <HelpLink option="chart-granularity" /></span><input type="range" min="4" max="60" v-model.number="granularity" /><span class="gval">{{ granularity }}</span></label>
        <label v-if="xField === 'day_of_year'" class="ctrl chk"><input type="checkbox" v-model="showToday" /> Today line <HelpLink option="chart-today" /></label>
      </template>

      <template v-else-if="chartType === 'stacked'">
        <label class="ctrl"><span>Group by <HelpLink option="chart-group-by" /></span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Split by <HelpLink option="chart-stack-field" /></span><select v-model="stackField"><option value="">(pick one)</option><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure <HelpLink option="chart-measure" /></span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
        <label class="ctrl chk"><input type="checkbox" v-model="horizontal" /> Horizontal <HelpLink option="chart-horizontal" /></label>
        <label class="ctrl chk"><input type="checkbox" v-model="normalise" /> 100% <HelpLink option="chart-normalise" /></label>
      </template>

      <template v-else-if="chartType === 'box'">
        <label class="ctrl"><span>Group by <HelpLink option="chart-group-by" /></span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Value <HelpLink option="chart-value" /></span><select v-model="valueField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
      </template>

      <template v-else-if="chartType === 'histogram'">
        <label class="ctrl"><span>Value <HelpLink option="chart-value" /></span><select v-model="valueField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Bins <HelpLink option="chart-bins" /></span><input type="number" min="4" max="30" v-model.number="bins" /></label>
      </template>

      <template v-else-if="chartType === 'heatmap'">
        <label class="ctrl"><span>Rows <HelpLink option="chart-rows" /></span><select v-model="rowField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Columns <HelpLink option="chart-columns" /></span><select v-model="colField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure <HelpLink option="chart-measure" /></span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
      </template>

      <template v-else-if="chartType === 'radar' || chartType === 'donut'">
        <label class="ctrl"><span>Group by <HelpLink option="chart-group-by" /></span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure <HelpLink option="chart-measure" /></span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
      </template>

      <label v-if="SORTABLE_TYPES.has(chartType)" class="ctrl">
        <span>Sort <HelpLink option="chart-sort" /></span>
        <select v-model="sortBy">
          <option v-for="m in SORT_MODES" :key="m.key" :value="m.key">{{ m.label }}</option>
        </select>
      </label>

      <LiveClusterControls class="lc-item" />
      <div class="actions">
        <ShareMenu :title="shareTitle" :extra="shareExtra" path="/charts"
                   note="This link opens this chart, over the same filtered data." />
        <HelpLink option="chart-share" />
        <button v-if="editingId" class="save alt" :disabled="!!justSaved"
                title="Keep the original and save this as a second chart"
                @click="saveAsNew">
          {{ justSaved === 'added' ? '✓ Saved a copy' : 'Save as new' }}
        </button>
        <button class="save" :disabled="!!justSaved"
                :title="editingId ? 'Write these changes back to the saved chart' : 'Add this chart to the Charts page'"
                @click="save">
          {{ saveLabel }}
        </button>
        <HelpLink :option="editingId ? 'chart-edit' : 'chart-save'" />
      </div>
    </div>

    <div v-if="editingId" class="editing">
      <span>Editing <strong>{{ editingTitle || 'a saved chart' }}</strong>, saving updates it in place.</span>
      <button class="ed-stop" title="Leave the saved chart alone and build a new one"
              @click="stopEditing">Stop editing</button>
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>
    <ChartCard v-else class="stage">
      <ChartRenderer :config="config" @select="selected = $event" />
    </ChartCard>

    <ObservationDrawer :selected="selected" @close="selected = null" />
  </div>
</template>

<script setup>
import { hasValue, useObservations } from '~/composables/useObservations'
import { ALL_NUMERIC, ALL_CATEGORY, SORT_MODES } from '~/composables/useChartFields'
import { useSavedCharts } from '~/composables/useSavedCharts'
import {
  chartConfigOf, decodeChartConfig, defaultChartConfig, describeChart, encodeChartConfig,
} from '~/composables/chartConfig'

const { rows, error, pending, load } = useObservations()
const saved = useSavedCharts()
const route = useRoute()
const router = useRouter()
onMounted(load)

function rawNum(r, key) {
  if (key === 'rain7') return [0, 1, 2, 3, 4, 5, 6].some((o) => hasValue(r[`prcp_d${o}`])) ? 1 : null
  return hasValue(r[key]) ? Number(r[key]) : null
}
const live = useLiveClusters()
function catPresent(r, key) {
  if (key === 'cluster') return hasValue(r.cluster)
  if (key === 'live_cluster') return false // handled via `live.active` below
  return hasValue(r[key])
}
const numericFields = computed(() => ALL_NUMERIC.filter((f) => rows.value.some((r) => rawNum(r, f.key) !== null)))
const categoryFields = computed(() => ALL_CATEGORY.filter((f) => (
  f.key === 'live_cluster' ? live.active.value : rows.value.some((r) => catPresent(r, f.key))
)))

// Defaults come from the shared config table rather than being repeated here,
// so the builder, a shared link and a saved chart cannot disagree about what an
// unset field means.
const D = defaultChartConfig()
const chartType = ref(D.type)
const xField = ref(D.xField)
const yField = ref(D.yField)
const colorField = ref(D.colorField)
const shapeField = ref(D.shapeField)
const sizeField = ref(D.sizeField)
const seriesField = ref(D.seriesField)
const groupField = ref(D.groupField)
const valueField = ref(D.valueField)
const measure = ref(D.measure)
const rowField = ref(D.rowField)
const colField = ref(D.colField)
const bins = ref(D.bins)
const granularity = ref(D.granularity)
const horizontal = ref(D.horizontal)
const showToday = ref(D.showToday)
const sortBy = ref(D.sortBy)
const stackField = ref(D.stackField)
const normalise = ref(D.normalise)

// The ? beside the type picker documents the chart you actually have selected —
// what it is good for and where it misleads — rather than the idea of charts.
const TYPE_DOCS = {
  scatter: 'chart-scatter', bar: 'chart-bar', stacked: 'chart-stacked',
  line: 'chart-line', area: 'chart-line',
  box: 'chart-box', histogram: 'chart-histogram', heatmap: 'chart-heatmap',
  radar: 'chart-radar', donut: 'chart-donut',
}
const typeDocId = computed(() => TYPE_DOCS[chartType.value] || 'chart-type')

// Only charts that lay categories out in a row have an order worth choosing.
// A scatter or histogram has no category axis to sort.
const SORTABLE_TYPES = new Set(['bar', 'stacked', 'box', 'radar'])

const config = computed(() => ({
  type: chartType.value,
  xField: xField.value, yField: yField.value, colorField: colorField.value,
  shapeField: shapeField.value, sizeField: sizeField.value, seriesField: seriesField.value,
  groupField: groupField.value, valueField: valueField.value, measure: measure.value,
  rowField: rowField.value, colField: colField.value, bins: bins.value, granularity: granularity.value,
  horizontal: horizontal.value, showToday: showToday.value, sortBy: sortBy.value,
  stackField: stackField.value, normalise: normalise.value,
}))

// Click a scatter point to open its observation (iNat link + open on map).
const selected = ref(null)

// Remember the builder configuration per viewer, so returning to Explore keeps
// the last chart you were designing.
const EXPLORE_KEY = 'explore-config'
const persisted = { type: chartType, xField, yField, colorField, shapeField, sizeField, seriesField, groupField, valueField, measure, rowField, colField, bins, granularity, horizontal, showToday, sortBy, stackField, normalise }

function applyConfig(cfg) {
  for (const [k, r] of Object.entries(persisted)) if (cfg?.[k] !== undefined) r.value = cfg[k]
}

// The chart being edited in place, when the builder was opened from a saved
// chart's ✎ button. Saving then replaces that chart rather than leaving the
// original beside a near-identical copy.
const editingId = ref('')
const editingTitle = computed(() => {
  const chart = editingId.value ? saved.byId(editingId.value) : null
  return chart ? describeChart(chart, labelFor) : ''
})

// Three sources for what the builder opens with, most specific first: a chart
// configuration in the link (someone shared this chart), the saved chart being
// edited, then the viewer's own last session.
onMounted(() => {
  if (!import.meta.client) return
  saved.loadFromStorage()

  const { cfg, edit } = route.query
  if (typeof cfg === 'string' && cfg) {
    applyConfig(decodeChartConfig(cfg))
    return
  }
  if (typeof edit === 'string' && edit) {
    const chart = saved.byId(edit)
    if (chart) {
      editingId.value = edit
      applyConfig(chartConfigOf(chart))
      return
    }
    // The link named a chart this browser does not have — fall through to the
    // remembered session rather than opening a chart the viewer never built.
  }
  try {
    const last = JSON.parse(localStorage.getItem(EXPLORE_KEY) || 'null')
    if (last) applyConfig(last)
  } catch { /* ignore malformed storage */ }
})

watch(config, (cfg) => {
  if (import.meta.client) localStorage.setItem(EXPLORE_KEY, JSON.stringify(cfg))
}, { deep: true })

// Sharing a built chart carries the configuration alongside the filters, so the
// recipient opens this chart over this data rather than an empty builder.
const shareExtra = computed(() => {
  const encoded = encodeChartConfig(config.value)
  return { tab: 'build', ...(encoded ? { cfg: encoded } : {}) }
})
const labelFor = (key) => (
  [...ALL_NUMERIC, ...ALL_CATEGORY].find((f) => f.key === key)?.label || key
)
const shareTitle = computed(() => describeChart(config.value, labelFor))

const justSaved = ref('')
function save() {
  // An edit that cannot find its chart any more (removed in another tab) is
  // saved as a new one rather than silently discarded.
  if (editingId.value && saved.update(editingId.value, { ...config.value })) {
    justSaved.value = 'updated'
  } else {
    editingId.value = saved.add({ ...config.value })
    justSaved.value = 'added'
  }
  setTimeout(() => { justSaved.value = '' }, 1800)
}

function saveAsNew() {
  editingId.value = saved.add({ ...config.value })
  justSaved.value = 'added'
  // Keep editing the copy, not the original, so a second save does not write
  // back over the chart this one was branched from.
  router.replace({ query: { ...route.query, edit: editingId.value, cfg: undefined } })
  setTimeout(() => { justSaved.value = '' }, 1800)
}

const saveLabel = computed(() => {
  if (justSaved.value === 'updated') return '✓ Chart updated'
  if (justSaved.value === 'added') return '✓ Saved to Charts'
  return editingId.value ? '✓ Update chart' : '+ Save to Charts'
})

function stopEditing() {
  editingId.value = ''
  router.replace({ query: { ...route.query, edit: undefined, cfg: undefined } })
}
</script>

<style scoped>
.explore {
  padding: 16px 18px; display: flex; flex-direction: column; gap: 14px;
  height: 100%; min-height: 0; box-sizing: border-box;
}
.panel {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px 18px; align-items: end; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; width: 100%; box-sizing: border-box;
}
.ctrl { display: inline-flex; align-items: center; gap: 7px; font-size: 0.85rem; color: var(--text); }
.ctrl > span { color: var(--muted); font-weight: 600; }
.ctrl select, .ctrl input[type="number"] {
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 0.85rem; background: var(--surface);
}
.ctrl input[type="number"] { width: 60px; }
.ctrl input[type="range"] { width: 96px; accent-color: var(--accent); }
.ctrl .gval { color: var(--text); font-weight: 600; min-width: 1.4em; text-align: right; }
.ctrl.chk { gap: 5px; }
.actions {
  grid-column: 1 / -1; display: flex; align-items: center; justify-content: flex-end;
  gap: 8px; flex-wrap: wrap;
}
.save {
  border: 1px solid #2b7a3d; background: #2b7a3d; color: #fff;
  border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; font-weight: 600; cursor: pointer;
  min-width: 150px;
}
.save:hover { background: #246833; }
.save:disabled { opacity: 0.8; cursor: default; }
.save.alt {
  background: var(--surface); color: var(--text); border-color: var(--border); min-width: 0;
}
.save.alt:hover { background: var(--surface-2); }

.editing {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px;
  padding: 7px 12px; font-size: 0.82rem; color: var(--muted); flex: 0 0 auto;
}
.editing strong { color: var(--text); }
.ed-stop {
  margin-left: auto; border: 1px solid var(--border); background: var(--surface);
  color: var(--muted); border-radius: 6px; padding: 3px 10px; font-size: 0.78rem; cursor: pointer;
}
.ed-stop:hover { color: var(--text); background: var(--surface-3); }

/* The chart card fills the space left below the controls, and the SVG scales to
   fit that box (preserveAspectRatio letterboxes it) — so the chart always fits
   on screen without the page scrolling. */
.stage { flex: 1 1 auto; min-height: 0; min-width: 0; display: flex; flex-direction: column; }
.panel { flex: 0 0 auto; }
/* The card body is a plain block in the gallery, where every chart is clipped to
   a fixed card height. Here the chart is the whole point, so the body has to be
   a flex column for the figure's `flex: 1 1 auto` to mean anything. Without it
   the figure keeps its content height — the title and nothing else — and any
   chart sized as a percentage of its parent (scatter, line, area) resolves to
   zero and renders invisibly. Bar and friends only escaped that by setting a
   pixel height of their own. */
.stage :deep(.card-body) { display: flex; flex-direction: column; }
/* One chart type has a height the data decides: a box plot grows a row per
   category, so 40 species do not fit whatever the card is. Its own chart area
   scrolls, which needs a definite height to scroll within — which the flex
   column above now gives it. */
.stage :deep(figure) { flex: 1 1 auto; min-height: 0; min-width: 0; margin: 0; display: flex; flex-direction: column; }
.stage :deep(figure > div) { flex: 1 1 auto; min-height: 0; }
.stage :deep(svg) { width: 100%; height: 100%; max-height: 100%; display: block; }
.msg { padding: 16px; color: var(--muted); }
.msg.error { color: var(--danger); }
</style>
