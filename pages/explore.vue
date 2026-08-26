<template>
  <div class="explore">
    <div class="panel">
      <label class="ctrl">
        <span>Chart</span>
        <select v-model="chartType">
          <option value="scatter">Scatter</option>
          <option value="bar">Bar (aggregate)</option>
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
        <label class="ctrl"><span>X</span><select v-model="xField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Y</span><select v-model="yField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Colour</span><select v-model="colorField"><option value="">— none —</option><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label v-if="xField === 'day_of_year'" class="ctrl chk"><input type="checkbox" v-model="showToday" /> Today line</label>
      </template>

      <template v-else-if="chartType === 'bar'">
        <label class="ctrl"><span>Group by</span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure</span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
        <label class="ctrl chk"><input type="checkbox" v-model="horizontal" /> Horizontal</label>
      </template>

      <template v-else-if="chartType === 'line' || chartType === 'area'">
        <label class="ctrl"><span>X</span><select v-model="xField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Y (mean)</span><select v-model="yField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
      </template>

      <template v-else-if="chartType === 'box'">
        <label class="ctrl"><span>Group by</span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Value</span><select v-model="valueField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
      </template>

      <template v-else-if="chartType === 'histogram'">
        <label class="ctrl"><span>Value</span><select v-model="valueField"><option v-for="f in numericFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Bins</span><input type="number" min="4" max="30" v-model.number="bins" /></label>
      </template>

      <template v-else-if="chartType === 'heatmap'">
        <label class="ctrl"><span>Rows</span><select v-model="rowField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Columns</span><select v-model="colField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure</span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
      </template>

      <template v-else-if="chartType === 'radar' || chartType === 'donut'">
        <label class="ctrl"><span>Group by</span><select v-model="groupField"><option v-for="f in categoryFields" :key="f.key" :value="f.key">{{ f.label }}</option></select></label>
        <label class="ctrl"><span>Measure</span><select v-model="measure"><option value="count">Count</option><option v-for="f in numericFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option></select></label>
      </template>

      <button class="save" :disabled="justSaved" @click="save">{{ justSaved ? '✓ Saved to Charts' : '+ Save to Charts' }}</button>
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
import { ALL_NUMERIC, ALL_CATEGORY } from '~/composables/useChartFields'
import { useSavedCharts } from '~/composables/useSavedCharts'

const { rows, error, pending, load } = useObservations()
const saved = useSavedCharts()
onMounted(load)

function rawNum(r, key) {
  if (key === 'rain7') return [0, 1, 2, 3, 4, 5, 6].some((o) => hasValue(r[`prcp_d${o}`])) ? 1 : null
  return hasValue(r[key]) ? Number(r[key]) : null
}
function catPresent(r, key) {
  if (key === 'cluster') return hasValue(r.cluster)
  return hasValue(r[key])
}
const numericFields = computed(() => ALL_NUMERIC.filter((f) => rows.value.some((r) => rawNum(r, f.key) !== null)))
const categoryFields = computed(() => ALL_CATEGORY.filter((f) => rows.value.some((r) => catPresent(r, f.key))))

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
const showToday = ref(false)

const config = computed(() => ({
  type: chartType.value,
  xField: xField.value, yField: yField.value, colorField: colorField.value,
  groupField: groupField.value, valueField: valueField.value, measure: measure.value,
  rowField: rowField.value, colField: colField.value, bins: bins.value, horizontal: horizontal.value,
  showToday: showToday.value,
}))

// Click a scatter point to open its observation (iNat link + open on map).
const selected = ref(null)

const justSaved = ref(false)
function save() {
  saved.add({ ...config.value })
  justSaved.value = true
  setTimeout(() => { justSaved.value = false }, 1800)
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
.ctrl.chk { gap: 5px; }
.save {
  justify-self: end; border: 1px solid #2b7a3d; background: #2b7a3d; color: #fff;
  border-radius: 6px; padding: 6px 12px; font-size: 0.85rem; font-weight: 600; cursor: pointer;
  min-width: 150px;
}
.save:hover { background: #246833; }
.save:disabled { opacity: 0.8; cursor: default; }

/* The chart card fills the space left below the controls, and the SVG scales to
   fit that box (preserveAspectRatio letterboxes it) — so the chart always fits
   on screen without the page scrolling. */
.stage { flex: 1 1 auto; min-height: 0; min-width: 0; display: flex; flex-direction: column; }
.panel { flex: 0 0 auto; }
.stage :deep(figure) { flex: 1 1 auto; min-height: 0; min-width: 0; margin: 0; display: flex; flex-direction: column; }
.stage :deep(figure > div) { flex: 1 1 auto; min-height: 0; }
.stage :deep(svg) { width: 100%; height: 100%; max-height: 100%; display: block; }
.msg { padding: 16px; color: var(--muted); }
.msg.error { color: var(--danger); }
</style>
