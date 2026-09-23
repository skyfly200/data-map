<template>
  <div class="dash-widget dash-chart">
    <!-- Header -->
    <div class="widget-head">
      <h3 class="widget-title" :title="chartTitle">{{ chartTitle }}</h3>
      <div class="widget-actions">
        <button class="icon-btn" :class="{ active: configuring }" title="Configure chart" @click="configuring = !configuring">⚙</button>
        <NuxtLink :to="exploreHref" class="widget-link" title="Open in Explore">↗</NuxtLink>
      </div>
    </div>

    <!-- Saved-chart picker (shown when no config set yet, or when configuring) -->
    <div v-if="configuring" class="cfg-panel">
      <!-- Pick a saved chart -->
      <div v-if="savedList.length" class="cfg-row">
        <label class="cfg-label">Use saved chart</label>
        <select class="cfg-sel" @change="loadSaved($event.target.value)">
          <option value="">— pick one —</option>
          <option v-for="c in savedList" :key="c.id" :value="c.id">{{ savedTitle(c) }}</option>
        </select>
      </div>

      <div class="cfg-divider" v-if="savedList.length">or build one</div>

      <!-- Chart type -->
      <div class="cfg-row">
        <label class="cfg-label">Type</label>
        <select class="cfg-sel" v-model="draft.type">
          <option value="scatter">Scatter</option>
          <option value="bar">Bar</option>
          <option value="stacked">Stacked bar</option>
          <option value="line">Line</option>
          <option value="area">Area</option>
          <option value="box">Box plot</option>
          <option value="histogram">Histogram</option>
          <option value="heatmap">Heatmap</option>
          <option value="radar">Radar</option>
          <option value="donut">Donut</option>
        </select>
      </div>

      <!-- Fields for scatter / line / area -->
      <template v-if="['scatter','line','area'].includes(draft.type)">
        <div class="cfg-row"><label class="cfg-label">X</label>
          <select class="cfg-sel" v-model="draft.xField">
            <option v-for="f in numFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div class="cfg-row"><label class="cfg-label">Y</label>
          <select class="cfg-sel" v-model="draft.yField">
            <option v-for="f in numFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div v-if="draft.type !== 'scatter'" class="cfg-row"><label class="cfg-label">Series</label>
          <select class="cfg-sel" v-model="draft.seriesField">
            <option value="">(one line)</option>
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div v-if="draft.type === 'scatter'" class="cfg-row"><label class="cfg-label">Color</label>
          <select class="cfg-sel" v-model="draft.colorField">
            <option value="">(none)</option>
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
      </template>

      <!-- Fields for bar / stacked / radar / donut -->
      <template v-else-if="['bar','stacked','radar','donut'].includes(draft.type)">
        <div class="cfg-row"><label class="cfg-label">Group by</label>
          <select class="cfg-sel" v-model="draft.groupField">
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div v-if="draft.type === 'stacked'" class="cfg-row"><label class="cfg-label">Split by</label>
          <select class="cfg-sel" v-model="draft.stackField">
            <option value="">(none)</option>
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div class="cfg-row"><label class="cfg-label">Measure</label>
          <select class="cfg-sel" v-model="draft.measure">
            <option value="count">Count</option>
            <option v-for="f in numFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option>
          </select>
        </div>
      </template>

      <!-- Fields for histogram / box -->
      <template v-else-if="['histogram','box'].includes(draft.type)">
        <div class="cfg-row"><label class="cfg-label">Value</label>
          <select class="cfg-sel" v-model="draft.valueField">
            <option v-for="f in numFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div v-if="draft.type === 'box'" class="cfg-row"><label class="cfg-label">Group by</label>
          <select class="cfg-sel" v-model="draft.groupField">
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div v-if="draft.type === 'histogram'" class="cfg-row"><label class="cfg-label">Bins</label>
          <input class="cfg-num" type="number" min="4" max="30" v-model.number="draft.bins" />
        </div>
      </template>

      <!-- Fields for heatmap -->
      <template v-else-if="draft.type === 'heatmap'">
        <div class="cfg-row"><label class="cfg-label">Rows</label>
          <select class="cfg-sel" v-model="draft.rowField">
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div class="cfg-row"><label class="cfg-label">Columns</label>
          <select class="cfg-sel" v-model="draft.colField">
            <option v-for="f in catFields" :key="f.key" :value="f.key">{{ f.label }}</option>
          </select>
        </div>
        <div class="cfg-row"><label class="cfg-label">Measure</label>
          <select class="cfg-sel" v-model="draft.measure">
            <option value="count">Count</option>
            <option v-for="f in numFields" :key="f.key" :value="f.key">Mean {{ f.label }}</option>
          </select>
        </div>
      </template>

      <div class="cfg-actions">
        <button class="cfg-apply" @click="applyDraft">Apply</button>
        <button class="cfg-cancel" @click="configuring = false">Cancel</button>
      </div>
    </div>

    <!-- Chart -->
    <div v-if="!configuring && hasConfig" class="chart-wrap">
      <p v-if="loading && !rows.length" class="ot-note">Loading…</p>
      <ChartRenderer v-else :config="activeConfig" :compact="true" />
    </div>

    <!-- Empty state -->
    <div v-if="!configuring && !hasConfig" class="chart-empty">
      <p>No chart configured.</p>
      <button class="cfg-apply" @click="configuring = true">Set up chart</button>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useSavedCharts } from '~/composables/useSavedCharts'
import { useDashboardState } from '~/composables/useDashboardState'
import { ALL_NUMERIC, ALL_CATEGORY } from '~/composables/useChartFields'
import { defaultChartConfig, describeChart, encodeChartConfig } from '~/composables/chartConfig'
import ChartRenderer from '~/components/ChartRenderer.vue'

const props = defineProps({
  widget: { type: Object, default: () => ({}) },
  isEditing: { type: Boolean, default: false },
})

const { rows, load, pending: loading } = useObservations()
const { charts, loadFromStorage } = useSavedCharts()
const { updateSettings } = useDashboardState()

const configuring = ref(false)

// Persisted config lives in widget.settings; we use a local reactive draft while editing.
const savedConfig = computed(() => props.widget?.settings?.chartConfig || null)
const hasConfig = computed(() => !!savedConfig.value)

const activeConfig = computed(() => savedConfig.value || defaultChartConfig())

const draft = reactive({ ...defaultChartConfig() })

function resetDraft() {
  const src = savedConfig.value || defaultChartConfig()
  Object.assign(draft, src)
}

function applyDraft() {
  if (props.widget?.id) {
    updateSettings(props.widget.id, { chartConfig: { ...draft } })
  }
  configuring.value = false
}

// Available fields (only those actually present in data)
const numFields = computed(() => ALL_NUMERIC.filter((f) =>
  rows.value.some((r) => r[f.key] != null && r[f.key] !== '')
))
const catFields = computed(() => ALL_CATEGORY.filter((f) =>
  rows.value.some((r) => r[f.key] != null && r[f.key] !== '')
))

const savedList = computed(() => charts.value || [])

function savedTitle(c) {
  return c.title || describeChart(c, (k) => ALL_NUMERIC.find((f) => f.key === k)?.label || k)
}

function loadSaved(id) {
  if (!id) return
  const c = charts.value.find((x) => x.id === id)
  if (!c) return
  Object.assign(draft, c)
}

const chartTitle = computed(() => {
  if (!hasConfig.value) return '📊 Chart'
  const t = describeChart(activeConfig.value, (k) => ALL_NUMERIC.find((f) => f.key === k)?.label || k)
  return `📊 ${t}`
})

const exploreHref = computed(() => {
  if (!hasConfig.value) return '/explore'
  try {
    return `/explore?cfg=${encodeURIComponent(encodeChartConfig(activeConfig.value))}`
  } catch {
    return '/explore'
  }
})

onMounted(() => {
  loadFromStorage()
  load()
  resetDraft()
  if (!hasConfig.value) configuring.value = true
})
</script>

<style scoped>
.dash-chart { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.4rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 70%; }
.widget-actions { display: flex; align-items: center; gap: 0.4rem; flex-shrink: 0; }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.icon-btn {
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff);
  color: var(--muted, #888); border-radius: 6px; font-size: 0.8rem;
  padding: 0.1rem 0.4rem; cursor: pointer; line-height: 1.4;
}
.icon-btn.active { background: var(--surface-2, #f0f0f0); color: var(--text, #333); }

.cfg-panel {
  display: flex; flex-direction: column; gap: 0.35rem;
  background: var(--surface-2, #f6f6f6); border: 1px solid var(--border-soft, #eee);
  border-radius: 8px; padding: 0.6rem 0.7rem;
}
.cfg-row { display: flex; align-items: center; gap: 0.4rem; }
.cfg-label { font-size: 0.68rem; color: var(--muted, #888); min-width: 4.5rem; flex-shrink: 0; }
.cfg-sel {
  flex: 1; font: inherit; font-size: 0.75rem; border: 1px solid var(--border, #ddd);
  border-radius: 5px; padding: 0.18rem 0.35rem; background: var(--surface, #fff);
  color: var(--text, #222); min-width: 0;
}
.cfg-num {
  width: 4rem; font: inherit; font-size: 0.75rem; border: 1px solid var(--border, #ddd);
  border-radius: 5px; padding: 0.18rem 0.35rem; background: var(--surface, #fff);
  color: var(--text, #222);
}
.cfg-divider {
  font-size: 0.65rem; color: var(--muted, #aaa); text-align: center;
  border-top: 1px solid var(--border-soft, #eee); padding-top: 0.35rem; margin-top: 0.1rem;
}
.cfg-actions { display: flex; gap: 0.4rem; margin-top: 0.2rem; }
.cfg-apply {
  font: inherit; font-size: 0.78rem; border: 1px solid var(--accent, #2a78d6);
  background: var(--accent, #2a78d6); color: #fff; border-radius: 6px;
  padding: 0.28rem 0.7rem; cursor: pointer;
}
.cfg-apply:hover { filter: brightness(1.08); }
.cfg-cancel {
  font: inherit; font-size: 0.78rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #777); border-radius: 6px;
  padding: 0.28rem 0.7rem; cursor: pointer;
}

.chart-wrap { flex: 1; min-height: 180px; overflow: hidden; }
.ot-note { font-size: 0.85rem; color: var(--muted, #888); }

.chart-empty {
  flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; gap: 0.6rem; color: var(--muted, #888); font-size: 0.85rem;
}
</style>
