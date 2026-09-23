<template>
  <div class="dash-widget saved-charts">
    <div class="widget-head">
      <h3 class="widget-title"><span aria-hidden="true">📊 </span>Saved charts</h3>
      <NuxtLink to="/charts" class="widget-link">Gallery ›</NuxtLink>
    </div>

    <p v-if="!list.length" class="widget-note">
      No saved charts yet. Build one on <NuxtLink to="/explore" class="widget-link">Explore</NuxtLink>.
    </p>

    <ul v-else class="chart-rows">
      <li v-for="c in list" :key="c.id" class="chart-row">
        <span class="chart-type">{{ typeLabel(c.type) }}</span>
        <NuxtLink :to="chartHref(c)" class="chart-name">{{ c.title || untitled(c) }}</NuxtLink>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useSavedCharts } from '~/composables/useSavedCharts'

const { charts, loadFromStorage } = useSavedCharts()

const list = computed(() => (charts.value || []).slice(0, 6))

const TYPE_LABELS = {
  scatter: 'Scatter', bar: 'Bar', box: 'Box', line: 'Line', heatmap: 'Heatmap', histogram: 'Histogram',
}
function typeLabel(t) { return TYPE_LABELS[t] || (t ? t[0].toUpperCase() + t.slice(1) : 'Chart') }
function untitled(c) { return `${typeLabel(c.type)} chart` }
function chartHref(c) {
  try { return `/explore?cfg=${encodeURIComponent(JSON.stringify(c))}` } catch { return '/charts' }
}

onMounted(() => { loadFromStorage() })
</script>

<style scoped>
.saved-charts { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }
.chart-rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }
.chart-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
.chart-type {
  flex: 0 0 auto; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted, #777); background: var(--surface-2, #f0f0f0); border-radius: 4px; padding: 1px 6px;
}
.chart-name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--accent, #2a78d6); text-decoration: none; }
.chart-name:hover { text-decoration: underline; }
</style>
