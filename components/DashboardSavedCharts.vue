<template>
  <div class="dashboard-saved-charts">
    <h3 class="widget-title">📊 Saved Charts</h3>
    <div v-if="charts.length === 0" class="no-charts">
    <div class="widget-head">
      <h3 class="widget-title">📊 Saved Charts</h3>
      <NuxtLink v-if="chartList.length" to="/charts" class="widget-link">All charts ›</NuxtLink>
    </div>
    <div v-if="chartList.length === 0" class="no-charts">
      <p>No saved charts yet.</p>
      <NuxtLink to="/charts" class="create-chart-link">Create your first chart</NuxtLink>
      <NuxtLink to="/charts" class="create-chart-link">Build your first chart</NuxtLink>
    </div>
    <div v-else class="charts-list">
      <div v-for="chart in charts" :key="chart.id" class="chart-item">
        <component 
          :is="chart.chartType || 'ChartCard'" 
          :config="chart.config"
          :compact="true"
        />
      <div v-for="chart in chartList" :key="chart.id" class="chart-item">
        <div class="chart-item-header">
          <span class="chart-item-title">{{ chart.title || chart.type || 'Custom Chart' }}</span>
          <NuxtLink to="/charts" class="chart-open-link" title="Open in Charts">↗</NuxtLink>
        </div>
        <div class="chart-render-wrapper">
          <ChartRenderer :config="chart" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { computed, onMounted } from 'vue'
import { useSavedCharts } from '~/composables/useSavedCharts'
import { useObservations } from '~/composables/useObservations'
import ChartRenderer from '~/components/ChartRenderer.vue'

const { savedCharts } = useSavedCharts()
const charts = ref([])
const { charts, loadFromStorage } = useSavedCharts()
const { load: loadObs } = useObservations()

onMounted(() => {
  charts.value = savedCharts.value?.slice(0, 6) || []
const chartList = computed(() => (charts.value || []).slice(0, 4))

onMounted(async () => {
  loadFromStorage()
  await loadObs()
})
</script>

<style scoped>
.dashboard-saved-charts {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.widget-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.widget-title {
  margin: 0 0 1rem 0;
  margin: 0;
  font-size: 1rem;
  color: var(--text, #222);
}

.widget-link {
  font-size: 0.8rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
}

.widget-link:hover {
  text-decoration: underline;
}

.no-charts {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted, #999);
}

.create-chart-link {
  display: inline-block;
  margin-top: 0.75rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
  font-weight: 500;
}

.create-chart-link:hover {
  text-decoration: underline;
}

.charts-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  gap: 1rem;
  overflow-y: auto;
  max-height: 480px;
}

.chart-item {
  border-bottom: 1px solid var(--border-soft, #eee);
  padding-bottom: 0.75rem;
  border: 1px solid var(--border-soft, #eee);
  border-radius: 8px;
  padding: 0.75rem;
  background: var(--surface, #fff);
}

.chart-item:last-child {
  border-bottom: none;
.chart-item-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 0.5rem;
}

.chart-item-title {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text, #222);
}

.chart-open-link {
  color: var(--muted, #888);
  text-decoration: none;
  font-size: 0.85rem;
}

.chart-open-link:hover {
  color: var(--primary, #2a78d6);
}

.chart-render-wrapper {
  min-height: 160px;
}
</style>
