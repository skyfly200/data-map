<template>
  <div class="dashboard-saved-charts">
    <h3 class="widget-title">📊 Saved Charts</h3>
    <div v-if="charts.length === 0" class="no-charts">
      <p>No saved charts yet.</p>
      <NuxtLink to="/charts" class="create-chart-link">Create your first chart</NuxtLink>
    </div>
    <div v-else class="charts-list">
      <div v-for="chart in charts" :key="chart.id" class="chart-item">
        <component 
          :is="chart.chartType || 'ChartCard'" 
          :config="chart.config"
          :compact="true"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useSavedCharts } from '~/composables/useSavedCharts'

const { savedCharts } = useSavedCharts()
const charts = ref([])

onMounted(() => {
  charts.value = savedCharts.value?.slice(0, 6) || []
})
</script>

<style scoped>
.dashboard-saved-charts {
  height: 100%;
}

.widget-title {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text, #222);
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
}

.chart-item {
  border-bottom: 1px solid var(--border-soft, #eee);
  padding-bottom: 0.75rem;
}

.chart-item:last-child {
  border-bottom: none;
}
</style>
