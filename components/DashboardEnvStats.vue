<template>
  <div class="dashboard-env-stats">
    <h3 class="widget-title">🌡️ Environmental Stats</h3>
    <div v-if="stats.length === 0" class="no-stats">
      <p>Select an area to view stats.</p>
    <div class="widget-head">
      <h3 class="widget-title">🌡️ Environmental Stats</h3>
      <NuxtLink to="/analysis" class="widget-link">Analysis ›</NuxtLink>
    </div>

    <div v-if="loading && !stats.length" class="no-stats">
      <p>Computing summary…</p>
    </div>

    <div v-else-if="stats.length === 0" class="no-stats">
      <p>No environmental data available.</p>
    </div>

    <div v-else class="stats-grid">
      <div v-for="stat in stats" :key="stat.label" class="stat-item">
        <span class="stat-label">{{ stat.label }}</span>
        <span class="stat-value">{{ stat.value }}{{ stat.unit ? ` ${stat.unit}` : '' }}</span>
        <span class="stat-value">{{ stat.value }}<small v-if="stat.unit" class="stat-unit"> {{ stat.unit }}</small></span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { computed, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const stats = ref([])
const { rows, load, pending: loading } = useObservations()
const { unit, tempUnit, elevValue, tempValue } = useUnits()

onMounted(() => {
  // Placeholder - would integrate with actual environmental data
  if (import.meta.client) {
    const saved = localStorage.getItem('map-overlay')
    if (saved) {
      try {
        const overlay = JSON.parse(saved)
        // Generate sample stats based on current view
        stats.value = [
          { label: 'Elevation', value: '1,250', unit: 'm' },
          { label: 'Avg Temp', value: '18.5', unit: '°C' },
          { label: 'Annual Rain', value: '850', unit: 'mm' },
          { label: 'NDVI', value: '0.65', unit: '' },
        ]
      } catch {
        stats.value = []
      }
const stats = computed(() => {
  const list = rows.value || []
  if (!list.length) return []

  // Extract valid numeric samples
  let elevSum = 0, elevCount = 0
  let tempSum = 0, tempCount = 0
  let precipSum = 0, precipCount = 0
  let ndviSum = 0, ndviCount = 0

  for (const r of list) {
    if (r.elevation != null && !Number.isNaN(Number(r.elevation))) {
      elevSum += Number(r.elevation)
      elevCount++
    }
    if (r.temp != null && !Number.isNaN(Number(r.temp))) {
      tempSum += Number(r.temp)
      tempCount++
    }
    if (r.prcp_ann != null && !Number.isNaN(Number(r.prcp_ann))) {
      precipSum += Number(r.prcp_ann)
      precipCount++
    }
    if (r.ndvi != null && !Number.isNaN(Number(r.ndvi))) {
      ndviSum += Number(r.ndvi)
      ndviCount++
    }
  }

  const result = [
    {
      label: 'Observations',
      value: list.length.toLocaleString(),
      unit: '',
    },
  ]

  if (elevCount > 0) {
    const avgM = elevSum / elevCount
    const converted = elevValue(avgM)
    result.push({
      label: 'Avg Elevation',
      value: Math.round(converted).toLocaleString(),
      unit: unit.value === 'imperial' ? 'ft' : 'm',
    })
  }

  if (tempCount > 0) {
    const avgC = tempSum / tempCount
    const converted = tempValue(avgC)
    result.push({
      label: 'Avg Temp',
      value: converted.toFixed(1),
      unit: tempUnit.value,
    })
  }

  if (precipCount > 0) {
    const avgMm = precipSum / precipCount
    result.push({
      label: 'Annual Rain',
      value: Math.round(avgMm).toLocaleString(),
      unit: 'mm',
    })
  }

  if (ndviCount > 0) {
    const avgNdvi = ndviSum / ndviCount
    result.push({
      label: 'Avg NDVI',
      value: avgNdvi.toFixed(2),
      unit: '',
    })
  }

  return result
})

onMounted(async () => {
  await load()
})
</script>

<style scoped>
.dashboard-env-stats {
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

.no-stats {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted, #999);
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
  gap: 0.65rem;
}

.stat-item {
  display: flex;
  flex-direction: column;
  padding: 0.75rem;
  padding: 0.65rem 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 6px;
  border: 1px solid var(--border-soft, #eee);
}

.stat-label {
  font-size: 0.75rem;
  font-size: 0.7rem;
  color: var(--muted, #666);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  letter-spacing: 0.04em;
  margin-bottom: 0.25rem;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 600;
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text, #222);
}

.stat-unit {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--muted, #777);
}
</style>
