<template>
  <div class="dashboard-env-stats">
    <h3 class="widget-title">🌡️ Environmental Stats</h3>
    <div v-if="stats.length === 0" class="no-stats">
      <p>Select an area to view stats.</p>
    </div>
    <div v-else class="stats-grid">
      <div v-for="stat in stats" :key="stat.label" class="stat-item">
        <span class="stat-label">{{ stat.label }}</span>
        <span class="stat-value">{{ stat.value }}{{ stat.unit ? ` ${stat.unit}` : '' }}</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const stats = ref([])

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
    }
  }
})
</script>

<style scoped>
.dashboard-env-stats {
  height: 100%;
}

.widget-title {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text, #222);
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
}

.stat-item {
  display: flex;
  flex-direction: column;
  padding: 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 6px;
}

.stat-label {
  font-size: 0.75rem;
  color: var(--muted, #666);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.25rem;
}

.stat-value {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text, #222);
}
</style>
