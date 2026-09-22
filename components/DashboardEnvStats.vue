<template>
  <div class="dash-widget env-stats">
    <div class="widget-head">
      <h3 class="widget-title">🌡️ Environmental stats</h3>
      <NuxtLink to="/analysis" class="widget-link">Analysis ›</NuxtLink>
    </div>

    <p v-if="loading && !stats.length" class="widget-note">Computing summary…</p>
    <p v-else-if="!stats.length" class="widget-note">No environmental data available.</p>

    <div v-else class="stats-grid">
      <div v-for="stat in stats" :key="stat.label" class="stat-item">
        <span class="stat-label">{{ stat.label }}</span>
        <span class="stat-value">{{ stat.value }}<small v-if="stat.unit" class="stat-unit"> {{ stat.unit }}</small></span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const { rows, load, pending: loading } = useObservations()
const { unit, tempUnit, elevValue, tempValue } = useUnits()

/** Average a numeric column over the rows that carry it. */
function mean(list, key) {
  let sum = 0
  let n = 0
  for (const r of list) {
    const v = Number(r[key])
    if (r[key] != null && !Number.isNaN(v)) { sum += v; n += 1 }
  }
  return n ? sum / n : null
}

const stats = computed(() => {
  const list = rows.value || []
  if (!list.length) return []
  const out = [{ label: 'Observations', value: list.length.toLocaleString(), unit: '' }]

  const elev = mean(list, 'elevation')
  if (elev !== null) {
    out.push({ label: 'Avg elevation', value: Math.round(elevValue(elev)).toLocaleString(), unit: unit.value === 'imperial' ? 'ft' : 'm' })
  }
  const temp = mean(list, 'tmax_d0')
  if (temp !== null) out.push({ label: 'Avg high', value: tempValue(temp).toFixed(1), unit: tempUnit.value })
  const ndvi = mean(list, 'ndvi')
  if (ndvi !== null) out.push({ label: 'Avg NDVI', value: ndvi.toFixed(2), unit: '' })
  const sm = mean(list, 'soil_moisture')
  if (sm !== null) out.push({ label: 'Avg soil moisture', value: sm.toFixed(2), unit: 'm³/m³' })
  return out
})

onMounted(() => { load() })
</script>

<style scoped>
.env-stats { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 0.65rem; }
.stat-item {
  display: flex; flex-direction: column; padding: 0.65rem 0.75rem;
  background: var(--surface-2, #f5f5f5); border-radius: 6px; border: 1px solid var(--border-soft, #eee);
}
.stat-label { font-size: 0.7rem; color: var(--muted, #666); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.25rem; }
.stat-value { font-size: 1.05rem; font-weight: 700; color: var(--text, #222); }
.stat-unit { font-size: 0.75rem; font-weight: 500; color: var(--muted, #777); }
</style>
