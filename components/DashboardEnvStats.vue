<template>
  <div class="dash-widget env-stats">
    <div class="widget-head">
      <h3 class="widget-title">📊 Data Stats</h3>
      <NuxtLink to="/analysis" class="widget-link">Analysis ›</NuxtLink>
    </div>

    <p v-if="loading && !stats.length" class="widget-note">Computing summary…</p>
    <p v-else-if="!stats.length" class="widget-note">No data available.</p>

    <div v-else class="stats-grid">
      <div v-for="stat in stats" :key="stat.label" class="stat-item" :class="stat.wide ? 'wide' : ''">
        <span class="stat-label">{{ stat.label }}</span>
        <span class="stat-value">{{ stat.value }}<small v-if="stat.unit" class="stat-unit"> {{ stat.unit }}</small></span>
        <span v-if="stat.sub" class="stat-sub">{{ stat.sub }}</span>
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

function mean(list, key) {
  let sum = 0, n = 0
  for (const r of list) {
    const v = Number(r[key])
    if (r[key] != null && !Number.isNaN(v)) { sum += v; n++ }
  }
  return n ? sum / n : null
}
function minMax(list, key) {
  let mn = Infinity, mx = -Infinity, n = 0
  for (const r of list) {
    const v = Number(r[key])
    if (r[key] != null && !Number.isNaN(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v); n++ }
  }
  return n ? { min: mn, max: mx } : null
}
function coverage(list, key) {
  const n = list.filter((r) => r[key] != null && r[key] !== '').length
  return list.length ? Math.round((n / list.length) * 100) : 0
}

const stats = computed(() => {
  const list = rows.value || []
  if (!list.length) return []
  const out = []

  out.push({ label: 'Observations', value: list.length.toLocaleString(), unit: '' })

  const speciesSet = new Set(list.map((r) => r.species).filter(Boolean))
  out.push({ label: 'Species', value: speciesSet.size.toLocaleString(), unit: '' })

  const genusSet = new Set([...speciesSet].map((s) => s.split(' ')[0]))
  out.push({ label: 'Genera', value: genusSet.size.toLocaleString(), unit: '' })

  // Date range
  const dates = list.map((r) => r.date).filter(Boolean).sort()
  if (dates.length) {
    const y0 = dates[0].slice(0, 4)
    const y1 = dates[dates.length - 1].slice(0, 4)
    out.push({ label: 'Date range', value: y0 === y1 ? y0 : `${y0}–${y1}`, unit: '', wide: true })
  }

  const elev = mean(list, 'elevation')
  if (elev !== null) {
    const elevMM = minMax(list, 'elevation')
    const lo = elevMM ? Math.round(elevValue(elevMM.min)) : null
    const hi = elevMM ? Math.round(elevValue(elevMM.max)) : null
    const unitStr = unit.value === 'imperial' ? 'ft' : 'm'
    out.push({
      label: 'Avg elevation', value: Math.round(elevValue(elev)).toLocaleString(), unit: unitStr,
      sub: lo !== null ? `${lo.toLocaleString()}–${hi.toLocaleString()} ${unitStr} range` : undefined,
    })
  }

  const temp = mean(list, 'tmax_d0')
  if (temp !== null) {
    const tmm = minMax(list, 'tmax_d0')
    out.push({
      label: 'Avg high temp', value: tempValue(temp).toFixed(1), unit: tempUnit.value,
      sub: tmm ? `${tempValue(tmm.min).toFixed(0)}–${tempValue(tmm.max).toFixed(0)} ${tempUnit.value} range` : undefined,
    })
  }

  const ndvi = mean(list, 'ndvi')
  if (ndvi !== null) out.push({ label: 'Avg NDVI', value: ndvi.toFixed(2), unit: '' })

  const sm = mean(list, 'soil_moisture')
  if (sm !== null) out.push({ label: 'Avg soil moisture', value: sm.toFixed(2), unit: 'm³/m³' })

  const prcp = mean(list, 'prcp_d0')
  if (prcp !== null) out.push({ label: 'Avg precip (obs day)', value: prcp.toFixed(1), unit: 'mm' })

  // Data source breakdown
  const sources = {}
  for (const r of list) {
    const src = r.source || r.dataset_source || 'unknown'
    sources[src] = (sources[src] || 0) + 1
  }
  const srcEntries = Object.entries(sources).sort((a, b) => b[1] - a[1]).slice(0, 3)
  if (srcEntries.length > 1) {
    for (const [src, n] of srcEntries) {
      out.push({ label: src, value: n.toLocaleString(), unit: '' })
    }
  }

  // Field coverage
  const elevCov = coverage(list, 'elevation')
  if (elevCov < 100) out.push({ label: 'Elevation coverage', value: `${elevCov}`, unit: '%' })

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
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 0.55rem; overflow-y: auto; }
.stat-item {
  display: flex; flex-direction: column; padding: 0.55rem 0.65rem;
  background: var(--surface-2, #f5f5f5); border-radius: 6px; border: 1px solid var(--border-soft, #eee);
}
.stat-item.wide { grid-column: span 2; }
.stat-label { font-size: 0.68rem; color: var(--muted, #666); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 0.2rem; }
.stat-value { font-size: 1rem; font-weight: 700; color: var(--text, #222); line-height: 1.2; }
.stat-unit { font-size: 0.72rem; font-weight: 500; color: var(--muted, #777); }
.stat-sub { font-size: 0.65rem; color: var(--muted, #aaa); margin-top: 0.2rem; }
</style>
