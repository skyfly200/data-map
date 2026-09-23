<template>
  <div class="dash-widget phenology">
    <div class="widget-head">
      <h3 class="widget-title">📅 Phenology</h3>
      <div class="widget-actions">
        <button class="toggle-btn" :class="{ active: mode === 'species' }" @click="mode = 'species'">Species</button>
        <button class="toggle-btn" :class="{ active: mode === 'total' }" @click="mode = 'total'">Total</button>
        <NuxtLink to="/charts" class="widget-link">Charts ›</NuxtLink>
      </div>
    </div>

    <p v-if="loading && !rows.length" class="ph-note">Loading…</p>
    <p v-else-if="!rows.length" class="ph-note">No observations loaded.</p>

    <div v-else class="ph-wrap">
      <!-- Month header row -->
      <div class="ph-grid" :style="gridStyle">
        <div class="ph-corner"></div>
        <div v-for="m in MONTHS" :key="m" class="ph-month-label"
             :class="{ 'ph-current': m === currentMonth }">{{ m }}</div>

        <!-- Total row (mode=total) -->
        <template v-if="mode === 'total'">
          <div class="ph-row-label ph-total-label">All</div>
          <div v-for="m in MONTHS" :key="m" class="ph-cell"
               :style="cellStyle(totalByMonth[m] || 0, totalMax)"
               :title="`${m}: ${(totalByMonth[m] || 0).toLocaleString()} obs`">
            <span class="ph-val">{{ fmt(totalByMonth[m] || 0) }}</span>
          </div>
        </template>

        <!-- Per-species rows (mode=species) -->
        <template v-else>
          <template v-for="sp in topSpecies" :key="sp">
            <div class="ph-row-label" :title="sp">{{ sp }}</div>
            <div v-for="m in MONTHS" :key="m" class="ph-cell"
                 :style="cellStyle(matrix[sp]?.[m] || 0, speciesMax)"
                 :title="`${sp} · ${m}: ${(matrix[sp]?.[m] || 0).toLocaleString()} obs`">
              <span class="ph-val">{{ fmt(matrix[sp]?.[m] || 0) }}</span>
            </div>
          </template>
        </template>
      </div>

      <p class="ph-caption">
        {{ mode === 'total' ? 'Monthly totals across all species' : `Top ${topSpecies.length} species by observation count` }}
        · {{ rows.length.toLocaleString() }} obs total
      </p>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const TOP_N = 8

const mode = ref('species') // 'species' | 'total'

const currentMonth = MONTHS[new Date().getMonth()]

// Build month → count map for all obs.
const totalByMonth = computed(() => {
  const m = {}
  for (const r of rows.value || []) {
    const mo = monthName(r)
    if (mo) m[mo] = (m[mo] || 0) + 1
  }
  return m
})

const totalMax = computed(() => Math.max(1, ...Object.values(totalByMonth.value)))

// Build species × month matrix for top-N species.
const matrix = computed(() => {
  const counts = new Map()
  for (const r of rows.value || []) {
    const sp = r.species
    if (!sp) continue
    const mo = monthName(r)
    if (!mo) continue
    if (!counts.has(sp)) counts.set(sp, {})
    counts.get(sp)[mo] = (counts.get(sp)[mo] || 0) + 1
  }
  return Object.fromEntries(counts)
})

const topSpecies = computed(() => {
  const totals = new Map()
  for (const r of rows.value || []) {
    if (r.species) totals.set(r.species, (totals.get(r.species) || 0) + 1)
  }
  return [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP_N)
    .map(([sp]) => sp)
})

const speciesMax = computed(() => {
  let max = 1
  for (const sp of topSpecies.value) {
    for (const m of MONTHS) {
      const v = matrix.value[sp]?.[m] || 0
      if (v > max) max = v
    }
  }
  return max
})

// CSS grid: label col + 12 month cols.
const gridStyle = computed(() => ({
  gridTemplateColumns: `5.5rem repeat(12, 1fr)`,
}))

// Interpolate a blue→teal→green→yellow→orange heat scale.
function cellStyle(val, max) {
  const t = max > 0 ? val / max : 0
  const alpha = t === 0 ? 0 : 0.12 + t * 0.78
  // hue: 220 (blue) → 40 (amber) as t goes 0→1
  const hue = Math.round(220 - t * 180)
  const sat = t === 0 ? 0 : 55 + t * 30
  const lit = t === 0 ? 97 : 72 - t * 28
  return {
    background: t === 0
      ? 'var(--surface-2, #f5f5f5)'
      : `hsla(${hue}, ${sat}%, ${lit}%, ${alpha + 0.5})`,
    color: t > 0.6 ? '#fff' : 'var(--text, #222)',
  }
}

function fmt(n) {
  if (!n) return ''
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

function monthName(r) {
  if (r.month_name) return r.month_name
  if (r.month) return MONTHS[r.month - 1]
  if (r.date) {
    const d = new Date(r.date)
    if (!isNaN(d)) return MONTHS[d.getUTCMonth()]
  }
  return null
}

onMounted(() => { load() })
</script>

<style scoped>
.phenology { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; gap: 0.4rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; align-items: center; gap: 0.4rem; }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }

.toggle-btn {
  font: inherit; font-size: 0.72rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #888);
  border-radius: 999px; padding: 0.15rem 0.55rem; cursor: pointer;
}
.toggle-btn.active { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }

.ph-note { font-size: 0.85rem; color: var(--muted, #888); text-align: center; padding: 1rem 0; }

.ph-wrap { overflow-x: auto; }

.ph-grid {
  display: grid;
  gap: 2px;
  min-width: 480px;
}

.ph-corner { /* empty top-left */ }

.ph-month-label {
  font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: var(--muted, #888); text-align: center; padding-bottom: 2px;
}
.ph-month-label.ph-current { color: var(--accent, #2a78d6); font-weight: 600; }

.ph-row-label {
  font-size: 0.68rem; font-style: italic; color: var(--text, #333);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  align-self: center; padding-right: 4px;
}
.ph-total-label { font-style: normal; font-weight: 600; font-size: 0.72rem; }

.ph-cell {
  border-radius: 3px; min-height: 20px; display: flex; align-items: center;
  justify-content: center; transition: filter 0.1s;
  cursor: default;
}
.ph-cell:hover { filter: brightness(0.9); }

.ph-val {
  font-size: 0.58rem; font-variant-numeric: tabular-nums; line-height: 1;
  pointer-events: none;
}

.ph-caption {
  margin: 0.25rem 0 0; font-size: 0.68rem; color: var(--muted, #999);
}
</style>
