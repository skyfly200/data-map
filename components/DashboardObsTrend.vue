<template>
  <div class="dash-widget obs-trend">
    <div class="widget-head">
      <h3 class="widget-title">📈 Observations trend</h3>
      <div class="widget-actions">
        <button class="toggle-btn" :class="{ active: mode === 'monthly' }" @click="mode = 'monthly'">Monthly</button>
        <button class="toggle-btn" :class="{ active: mode === 'yearly' }" @click="mode = 'yearly'">Yearly</button>
        <NuxtLink to="/explore" class="widget-link">Explore ›</NuxtLink>
      </div>
    </div>

    <p v-if="loading && !rows.length" class="ot-note">Loading…</p>
    <p v-else-if="!bins.length" class="ot-note">No observations with dates loaded.</p>

    <div v-else class="ot-body">
      <svg class="ot-svg" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none">
        <g>
          <rect
            v-for="(b, i) in bins"
            :key="i"
            :x="bx(i) + 1"
            :y="H - bh(b.count)"
            :width="Math.max(1, bw - 2)"
            :height="bh(b.count)"
            class="ot-bar"
            :title="`${b.label}: ${b.count.toLocaleString()}`"
          />
        </g>
        <!-- zero line -->
        <line x1="0" :y1="H" :x2="W" :y2="H" class="ot-baseline" />
      </svg>

      <div class="ot-xaxis">
        <span v-for="(b, i) in xTicks" :key="i" class="ot-tick" :style="{ left: `${b.pct}%` }">{{ b.label }}</span>
      </div>

      <div class="ot-footer">
        <span class="ot-total">{{ totalObs.toLocaleString() }} total</span>
        <span class="ot-peak" v-if="peak">Peak: <strong>{{ peak.label }}</strong> ({{ peak.count.toLocaleString() }})</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()

const mode = ref('monthly')

const W = 260
const H = 72

// Build bins from rows
const bins = computed(() => {
  if (!rows.value?.length) return []
  if (mode.value === 'yearly') return yearlyBins()
  return monthlyBins()
})

function yearlyBins() {
  const m = new Map()
  for (const r of rows.value) {
    const y = r.year ?? (r.date ? new Date(r.date).getFullYear() : null)
    if (!y || isNaN(y)) continue
    m.set(y, (m.get(y) || 0) + 1)
  }
  return [...m.entries()]
    .sort(([a], [b]) => a - b)
    .map(([y, count]) => ({ label: String(y), count }))
}

function monthlyBins() {
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  const now = Date.now()
  const m = new Map()
  for (const r of rows.value) {
    if (!r.date) continue
    const d = new Date(r.date)
    if (isNaN(d)) continue
    const ageDays = (now - d.getTime()) / 86400000
    if (ageDays > 730) continue  // last 2 years
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}`
    m.set(key, (m.get(key) || 0) + 1)
  }
  return [...m.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, count]) => {
      const [y, mo] = key.split('-')
      return { label: `${MONTHS[+mo - 1]} ${y}`, key, count }
    })
}

const maxCount = computed(() => Math.max(1, ...bins.value.map((b) => b.count)))
const totalObs = computed(() => bins.value.reduce((s, b) => s + b.count, 0))
const peak = computed(() => {
  if (!bins.value.length) return null
  return bins.value.reduce((best, b) => (b.count > best.count ? b : best), bins.value[0])
})

const bw = computed(() => bins.value.length ? W / bins.value.length : W)
function bx(i) { return i * bw.value }
function bh(count) { return (count / maxCount.value) * (H - 4) }

// Show ~6 evenly spaced x-axis labels
const xTicks = computed(() => {
  const n = bins.value.length
  if (!n) return []
  const step = Math.max(1, Math.floor(n / 6))
  return bins.value
    .map((b, i) => ({ label: b.label, pct: (i / (n - 1)) * 100 }))
    .filter((_, i) => i % step === 0 || i === n - 1)
})

onMounted(() => { load() })
</script>

<style scoped>
.obs-trend { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: center; justify-content: space-between; gap: 0.4rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; align-items: center; gap: 0.4rem; }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }

.toggle-btn {
  font: inherit; font-size: 0.68rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #888);
  border-radius: 999px; padding: 0.12rem 0.48rem; cursor: pointer;
}
.toggle-btn.active { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }

.ot-note { font-size: 0.85rem; color: var(--muted, #888); }

.ot-body { display: flex; flex-direction: column; gap: 0.2rem; flex: 1; }

.ot-svg { width: 100%; height: 72px; display: block; overflow: visible; }
.ot-bar { fill: var(--accent, #2a78d6); opacity: 0.75; transition: opacity 0.15s; }
.ot-bar:hover { opacity: 1; }
.ot-baseline { stroke: var(--border-soft, #ddd); stroke-width: 1; }

.ot-xaxis {
  position: relative; height: 1.1rem;
}
.ot-tick {
  position: absolute; transform: translateX(-50%);
  font-size: 0.58rem; color: var(--muted, #aaa); white-space: nowrap;
}

.ot-footer {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 0.7rem; color: var(--muted, #888);
}
.ot-peak strong { color: var(--text, #444); }
</style>
