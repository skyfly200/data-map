<template>
  <div class="dash-widget recent-rain-temp">
    <div class="widget-head">
      <h3 class="widget-title">🌡️ Recent rain & temps</h3>
      <div class="widget-actions">
        <button class="toggle-btn" :class="{ active: metric === 'precip' }" @click="metric = 'precip'">Rain</button>
        <button class="toggle-btn" :class="{ active: metric === 'temp' }" @click="metric = 'temp'">Temp</button>
        <button class="toggle-btn" :class="{ active: metric === 'both' }" @click="metric = 'both'">Both</button>
      </div>
    </div>

    <p v-if="loading && !bins.length" class="widget-note">Loading…</p>
    <p v-else-if="!hasPrecip && !hasTemp" class="widget-note">No recent rain/temp data in observations.</p>

    <div v-else class="chart-body">
      <svg class="chart-svg" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none">
        <!-- Rain bars -->
        <g v-if="metric !== 'temp'">
          <rect
            v-for="(b, i) in bins" :key="`r${i}`"
            :x="barX(i)" :y="H - barH(b.precipMean)" :width="barW - 1" :height="barH(b.precipMean)"
            class="rain-bar"
          />
        </g>
        <!-- Temp max/min range band -->
        <g v-if="metric !== 'precip'">
          <polygon v-if="tempBandPts" :points="tempBandPts" class="temp-band" />
          <polyline v-if="tempMaxLine" :points="tempMaxLine" class="temp-max-line" fill="none" />
          <polyline v-if="tempMinLine" :points="tempMinLine" class="temp-min-line" fill="none" />
        </g>
      </svg>

      <div class="x-axis">
        <span v-for="lbl in xLabels" :key="lbl">{{ lbl }}</span>
      </div>

      <div class="legend">
        <span v-if="metric !== 'temp'" class="legend-item">
          <span class="dot rain-dot"></span>Precip avg
        </span>
        <span v-if="metric !== 'precip'" class="legend-item">
          <span class="dot temp-hi-dot"></span>Hi
          <span class="dot temp-lo-dot"></span>Lo
        </span>
        <span v-if="metric !== 'temp' && hasPrecip" class="legend-val">
          {{ totalPrecip.toFixed(1) }}mm total
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const { rows, load, pending: loading } = useObservations()
const { tempValue } = useUnits()

const metric = ref('both')
const W = 260
const H = 90
const N_BINS = 16
const BIN_DAYS = 7
const now = Date.now()

const bins = computed(() => {
  const buckets = Array.from({ length: N_BINS }, () => ({ precipSum: 0, precipN: 0, tmaxSum: 0, tminSum: 0, tempN: 0 }))
  for (const r of rows.value || []) {
    if (!r.date) continue
    const d = new Date(r.date)
    if (isNaN(d)) continue
    const ageDays = (now - d.getTime()) / 86400000
    if (ageDays < 0 || ageDays > N_BINS * BIN_DAYS) continue
    const idx = N_BINS - 1 - Math.floor(ageDays / BIN_DAYS)
    if (idx < 0 || idx >= N_BINS) continue
    const b = buckets[idx]
    const p = Number(r.prcp_d0)
    if (Number.isFinite(p)) { b.precipSum += p; b.precipN++ }
    const tmax = Number(r.tmax_d0)
    const tmin = Number(r.tmin_d0 ?? r.tmin)
    if (Number.isFinite(tmax) && Number.isFinite(tmin)) { b.tmaxSum += tmax; b.tminSum += tmin; b.tempN++ }
  }
  return buckets.map((b) => ({
    precipMean: b.precipN ? b.precipSum / b.precipN : 0,
    tmaxMean: b.tempN ? tempValue(b.tmaxSum / b.tempN) : null,
    tminMean: b.tempN ? tempValue(b.tminSum / b.tempN) : null,
  }))
})

const hasPrecip = computed(() => bins.value.some((b) => b.precipMean > 0))
const hasTemp = computed(() => bins.value.some((b) => b.tmaxMean !== null))
const totalPrecip = computed(() => bins.value.reduce((s, b) => s + b.precipMean, 0))

const maxPrecip = computed(() => Math.max(0.01, ...bins.value.map((b) => b.precipMean)))
const tempVals = computed(() => bins.value.flatMap((b) => [b.tmaxMean, b.tminMean]).filter((v) => v !== null))
const tempMin = computed(() => tempVals.value.length ? Math.min(...tempVals.value) : 0)
const tempMax = computed(() => tempVals.value.length ? Math.max(...tempVals.value) : 1)
const tempRange = computed(() => Math.max(0.01, tempMax.value - tempMin.value))

const barW = computed(() => W / N_BINS)
function barX(i) { return i * barW.value }
function barH(p) { return (p / maxPrecip.value) * (H * 0.7) }
function tempY(v) { return H - 4 - ((v - tempMin.value) / tempRange.value) * (H * 0.8) }

const tempMaxLine = computed(() => {
  const pts = bins.value.map((b, i) => b.tmaxMean !== null ? `${barX(i) + barW.value / 2},${tempY(b.tmaxMean)}` : null).filter(Boolean)
  return pts.length >= 2 ? pts.join(' ') : null
})
const tempMinLine = computed(() => {
  const pts = bins.value.map((b, i) => b.tminMean !== null ? `${barX(i) + barW.value / 2},${tempY(b.tminMean)}` : null).filter(Boolean)
  return pts.length >= 2 ? pts.join(' ') : null
})
const tempBandPts = computed(() => {
  const top = bins.value.map((b, i) => b.tmaxMean !== null ? `${barX(i) + barW.value / 2},${tempY(b.tmaxMean)}` : null).filter(Boolean)
  const bot = bins.value.map((b, i) => b.tminMean !== null ? `${barX(i) + barW.value / 2},${tempY(b.tminMean)}` : null).filter(Boolean).reverse()
  return top.length >= 2 ? `${top.join(' ')} ${bot.join(' ')}` : null
})

const xLabels = computed(() => {
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  return Array.from({ length: N_BINS }, (_, i) => {
    if (i % 4 !== 0) return ''
    const ageDays = (N_BINS - i) * BIN_DAYS
    const d = new Date(now - ageDays * 86400000)
    return MONTHS[d.getMonth()]
  })
})

onMounted(() => { load() })
</script>

<style scoped>
.recent-rain-temp { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: center; justify-content: space-between; gap: 0.4rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.3rem; }
.toggle-btn {
  font: inherit; font-size: 0.68rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #888);
  border-radius: 999px; padding: 0.12rem 0.48rem; cursor: pointer;
}
.toggle-btn.active { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }
.widget-note { font-size: 0.85rem; color: var(--muted, #888); }

.chart-body { display: flex; flex-direction: column; gap: 0.2rem; flex: 1; }
.chart-svg { width: 100%; height: 90px; display: block; overflow: visible; }
.rain-bar { fill: #4a90d9; opacity: 0.55; }
.temp-band { fill: rgba(230, 80, 50, 0.1); }
.temp-max-line { stroke: #e65032; stroke-width: 1.6; stroke-linejoin: round; }
.temp-min-line { stroke: #4a90d9; stroke-width: 1.6; stroke-linejoin: round; }

.x-axis { display: flex; justify-content: space-between; font-size: 0.6rem; color: var(--muted, #aaa); }
.x-axis span { flex: 1; text-align: left; }

.legend { display: flex; align-items: center; gap: 0.6rem; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 0.3rem; font-size: 0.65rem; color: var(--muted, #888); }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.rain-dot { background: #4a90d9; opacity: 0.7; }
.temp-hi-dot { background: #e65032; }
.temp-lo-dot { background: #4a90d9; }
.legend-val { margin-left: auto; font-size: 0.7rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }
</style>
