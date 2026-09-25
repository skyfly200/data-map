<template>
  <div class="dash-widget recent-rain-temp">
    <div class="widget-head">
      <h3 class="widget-title">🌡️ Recent rain & temps</h3>
      <div class="widget-actions">
        <select v-model="rangeKey" class="ctrl-select" title="Time range">
          <option v-for="r in RANGES" :key="r.key" :value="r.key">{{ r.label }}</option>
        </select>
        <div class="toggle-group">
          <button class="toggle-btn" :class="{ active: metric === 'precip' }" @click="metric = 'precip'">Rain</button>
          <button class="toggle-btn" :class="{ active: metric === 'temp' }" @click="metric = 'temp'">Temp</button>
          <button class="toggle-btn" :class="{ active: metric === 'both' }" @click="metric = 'both'">Both</button>
        </div>
        <div class="toggle-group">
          <button class="toggle-btn" :class="{ active: tempUnitLocal === 'C' }" @click="tempUnitLocal = 'C'">°C</button>
          <button class="toggle-btn" :class="{ active: tempUnitLocal === 'F' }" @click="tempUnitLocal = 'F'">°F</button>
        </div>
      </div>
    </div>

    <p v-if="loading && !hasAnyData" class="widget-note">Loading…</p>
    <p v-else-if="!hasAnyData" class="widget-note">No rain/temp data in this window.</p>

    <div v-else class="chart-body">
      <!-- Y-axis labels -->
      <div class="chart-area">
        <div v-if="metric !== 'precip'" class="y-axis temp-axis">
          <span>{{ fmt(tempAxisMax) }}°</span>
          <span>{{ fmt((tempAxisMax + tempAxisMin) / 2) }}°</span>
          <span>{{ fmt(tempAxisMin) }}°</span>
        </div>
        <div class="chart-col">
          <svg class="chart-svg" :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none">
            <!-- Rain bars -->
            <g v-if="metric !== 'temp'">
              <rect
                v-for="(b, i) in bins" :key="`r${i}`"
                :x="barX(i)" :y="H - barH(b.precipMean)" :width="barW - 1" :height="barH(b.precipMean)"
                class="rain-bar"
              />
            </g>
            <!-- Temp range band + lines -->
            <g v-if="metric !== 'precip'">
              <polygon v-if="tempBandPts" :points="tempBandPts" class="temp-band" />
              <polyline v-if="tempMaxLine" :points="tempMaxLine" class="temp-max-line" fill="none" />
              <polyline v-if="tempMinLine" :points="tempMinLine" class="temp-min-line" fill="none" />
            </g>
            <!-- Zero/reference line -->
            <line v-if="metric !== 'precip' && zeroY !== null"
                  x1="0" :y1="zeroY" :x2="W" :y2="zeroY"
                  class="zero-line" />
          </svg>
          <div class="x-axis">
            <span v-for="(lbl, i) in xLabels" :key="i">{{ lbl }}</span>
          </div>
        </div>
        <div v-if="metric !== 'temp'" class="y-axis precip-axis">
          <span>{{ precipAxisMax.toFixed(precipAxisMax < 10 ? 1 : 0) }}</span>
          <span></span>
          <span>0</span>
        </div>
      </div>

      <div class="legend">
        <span v-if="metric !== 'temp'" class="legend-item">
          <span class="dot rain-dot"></span>Precip (mm)
        </span>
        <span v-if="metric !== 'precip'" class="legend-item">
          <span class="dot temp-hi-dot"></span>Hi &nbsp;
          <span class="dot temp-lo-dot"></span>Lo (°{{ tempUnitLocal }})
        </span>
        <span v-if="metric !== 'temp' && hasPrecip" class="legend-val">
          {{ totalPrecip.toFixed(1) }} mm total
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, ref, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()

const metric = ref('both')
const tempUnitLocal = ref('F')

// Responsive range: bins × bin-size covers the full window
const RANGES = [
  { key: '7d',   label: 'Past 7 days',   bins: 7,  binDays: 1 },
  { key: '30d',  label: 'Past 30 days',  bins: 15, binDays: 2 },
  { key: '90d',  label: 'Past 90 days',  bins: 18, binDays: 5 },
  { key: '1y',   label: 'Past year',     bins: 24, binDays: 15 },
  { key: '2y',   label: 'Past 2 years',  bins: 24, binDays: 30 },
]
const rangeKey = ref('30d')
const range = computed(() => RANGES.find((r) => r.key === rangeKey.value) || RANGES[1])

const W = 260
const H = 90
const now = Date.now()

function toDisplay(celsius) {
  return tempUnitLocal.value === 'F' ? celsius * 9 / 5 + 32 : celsius
}
function fmt(v) { return v == null ? '' : Math.round(v) }

const bins = computed(() => {
  const { bins: N, binDays } = range.value
  const buckets = Array.from({ length: N }, () => ({ precipSum: 0, precipN: 0, tmaxSum: 0, tminSum: 0, tempN: 0 }))
  for (const r of rows.value || []) {
    if (!r.date) continue
    const d = new Date(r.date)
    if (isNaN(d)) continue
    const ageDays = (now - d.getTime()) / 86400000
    if (ageDays < 0 || ageDays > N * binDays) continue
    const idx = N - 1 - Math.floor(ageDays / binDays)
    if (idx < 0 || idx >= N) continue
    const b = buckets[idx]
    const p = Number(r.prcp_d0)
    if (Number.isFinite(p)) { b.precipSum += p; b.precipN++ }
    const tmax = Number(r.tmax_d0)
    const tmin = Number(r.tmin_d0 ?? r.tmin)
    if (Number.isFinite(tmax) && Number.isFinite(tmin)) {
      b.tmaxSum += tmax; b.tminSum += tmin; b.tempN++
    }
  }
  return buckets.map((b) => ({
    precipMean: b.precipN ? b.precipSum / b.precipN : 0,
    tmaxMean: b.tempN ? toDisplay(b.tmaxSum / b.tempN) : null,
    tminMean: b.tempN ? toDisplay(b.tminSum / b.tempN) : null,
  }))
})

const hasPrecip = computed(() => bins.value.some((b) => b.precipMean > 0))
const hasTemp = computed(() => bins.value.some((b) => b.tmaxMean !== null))
const hasAnyData = computed(() => hasPrecip.value || hasTemp.value)
const totalPrecip = computed(() => bins.value.reduce((s, b) => s + b.precipMean, 0))

const precipAxisMax = computed(() => Math.max(0.01, ...bins.value.map((b) => b.precipMean)))
const tempVals = computed(() => bins.value.flatMap((b) => [b.tmaxMean, b.tminMean]).filter((v) => v !== null))
const tempAxisMin = computed(() => {
  if (!tempVals.value.length) return 0
  const mn = Math.min(...tempVals.value)
  return Math.floor(mn / 5) * 5 - 5
})
const tempAxisMax = computed(() => {
  if (!tempVals.value.length) return 30
  const mx = Math.max(...tempVals.value)
  return Math.ceil(mx / 5) * 5 + 5
})
const tempRange = computed(() => Math.max(0.01, tempAxisMax.value - tempAxisMin.value))

// 0°C / 32°F reference
const freezeC = computed(() => tempUnitLocal.value === 'F' ? 32 : 0)
const zeroY = computed(() => {
  if (!hasTemp.value) return null
  const v = freezeC.value
  if (v < tempAxisMin.value || v > tempAxisMax.value) return null
  return H - ((v - tempAxisMin.value) / tempRange.value) * H
})

const barW = computed(() => W / range.value.bins)
function barX(i) { return i * barW.value }
function barH(p) { return (p / precipAxisMax.value) * (H * 0.7) }
function tempY(v) { return H - ((v - tempAxisMin.value) / tempRange.value) * H }

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

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
const xLabels = computed(() => {
  const { bins: N, binDays } = range.value
  return Array.from({ length: N }, (_, i) => {
    // Show label every ~4 bins, always show the first and last
    if (i !== 0 && i !== N - 1 && i % Math.max(1, Math.floor(N / 5)) !== 0) return ''
    const ageDays = (N - 1 - i) * binDays
    const d = new Date(now - ageDays * 86400000)
    if (binDays <= 2) {
      // Daily: show "Mon 25" style
      return `${d.getDate()}`
    }
    return MONTHS[d.getMonth()]
  })
})

onMounted(() => { load() })
</script>

<style scoped>
.recent-rain-temp { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.4rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); flex-shrink: 0; }
.widget-actions { display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap; }

.ctrl-select {
  font: inherit; font-size: 0.68rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #555);
  border-radius: 6px; padding: 0.15rem 0.4rem; cursor: pointer;
}
.toggle-group { display: flex; gap: 0; }
.toggle-btn {
  font: inherit; font-size: 0.65rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #888);
  padding: 0.12rem 0.4rem; cursor: pointer; line-height: 1.4;
}
.toggle-group .toggle-btn:first-child { border-radius: 4px 0 0 4px; }
.toggle-group .toggle-btn:last-child { border-radius: 0 4px 4px 0; }
.toggle-group .toggle-btn:not(:first-child) { border-left: none; }
.toggle-btn.active { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; z-index: 1; }

.widget-note { font-size: 0.85rem; color: var(--muted, #888); }

.chart-body { display: flex; flex-direction: column; gap: 0.15rem; flex: 1; min-height: 0; }
.chart-area { display: flex; align-items: stretch; gap: 0.25rem; flex: 1; min-height: 0; }
.chart-col { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.chart-svg { width: 100%; flex: 1; display: block; overflow: visible; }

.y-axis {
  display: flex; flex-direction: column; justify-content: space-between;
  font-size: 0.55rem; color: var(--muted, #bbb); text-align: right;
  padding: 2px 0; min-width: 22px;
}
.precip-axis { color: #4a90d9; }
.temp-axis { color: var(--muted, #aaa); }

.rain-bar { fill: #4a90d9; opacity: 0.55; }
.temp-band { fill: rgba(230, 80, 50, 0.1); }
.temp-max-line { stroke: #e65032; stroke-width: 1.6; stroke-linejoin: round; }
.temp-min-line { stroke: #4a90d9; stroke-width: 1.6; stroke-linejoin: round; stroke-dasharray: 4 2; }
.zero-line { stroke: #aaa; stroke-width: 0.8; stroke-dasharray: 3 3; }

.x-axis { display: flex; justify-content: space-between; font-size: 0.58rem; color: var(--muted, #aaa); margin-top: 2px; }
.x-axis span { flex: 1; text-align: center; }

.legend { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.legend-item { display: flex; align-items: center; gap: 0.25rem; font-size: 0.63rem; color: var(--muted, #888); }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.rain-dot { background: #4a90d9; opacity: 0.7; }
.temp-hi-dot { background: #e65032; }
.temp-lo-dot { background: #4a90d9; }
.legend-val { margin-left: auto; font-size: 0.68rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }
</style>
