<template>
  <div class="dash-widget weather-corr">
    <div class="widget-head">
      <h3 class="widget-title">🌧️ Weather correlation</h3>
      <div class="widget-actions">
        <button class="toggle-btn" :class="{ active: field === 'prcp_d0' }" @click="field = 'prcp_d0'">Precip</button>
        <button class="toggle-btn" :class="{ active: field === 'ndvi' }" @click="field = 'ndvi'">NDVI</button>
        <button class="toggle-btn" :class="{ active: field === 'soil_moisture' }" @click="field = 'soil_moisture'">Soil</button>
      </div>
    </div>

    <p v-if="loading && !rows.length" class="wc-note">Loading…</p>
    <p v-else-if="!hasData" class="wc-note">No {{ fieldLabel }} data in observations.</p>

    <div v-else class="wc-body">
      <svg class="wc-svg" viewBox="0 0 260 80" preserveAspectRatio="none">
        <g>
          <rect
            v-for="(b, i) in bins"
            :key="i"
            :x="barX(i)"
            :y="80 - barH(b.count)"
            :width="barW"
            :height="barH(b.count)"
            class="wc-bar"
          />
        </g>
        <polyline
          v-if="linePts"
          :points="linePts"
          class="wc-line"
          fill="none"
        />
        <line v-if="todayX != null" :x1="todayX" y1="0" :x2="todayX" y2="80" class="wc-today" />
      </svg>

      <div class="wc-xaxis">
        <span v-for="lbl in xLabels" :key="lbl">{{ lbl }}</span>
      </div>

      <div class="wc-footer">
        <span class="wc-corr" :class="corrClass" :title="`Pearson r = ${corrValue.toFixed(3)}`">
          r = {{ corrValue.toFixed(2) }} ({{ corrLabel }})
        </span>
        <span class="wc-legend">
          <span class="wc-dot dot-bar"></span>Obs count
          <span class="wc-dot dot-line"></span>Avg {{ fieldLabel }}
        </span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()

const field = ref('prcp_d0')

const FIELD_LABELS = { prcp_d0: 'Precipitation', ndvi: 'NDVI', soil_moisture: 'Soil moisture' }
const fieldLabel = computed(() => FIELD_LABELS[field.value] || field.value)

const N_BINS = 24
const BIN_DAYS = 14
const now = Date.now()

const bins = computed(() => {
  const buckets = Array.from({ length: N_BINS }, () => ({ count: 0, fieldSum: 0, fieldN: 0 }))
  for (const r of rows.value || []) {
    if (!r.date) continue
    const d = new Date(r.date)
    if (isNaN(d)) continue
    const ageDays = (now - d.getTime()) / 86400000
    if (ageDays < 0 || ageDays > N_BINS * BIN_DAYS) continue
    const idx = N_BINS - 1 - Math.floor(ageDays / BIN_DAYS)
    if (idx < 0 || idx >= N_BINS) continue
    buckets[idx].count++
    const val = Number(r[field.value])
    if (Number.isFinite(val)) { buckets[idx].fieldSum += val; buckets[idx].fieldN++ }
  }
  return buckets.map((b) => ({
    count: b.count,
    fieldMean: b.fieldN > 0 ? b.fieldSum / b.fieldN : null,
  }))
})

const hasData = computed(() => bins.value.some((b) => b.count > 0))

const maxCount = computed(() => Math.max(1, ...bins.value.map((b) => b.count)))
const fieldVals = computed(() => bins.value.map((b) => b.fieldMean).filter((v) => v != null))
const fieldMin = computed(() => fieldVals.value.length ? Math.min(...fieldVals.value) : 0)
const fieldMax = computed(() => fieldVals.value.length ? Math.max(...fieldVals.value) : 1)
const fieldRange = computed(() => Math.max(0.001, fieldMax.value - fieldMin.value))

const W = 260
const H = 80
const barW = computed(() => W / N_BINS - 1)

function barX(i) { return i * (W / N_BINS) }
function barH(count) { return (count / maxCount.value) * (H * 0.65) }

const linePts = computed(() => {
  const pts = []
  bins.value.forEach((b, i) => {
    if (b.fieldMean == null) return
    const x = barX(i) + barW.value / 2
    const y = H - 4 - ((b.fieldMean - fieldMin.value) / fieldRange.value) * (H * 0.55)
    pts.push(`${x},${y}`)
  })
  return pts.length >= 2 ? pts.join(' ') : null
})

const todayX = computed(() => W - 1)

const xLabels = computed(() => {
  const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
  return Array.from({ length: N_BINS }, (_, i) => {
    if (i % 4 !== 0) return ''
    const ageDays = (N_BINS - i) * BIN_DAYS
    const d = new Date(now - ageDays * 86400000)
    return MONTHS[d.getMonth()]
  })
})

const corrValue = computed(() => {
  const pairs = bins.value.filter((b) => b.fieldMean != null && b.count > 0)
  if (pairs.length < 3) return 0
  const n = pairs.length
  const xs = pairs.map((b) => b.count)
  const ys = pairs.map((b) => b.fieldMean)
  const mx = xs.reduce((a, v) => a + v, 0) / n
  const my = ys.reduce((a, v) => a + v, 0) / n
  const num = xs.reduce((a, v, i) => a + (v - mx) * (ys[i] - my), 0)
  const dx = Math.sqrt(xs.reduce((a, v) => a + (v - mx) ** 2, 0))
  const dy = Math.sqrt(ys.reduce((a, v) => a + (v - my) ** 2, 0))
  return dx * dy > 0 ? num / (dx * dy) : 0
})

const corrLabel = computed(() => {
  const r = Math.abs(corrValue.value)
  if (r >= 0.7) return corrValue.value > 0 ? 'strong positive' : 'strong negative'
  if (r >= 0.4) return corrValue.value > 0 ? 'moderate positive' : 'moderate negative'
  if (r >= 0.2) return corrValue.value > 0 ? 'weak positive' : 'weak negative'
  return 'no correlation'
})

const corrClass = computed(() => {
  const r = corrValue.value
  if (r >= 0.4) return 'corr-pos'
  if (r <= -0.4) return 'corr-neg'
  return 'corr-none'
})

onMounted(() => { load() })
</script>

<style scoped>
.weather-corr { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: center; justify-content: space-between; gap: 0.4rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.3rem; }
.toggle-btn {
  font: inherit; font-size: 0.68rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #888);
  border-radius: 999px; padding: 0.12rem 0.48rem; cursor: pointer;
}
.toggle-btn.active { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }

.wc-note { font-size: 0.85rem; color: var(--muted, #888); }

.wc-body { display: flex; flex-direction: column; gap: 0.25rem; }

.wc-svg { width: 100%; height: 80px; display: block; overflow: visible; }
.wc-bar { fill: var(--accent, #2a78d6); opacity: 0.25; }
.wc-line { stroke: #e06830; stroke-width: 1.8; stroke-linejoin: round; }
.wc-today { stroke: var(--muted, #bbb); stroke-width: 1; stroke-dasharray: 3 2; }

.wc-xaxis {
  display: flex; justify-content: space-between;
  font-size: 0.6rem; color: var(--muted, #aaa);
}
.wc-xaxis span { flex: 1; text-align: left; }

.wc-footer { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.3rem; }

.wc-corr { font-size: 0.72rem; font-variant-numeric: tabular-nums; }
.corr-pos { color: #2a7a40; }
.corr-neg { color: #a03020; }
.corr-none { color: var(--muted, #888); }

.wc-legend { display: flex; align-items: center; gap: 0.35rem; font-size: 0.65rem; color: var(--muted, #888); }
.wc-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; }
.dot-bar { background: var(--accent, #2a78d6); opacity: 0.5; }
.dot-line { background: #e06830; }
</style>
