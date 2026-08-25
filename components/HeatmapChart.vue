<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area" @mousemove="onMove" @mouseleave="active = null">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
        <text v-for="(c, j) in cols" :key="`c${j}`" :x="cx(j) + cw / 2" :y="padT - 4" class="lbl lbl-col">{{ c }}</text>
        <text v-for="(r, i) in rows" :key="`r${i}`" :x="padL - 6" :y="cy(i) + ch / 2 + 3" class="lbl lbl-row">{{ r }}</text>

        <template v-for="(r, i) in rows">
          <g v-for="(c, j) in cols" :key="`${i}-${j}`">
            <rect :x="cx(j)" :y="cy(i)" :width="cw - 2" :height="ch - 2" rx="2"
                  :fill="cellColor(matrix[i][j])" class="cell"
                  @mouseenter="active = { r, c, v: matrix[i][j] }" />
            <text v-if="showValues" :x="cx(j) + cw / 2" :y="cy(i) + ch / 2 + 3"
                  class="cell-val" :fill="textColor(matrix[i][j])">{{ cellText(matrix[i][j]) }}</text>
          </g>
        </template>
      </svg>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.r }} · {{ active.c }}</strong>
        <span>{{ cellText(active.v) }}</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
const props = defineProps({
  title: { type: String, default: '' },
  rows: { type: Array, required: true },   // row labels
  cols: { type: Array, required: true },   // column labels
  matrix: { type: Array, required: true }, // rows × cols numbers (null allowed)
  format: { type: Function, default: (v) => `${Math.round(v)}` },
  showValues: { type: Boolean, default: true },
})

const W = 640
const padL = 130
const padR = 12
const padT = 26
const padB = 8
const ch = 30

const H = computed(() => padT + padB + props.rows.length * ch)
const cw = computed(() => (W - padL - padR) / Math.max(1, props.cols.length))

const cx = (j) => padL + j * cw.value
const cy = (i) => padT + i * ch

const flat = computed(() => props.matrix.flat().filter((v) => Number.isFinite(v)))
const lo = computed(() => (flat.value.length ? Math.min(...flat.value) : 0))
const hi = computed(() => (flat.value.length ? Math.max(...flat.value) : 1))

// Sequential single-hue ramp (light → dark blue).
function cellColor(v) {
  if (!Number.isFinite(v)) return '#f3f4f6'
  const t = (v - lo.value) / ((hi.value - lo.value) || 1)
  const a = [232, 241, 251], b = [11, 61, 145]
  const c = a.map((ch2, k) => Math.round(ch2 + (b[k] - ch2) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}
function textColor(v) {
  if (!Number.isFinite(v)) return '#9aa0a6'
  const t = (v - lo.value) / ((hi.value - lo.value) || 1)
  return t > 0.55 ? '#fff' : '#1f2933'
}
function cellText(v) { return Number.isFinite(v) ? props.format(v) : '—' }

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: #1f2933; margin-bottom: 6px; }
.chart-area { position: relative; }
svg { width: 100%; height: auto; display: block; }
.lbl { fill: #4b5563; font-size: 10px; }
.lbl-col { text-anchor: middle; }
.lbl-row { text-anchor: end; }
.cell { stroke: #fff; stroke-width: 2; }
.cell-val { font-size: 10px; text-anchor: middle; font-variant-numeric: tabular-nums; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: #1f2933; color: #fff; padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
</style>
