<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
          <text v-for="(c, j) in sortedCols" :key="`c${j}`" :x="cx(j) + cw / 2" :y="padT - 4" class="lbl lbl-col" :style="{ fontSize: `${labelFontSize}px` }">{{ compactLabel(c, Math.max(6, 12 - Math.max(0, sortedCols.length - 8))) }}</text>
          <text v-for="(r, i) in rows" :key="`r${i}`" :x="padL - 6" :y="cy(i) + ch / 2 + 3" class="lbl lbl-row" :style="{ fontSize: `${labelFontSize}px` }">{{ compactLabel(r, Math.max(8, 16 - Math.max(0, props.rows.length - 6))) }}</text>

          <template v-for="(r, i) in rows">
            <g v-for="(c, j) in sortedCols" :key="`${i}-${j}`">
              <rect :x="cx(j)" :y="cy(i)" :width="cw - 2" :height="ch - 2" rx="2"
                    :fill="cellColor(sortedMatrix[i][j])" class="cell"
                    @mouseenter="active = { r, c, v: sortedMatrix[i][j] }" />
              <text v-if="showValues" :x="cx(j) + cw / 2" :y="cy(i) + ch / 2 + 3"
                    class="cell-val" :fill="textColor(sortedMatrix[i][j])">{{ cellText(sortedMatrix[i][j]) }}</text>
            </g>
          </template>
        </svg>
      </div>

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

// Compute sorted column order (chronological months) and reorder matrix accordingly.
const sortedIndices = computed(() => {
  // Determine ordering based on month names or numeric values.
  const cols = props.cols || []
  if (cols.length === 0) return []
  const first = cols[0]
  // If strings, try month name order.
  if (typeof first === 'string') {
    const monthOrder = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    return cols.map((c, idx) => ({ c, idx })).sort((a, b) => {
      const ia = monthOrder.indexOf(a.c)
      const ib = monthOrder.indexOf(b.c)
      if (ia === -1 && ib === -1) return 0
      if (ia === -1) return 1
      if (ib === -1) return -1
      return ia - ib
    }).map(item => item.idx)
  }
  // Assume numeric month values.
  return cols.map((c, idx) => ({ c, idx })).sort((a, b) => a.c - b.c).map(item => item.idx)
})

const sortedCols = computed(() => {
  const idx = sortedIndices.value
  return idx.map(i => props.cols[i])
})

const sortedMatrix = computed(() => {
  const idx = sortedIndices.value
  return props.matrix.map(row => idx.map(i => row[i]))
})

function compactLabel(label, maxLen = 16) {
  const value = String(label ?? '')
  if (value.length <= maxLen) return value
  return `${value.slice(0, Math.max(0, maxLen - 1)).trimEnd()}…`
}

const labelFontSize = computed(() => {
  const n = Math.max(props.rows.length || 1, sortedCols.value.length || 1)
  return Math.max(8, 10 - Math.max(0, n - 8) * 0.35)
})
const W = computed(() => Math.max(640, (props.cols.length || 1) * 90 + 180))
const padL = computed(() => Math.max(90, 130 - Math.min(28, Math.max(0, (props.rows.length || 1) - 5) * 4)))
const padR = 12
const padT = 26
const padB = 8
const ch = 30

const H = computed(() => padT + padB + props.rows.length * ch)
const cw = computed(() => (W.value - padL.value - padR) / Math.max(1, sortedCols.value.length))
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const dragStart = ref(null)
const viewportStyle = computed(() => ({
  width: `${W.value}px`,
  height: `${H.value}px`,
  transform: `translate(${pan.value.x}px, ${pan.value.y}px) scale(${zoom.value})`,
  transformOrigin: '0 0',
  transition: dragStart.value ? 'none' : 'transform 0.15s ease-out',
}))

const cx = (j) => padL.value + j * cw.value
const cy = (i) => padT + i * ch

const flat = computed(() => props.matrix.flat().filter((v) => Number.isFinite(v)))
const lo = computed(() => (flat.value.length ? Math.min(...flat.value) : 0))
const hi = computed(() => (flat.value.length ? Math.max(...flat.value) : 1))

// Sequential single-hue ramp (light → dark blue).
function cellColor(v) {
  if (!Number.isFinite(v)) return 'var(--surface-2)'
  const t = (v - lo.value) / ((hi.value - lo.value) || 1)
  const a = [232, 241, 251], b = [11, 61, 145]
  const c = a.map((ch2, k) => Math.round(ch2 + (b[k] - ch2) * t))
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`
}
function textColor(v) {
  if (!Number.isFinite(v)) return 'var(--muted)'
  const t = (v - lo.value) / ((hi.value - lo.value) || 1)
  return t > 0.55 ? '#fff' : 'var(--text)'
}
function cellText(v) { return Number.isFinite(v) ? props.format(v) : ': ' }

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}
function clamp(v, min, max) { return Math.min(max, Math.max(min, v)) }
function onWheel(e) {
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  zoom.value = clamp(zoom.value * delta, 0.7, 2.5)
}
function onPointerDown(e) {
  dragStart.value = { x: e.clientX, y: e.clientY, panX: pan.value.x, panY: pan.value.y }
  e.currentTarget.setPointerCapture?.(e.pointerId)
}
function onPointerMove(e) {
  if (!dragStart.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  pan.value = { x: dragStart.value.panX + dx / zoom.value, y: dragStart.value.panY + dy / zoom.value }
}
function onPointerUp() { dragStart.value = null }
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area {
  position: relative; overflow: auto; max-width: 100%; border-radius: 8px;
  background: linear-gradient(180deg, rgba(148, 163, 184, 0.03), rgba(148, 163, 184, 0.01));
  cursor: grab; user-select: none; touch-action: none;
}
.chart-area:active { cursor: grabbing; }
.chart-viewport {
  position: relative; display: block; min-width: 100%; min-height: 100%;
}
svg { width: 100%; height: 100%; display: block; }
.lbl { fill: var(--text); font-size: 10px; }
.lbl-col { text-anchor: middle; }
.lbl-row { text-anchor: end; }
.cell { stroke: var(--surface); stroke-width: 2; }
.cell-val { font-size: 10px; text-anchor: middle; font-variant-numeric: tabular-nums; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
</style>
