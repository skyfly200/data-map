<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
          <text v-for="(r, i) in rows" :key="`r${i}`" :x="padL - 6" :y="cy(i) + ch / 2 + 3" class="lbl lbl-row" :style="{ fontSize: `${labelFontSize}px` }">{{ compactLabel(r) }}</text>
          <text v-for="(c, j) in cols" :key="`c${j}`" :x="cx(j) + cw / 2" :y="padT - 4" class="lbl lbl-col" :style="{ fontSize: `${labelFontSize}px` }">{{ compactLabel(c) }}</text>

          <template v-for="(r, i) in rows">
            <g v-for="(c, j) in cols" :key="`${i}-${j}`">
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
        <strong v-if="active.r">{{ active.r }}</strong> <span v-if="active.c">· {{ active.c }}</span>
        <span v-if="active.v">{{ cellText(active.v) }}</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { matrixCellColor, matrixTextColor, matrixCellCoord } from '~/composables/maxentViz'

const props = defineProps({
  title: { type: String, default: '' },
  rows: { type: Array, required: true },   // row labels
  cols: { type: Array, required: true },   // column labels
  matrix: { type: Array, required: true }, // rows × cols numbers
  format: { type: Function, default: (v) => String(v) },
  showValues: { type: Boolean, default: true },
})

// Logic for sorting/calculating colors shared with HeatmapChart.vue
const sortedIndices = computed(() => {
  const cols = props.cols || []
  if (cols.length === 0) return []
  return cols.map((c, idx) => ({ c, idx })).sort((a, b) => a.c.localeCompare(b.c)).map(item => item.idx)
})
const sortedCols = computed(() => sortedIndices.value.map(i => props.cols[i]))
const sortedMatrix = computed(() => props.matrix.map(row => sortedIndices.value.map(i => row[i])))

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

const flat = computed(() => props.matrix.flat().filter((v) => Number.isFinite(v)))
const lo = computed(() => (flat.value.length ? Math.min(...flat.value) : 0))
const hi = computed(() => (flat.value.length ? Math.max(...flat.value) : 1))

function cellColor(v) {
  return matrixCellColor(v, lo.value, hi.value)
}
function textColor(v) {
  return matrixTextColor(v, lo.value, hi.value)
}
function cellText(v) { return props.format(v) }

const cx = (j) => {
  const coord = matrixCellCoord(0, j, props.rows.length, props.cols.length, { 
    W: W.value, H: H.value, padL: padL.value, padR, padT, padB 
  })
  return coord.cx
}
const cy = (i) => {
  const coord = matrixCellCoord(i, 0, props.rows.length, props.cols.length, { 
    W: W.value, H: H.value, padL: padL.value, padR, padT, padB 
  })
  return coord.cy
}

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

const viewportStyle = computed(() => ({
  width: `${W.value}px`,
  height: `${H.value}px`,
  transform: 'none',
  transition: 'none',
}))
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area {
  position: relative; overflow: auto; max-width: 100%; border-radius: 8px;
  background: linear-gradient(180deg, rgba(148, 163, 184, 0.03), rgba(148, 163, 184, 0.01));
  cursor: crosshair;
}
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
