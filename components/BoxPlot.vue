<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
          <g v-for="t in xTicks" :key="t.v">
            <line :x1="t.p" :y1="padT" :x2="t.p" :y2="H - padB" class="grid" />
            <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
          </g>

          <g v-for="(b, i) in boxes" :key="i" @mouseenter="active = b">
            <text :x="padL.value - 8" :y="b.cy + 4" class="tick tick-y" :style="{ fontSize: `${labelFontSize}px` }">{{ b.short }}</text>
            <line :x1="b.min" :y1="b.cy" :x2="b.max" :y2="b.cy" class="whisker" />
            <line :x1="b.min" :y1="b.cy - 5" :x2="b.min" :y2="b.cy + 5" class="cap" />
            <line :x1="b.max" :y1="b.cy - 5" :x2="b.max" :y2="b.cy + 5" class="cap" />
            <rect :x="b.q1" :y="b.cy - bh / 2" :width="Math.max(1, b.q3 - b.q1)" :height="bh" rx="2"
                  :fill="b.color" class="box" />
            <line :x1="b.med" :y1="b.cy - bh / 2" :x2="b.med" :y2="b.cy + bh / 2" class="median" />
          </g>

          <text :x="(padL.value + W.value - padR) / 2" :y="H - 2" class="axis-label">{{ xLabel }}</text>
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.label }} (n={{ active.n }})</strong>
        <span>median {{ fmt(active.medVal) }}</span>
        <span>{{ fmt(active.q1Val) }} – {{ fmt(active.q3Val) }} (IQR)</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, values: number[], color? }]
  data: { type: Array, required: true },
  xLabel: { type: String, default: '' },
  format: { type: Function, default: (v) => `${Math.round(v)}` },
})

const labelFontSize = computed(() => {
  const n = props.data.length || 1
  return Math.max(8, 10 - Math.max(0, n - 6) * 0.4)
})
const W = computed(() => Math.max(640, (props.data.length || 1) * 120 + 180))
const padL = computed(() => Math.max(82, 118 - Math.min(28, Math.max(0, (props.data.length || 1) - 5) * 4)))
const padR = 20
const padT = 10
const padB = 30
const bh = 16
const rowH = 30

const H = computed(() => padT + padB + Math.max(1, props.data.length) * rowH)
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

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function compactLabel(label, maxLen = 18) {
  const value = String(label ?? '')
  if (value.length <= maxLen) return value
  return `${value.slice(0, Math.max(0, maxLen - 1)).trimEnd()}…`
}
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
const fmt = (v) => props.format(v)

function quantile(sorted, p) {
  const idx = (sorted.length - 1) * p
  const lo = Math.floor(idx), hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

const stats = computed(() => props.data
  .map((d) => ({ ...d, values: (d.values || []).filter((v) => Number.isFinite(v)).sort((a, b) => a - b) }))
  .filter((d) => d.values.length >= 1)
  .map((d) => ({
    label: d.label,
    short: compactLabel(d.label || d.short, Math.max(8, 18 - Math.max(0, props.data.length - 6))),
    color: d.color || SERIES_1,
    n: d.values.length,
    minVal: d.values[0],
    maxVal: d.values[d.values.length - 1],
    q1Val: quantile(d.values, 0.25),
    medVal: quantile(d.values, 0.5),
    q3Val: quantile(d.values, 0.75),
  })))

const domain = computed(() => {
  const all = stats.value.flatMap((d) => [d.minVal, d.maxVal])
  if (!all.length) return [0, 1]
  let lo = Math.min(...all), hi = Math.max(...all)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.04
  return [lo - pad, hi + pad]
})
const sx = (v) => padL.value + ((v - domain.value[0]) / (domain.value[1] - domain.value[0])) * (W.value - padL.value - padR)

const boxes = computed(() => stats.value.map((d, i) => ({
  ...d, cy: padT + i * rowH + rowH / 2,
  min: sx(d.minVal), max: sx(d.maxVal), q1: sx(d.q1Val), q3: sx(d.q3Val), med: sx(d.medVal),
})))

function ticks() {
  const [lo, hi] = domain.value
  const out = []
  for (let i = 0; i <= 4; i++) { const v = lo + ((hi - lo) * i) / 4; out.push({ v, p: sx(v), label: props.format(v) }) }
  return out
}
const xTicks = computed(ticks)
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: #1f2933; margin-bottom: 6px; }
.chart-area { position: relative; }
svg { width: 100%; height: auto; display: block; }
.grid { stroke: #eef0f2; stroke-width: 1; }
.tick { fill: #9aa0a6; font-size: 10px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; fill: #4b5563; }
.axis-label { fill: #6b7280; font-size: 11px; text-anchor: middle; }
.whisker { stroke: #9aa0a6; stroke-width: 1.5; }
.cap { stroke: #9aa0a6; stroke-width: 1.5; }
.box { stroke: #1f2933; stroke-opacity: 0.15; }
.median { stroke: #fff; stroke-width: 2; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: #1f2933; color: #fff; padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
</style>
