<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="areaEl" class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
          <g v-for="t in xTicks" :key="t.v">
            <line :x1="t.p" :y1="padT" :x2="t.p" :y2="H - padB" class="grid" />
            <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
          </g>

          <g v-for="(b, i) in boxes" :key="i" class="boxrow" @mouseenter="active = b" @click="onBoxTap($event, b)">
            <rect class="hit" x="0" :y="b.cy - rowH / 2" :width="W" :height="rowH" />
            <text :x="padL - 8" :y="b.cy + 4" class="tick tick-y" :style="{ fontSize: `${labelFontSize}px` }">{{ b.short }}</text>
            <line :x1="b.min" :y1="b.cy" :x2="b.max" :y2="b.cy" class="whisker" />
            <line :x1="b.min" :y1="b.cy - 5" :x2="b.min" :y2="b.cy + 5" class="cap" />
            <line :x1="b.max" :y1="b.cy - 5" :x2="b.max" :y2="b.cy + 5" class="cap" />
            <rect :x="b.q1" :y="b.cy - bh / 2" :width="Math.max(1, b.q3 - b.q1)" :height="bh" rx="2"
                  :fill="b.color" class="box" />
            <line :x1="b.med" :y1="b.cy - bh / 2" :x2="b.med" :y2="b.cy + bh / 2" class="median" />
          </g>

          <text :x="(padL + W - padR) / 2" :y="H - 2" class="axis-label">{{ xLabel }}</text>
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.label }} (n={{ active.n }})</strong>
        <span>median {{ fmt(active.medVal) }}</span>
        <span>{{ fmt(active.q1Val) }} – {{ fmt(active.q3Val) }} (IQR)</span>
      </div>
    </div>

    <!-- Colour key: full category labels (the y-axis labels are truncated). -->
    <div v-if="showKey && boxes.length" class="boxkey">
      <span v-for="b in boxes" :key="b.label" class="k" :class="{ on: active && active.label === b.label }"
            @click="active = b">
        <span class="sw" :style="{ background: b.color }"></span>{{ b.label }}
      </span>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'
import { boundsFor, clampDomain } from '~/composables/useChartFields'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, values: number[], color? }]
  data: { type: Array, required: true },
  xLabel: { type: String, default: '' },
  // Field key (and optional explicit override) so the axis can be clamped to
  // the range the quantity can actually take.
  valueKey: { type: String, default: '' },
  bounds: { type: Array, default: null },
  format: {
    type: Function,
    default: (v) => {
      if (!Number.isFinite(Number(v))) return ''
      const num = Number(v)
      if (Number.isInteger(num)) return Math.round(num).toLocaleString()
      return Math.abs(num) < 10 ? num.toFixed(2) : num.toFixed(1)
    },
  },
  showKey: { type: Boolean, default: true },
})

const labelFontSize = computed(() => {
  const n = props.data.length || 1
  return Math.max(8, 10 - Math.max(0, n - 6) * 0.4)
})
const W = 640
const padL = computed(() => Math.max(90, 130 - Math.min(36, Math.max(0, (props.data.length || 1) - 5) * 3)))
const padR = 24
const padT = 12
const padB = 34
const bh = 14
const rowH = 28

const H = computed(() => padT + padB + Math.max(1, props.data.length) * rowH)
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const dragStart = ref(null)
const viewportStyle = computed(() => ({
  width: `${W}px`,
  height: `${H.value}px`,
  transform: `translate(${pan.value.x}px, ${pan.value.y}px) scale(${zoom.value})`,
  transformOrigin: '0 0',
  transition: dragStart.value ? 'none' : 'transform 0.15s ease-out',
}))

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
const areaEl = ref(null)
const dragged = ref(false)
function compactLabel(label, maxLen = 18) {
  const value = String(label ?? '')
  if (value.length <= maxLen) return value
  return `${value.slice(0, Math.max(0, maxLen - 1)).trimEnd()}…`
}
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}
// Tap/click a box shows its stats — the info that hover gives on desktop, made
// reachable on touch devices.
function onBoxTap(e, b) {
  if (dragged.value) return
  const r = areaEl.value?.getBoundingClientRect()
  if (r) ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
  active.value = b
}
function clamp(v, min, max) { return Math.min(max, Math.max(min, v)) }
function onWheel(e) {
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  zoom.value = clamp(zoom.value * delta, 0.7, 2.5)
}
function onPointerDown(e) {
  dragStart.value = { x: e.clientX, y: e.clientY, panX: pan.value.x, panY: pan.value.y }
  dragged.value = false
  // No setPointerCapture — it would retarget the tap off the box and break
  // tap-to-show-info. Panning still works via the move handler below.
}
function onPointerMove(e) {
  if (!dragStart.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  if (Math.abs(dx) + Math.abs(dy) > 4) dragged.value = true
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
  // Clamp the padding to what the quantity can be: a day-of-year axis must not
  // run past 365, and an elevation axis must not go negative.
  return clampDomain([lo - pad, hi + pad], props.bounds ?? boundsFor(props.valueKey), all)
})
const sx = (v) => padL.value + ((v - domain.value[0]) / (domain.value[1] - domain.value[0] || 1)) * (W - padL.value - padR)

const boxes = computed(() => stats.value.map((d, i) => ({
  ...d, cy: padT + i * rowH + rowH / 2,
  min: sx(d.minVal), max: sx(d.maxVal), q1: sx(d.q1Val), q3: sx(d.q3Val), med: sx(d.medVal),
})))

function ticks() {
  const [lo, hi] = domain.value
  const span = hi - lo
  const out = []
  for (let i = 0; i <= 4; i++) {
    const v = lo + (span * i) / 4
    out.push({ v, p: sx(v), label: props.format(v) })
  }
  return out
}
const xTicks = computed(ticks)
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
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; fill: var(--text); }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.whisker { stroke: var(--muted); stroke-width: 1.5; }
.cap { stroke: var(--muted); stroke-width: 1.5; }
.box { stroke: var(--text); stroke-opacity: 0.15; }
.median { stroke: #fff; stroke-width: 2; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }

.boxrow { cursor: pointer; }
.hit { fill: transparent; pointer-events: all; }
.boxkey {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px;
  font-size: 0.75rem; color: var(--text); max-height: 92px; overflow-y: auto;
}
.boxkey .k { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; opacity: 0.9; }
.boxkey .k:hover, .boxkey .k.on { opacity: 1; font-weight: 600; }
.boxkey .sw { width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
