<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area" @mousemove="onMove" @mouseleave="active = null">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
        <!-- gridlines + axis ticks -->
        <g v-for="t in yTicks" :key="`y${t.v}`">
          <line :x1="padL" :y1="t.p" :x2="W - padR" :y2="t.p" class="grid" />
          <text :x="padL - 6" :y="t.p + 3" class="tick tick-y">{{ t.label }}</text>
        </g>
        <g v-for="t in xTicks" :key="`x${t.v}`">
          <line :x1="t.p" :y1="padT" :x2="t.p" :y2="H - padB" class="grid" />
          <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
        </g>

        <!-- points -->
        <circle v-for="(pt, i) in scaled" :key="i" :cx="pt.cx" :cy="pt.cy" r="4"
                :fill="pt.color" class="dot" @mouseenter="active = pt" />

        <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
        <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>
      </svg>

      <div v-if="legend && legend.length" class="legend">
        <span v-for="l in legend" :key="l.label" class="lg"><span class="sw" :style="{ background: l.color }"></span>{{ l.label }}</span>
      </div>
      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong v-if="active.label">{{ active.label }}</strong>
        <span>{{ xLabel }}: {{ xFormat(active.x) }}</span>
        <span>{{ yLabel }}: {{ yFormat(active.y) }}</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ x, y, color?, label? }]
  data: { type: Array, required: true },
  xLabel: { type: String, default: 'x' },
  yLabel: { type: String, default: 'y' },
  xFormat: { type: Function, default: (v) => `${Math.round(v)}` },
  yFormat: { type: Function, default: (v) => `${Math.round(v)}` },
  legend: { type: Array, default: () => [] },
})

const W = 640
const H = 360
const padL = 52
const padR = 16
const padT = 12
const padB = 34

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  // Convert to the same coordinate space the tooltip uses (CSS px of the container).
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

function domain(vals) {
  if (!vals.length) return [0, 1]
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.05
  return [lo - pad, hi + pad]
}

const points = computed(() => props.data.filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y)))
const xDom = computed(() => domain(points.value.map((d) => d.x)))
const yDom = computed(() => domain(points.value.map((d) => d.y)))

const sx = (v) => padL + ((v - xDom.value[0]) / (xDom.value[1] - xDom.value[0])) * (W - padL - padR)
const sy = (v) => (H - padB) - ((v - yDom.value[0]) / (yDom.value[1] - yDom.value[0])) * (H - padT - padB)

const scaled = computed(() => points.value.map((d) => ({
  ...d, cx: sx(d.x), cy: sy(d.y), color: d.color || SERIES_1,
})))

function ticks(dom, fmt, toPix) {
  const [lo, hi] = dom
  const n = 4
  const out = []
  for (let i = 0; i <= n; i++) {
    const v = lo + ((hi - lo) * i) / n
    out.push({ v, p: toPix(v), label: fmt(v) })
  }
  return out
}
const xTicks = computed(() => ticks(xDom.value, props.xFormat, sx))
const yTicks = computed(() => ticks(yDom.value, props.yFormat, sy))
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: #1f2933; margin-bottom: 6px; }
.chart-area { position: relative; }
svg { width: 100%; height: auto; display: block; }

.grid { stroke: #eef0f2; stroke-width: 1; }
.tick { fill: #9aa0a6; font-size: 10px; }
.tick-y { text-anchor: end; }
.tick-x { text-anchor: middle; }
.axis-label { fill: #6b7280; font-size: 11px; text-anchor: middle; }
.dot { stroke: #fff; stroke-width: 1; opacity: 0.9; }
.dot:hover { stroke: #1f2933; stroke-width: 1.5; }

.legend { position: absolute; top: 4px; right: 8px; display: flex; flex-wrap: wrap; gap: 4px 10px; font-size: 0.72rem; color: #4b5563; }
.legend .lg { display: inline-flex; align-items: center; gap: 4px; }
.legend .sw { width: 10px; height: 10px; border-radius: 50%; border: 1px solid #cbd2d9; }

.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: #1f2933; color: #fff; padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
</style>
