<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="container" class="chart-area">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
        <g v-for="t in yTicks" :key="`y${t.v}`">
          <line :x1="padL" :y1="t.p" :x2="W - padR" :y2="t.p" class="grid" />
          <text :x="padL - 6" :y="t.p + 3" class="tick tick-y">{{ t.label }}</text>
        </g>
        <g v-for="t in xTicks" :key="`x${t.v}`">
          <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
        </g>

        <polyline :points="polyline" class="line" :style="{ stroke: color, color }" />
        <circle v-for="(pt, i) in scaled" :key="i" :cx="pt.cx" :cy="pt.cy" r="3" class="dot" :style="{ fill: color, color }" />

        <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
        <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>
      </svg>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  data: { type: Array, required: true }, // [{ x, y }] ordered by x
  xLabel: { type: String, default: 'x' },
  yLabel: { type: String, default: 'y' },
  xFormat: { type: Function, default: (v) => `${Math.round(v)}` },
  yFormat: { type: Function, default: (v) => `${Math.round(v)}` },
  color: { type: String, default: SERIES_1 },
})

const { container, width: W, height: H } = useChartSize()
const padL = 52
const padR = 16
const padT = 12
const padB = 34

const points = computed(() => props.data.filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y)))

function domain(vals) {
  if (!vals.length) return [0, 1]
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (lo === hi) { lo -= 1; hi += 1 }
  return [lo, hi]
}
const xDom = computed(() => domain(points.value.map((d) => d.x)))
const yDom = computed(() => {
  const [lo, hi] = domain(points.value.map((d) => d.y))
  // Anchor the value axis at zero when the data is non-negative (counts, means).
  return [Math.min(0, lo), hi + (hi - lo) * 0.05]
})

const sx = (v) => padL + ((v - xDom.value[0]) / (xDom.value[1] - xDom.value[0] || 1)) * (W.value - padL - padR)
const sy = (v) => (H.value - padB) - ((v - yDom.value[0]) / (yDom.value[1] - yDom.value[0] || 1)) * (H.value - padT - padB)

const scaled = computed(() => points.value.map((d) => ({ cx: sx(d.x), cy: sy(d.y) })))
const polyline = computed(() => scaled.value.map((p) => `${p.cx},${p.cy}`).join(' '))

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
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area { position: relative; height: 100%; min-height: 260px; overflow: hidden; }
svg { position: absolute; inset: 0; width: 100%; height: 100%; display: block; }
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-y { text-anchor: end; }
.tick-x { text-anchor: middle; }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.line { fill: none; stroke-width: 2; filter: var(--chart-glow); }
.dot { stroke: var(--surface); stroke-width: 1; filter: var(--chart-glow); }
</style>
