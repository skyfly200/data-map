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

        <g v-for="s in scaledSeries" :key="s.label">
          <polyline :points="s.polyline" class="line" :style="{ stroke: s.color, color: s.color }" />
          <circle v-for="(pt, i) in s.points" :key="i" :cx="pt.cx" :cy="pt.cy" r="3" class="dot" :style="{ fill: s.color, color: s.color }" />
        </g>

        <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
        <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>
      </svg>

      <div v-if="showLegend" class="legend">
        <span v-for="s in scaledSeries" :key="`lg-${s.label}`" class="lg">
          <span class="sw" :style="{ background: s.color }"></span>{{ s.label }}
        </span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  // Single series: [{ x, y }] ordered by x. Or multi-series via `series`.
  data: { type: Array, default: () => [] },
  // [{ label, color, data: [{ x, y }] }] — one line each.
  series: { type: Array, default: () => [] },
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

// Normalise to a list of series so single- and multi-line share one path.
const allSeries = computed(() => (props.series.length
  ? props.series
  : [{ label: '', color: props.color, data: props.data }])
  .map((s) => ({ ...s, data: (s.data || []).filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y)) }))
  .filter((s) => s.data.length))
const showLegend = computed(() => props.series.length > 1)

function domain(vals) {
  if (!vals.length) return [0, 1]
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (lo === hi) { lo -= 1; hi += 1 }
  return [lo, hi]
}
const allPoints = computed(() => allSeries.value.flatMap((s) => s.data))
const xDom = computed(() => domain(allPoints.value.map((d) => d.x)))
const yDom = computed(() => {
  const [lo, hi] = domain(allPoints.value.map((d) => d.y))
  // Anchor the value axis at zero when the data is non-negative (counts, means).
  return [Math.min(0, lo), hi + (hi - lo) * 0.05]
})

const sx = (v) => padL + ((v - xDom.value[0]) / (xDom.value[1] - xDom.value[0] || 1)) * (W.value - padL - padR)
const sy = (v) => (H.value - padB) - ((v - yDom.value[0]) / (yDom.value[1] - yDom.value[0] || 1)) * (H.value - padT - padB)

const scaledSeries = computed(() => allSeries.value.map((s) => {
  const points = s.data.map((d) => ({ cx: sx(d.x), cy: sy(d.y) }))
  return { label: s.label, color: s.color || SERIES_1, points, polyline: points.map((p) => `${p.cx},${p.cy}`).join(' ') }
}))

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

.legend {
  position: absolute; top: 4px; right: 8px; display: flex; flex-wrap: wrap; gap: 4px 10px;
  font-size: 0.72rem; color: var(--text); max-width: 62%; justify-content: flex-end;
  background: color-mix(in srgb, var(--surface) 72%, transparent); padding: 3px 6px; border-radius: 6px;
}
.legend .lg { display: inline-flex; align-items: center; gap: 4px; }
.legend .sw { width: 10px; height: 10px; border-radius: 50%; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
