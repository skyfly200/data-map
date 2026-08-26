<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="container" class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport" :style="viewportStyle">
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

          <!-- points (path so shape can vary; radius can vary too) -->
          <path v-for="(pt, i) in scaled" :key="i" :d="pt.d"
                :fill="pt.color" :style="{ color: pt.color }" class="dot"
                :class="{ selectable: pt.obs }" @mouseenter="active = pt"
                @click="onDotClick(pt)" />

          <g v-if="todayX !== null && Number.isFinite(todayX)">
            <line :x1="sx(todayX)" :y1="padT" :x2="sx(todayX)" :y2="H - padB" class="today-line" />
            <text :x="sx(todayX) + 4" :y="padT + 12" class="today-label">{{ todayLabel }}</text>
          </g>

          <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
          <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>
        </svg>
      </div>

      <div v-if="(legend && legend.length) || (shapeLegend && shapeLegend.length) || sizeLegend" class="legend">
        <template v-if="legend && legend.length">
          <span v-for="l in legend" :key="l.label" class="lg"><span class="sw" :style="{ background: l.color }"></span>{{ l.label }}</span>
        </template>
        <template v-if="shapeLegend && shapeLegend.length">
          <span class="lg lg-sep" v-if="legend && legend.length"></span>
          <span v-for="s in shapeLegend" :key="`sh-${s.label}`" class="lg">
            <svg class="sw-shape" viewBox="-6 -6 12 12"><path :d="glyph(s.shape)" fill="var(--muted)" /></svg>{{ s.label }}
          </span>
        </template>
        <span v-if="sizeLegend" class="lg lg-size">
          <svg class="sw-shape" viewBox="-6 -6 12 12"><circle r="2.5" fill="var(--muted)" /></svg>
          <svg class="sw-shape" viewBox="-6 -6 12 12"><circle r="5" fill="var(--muted)" /></svg>
          {{ sizeLegend }}
        </span>
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
  shapeLegend: { type: Array, default: () => [] }, // [{ label, shape }]
  sizeLegend: { type: String, default: '' },       // e.g. "Slope (small→large)"
  todayX: { type: Number, default: null },
  todayLabel: { type: String, default: 'Today' },
})
const emit = defineEmits(['select'])

// SVG path for a mark of radius r centered at (cx, cy). Six distinguishable
// shapes; `glyph` is the same centered at the origin for legend swatches.
function shapePath(shape, cx, cy, r) {
  const a = r * 0.6
  switch (shape) {
    case 'square': return `M${cx - r},${cy - r}h${2 * r}v${2 * r}h${-2 * r}z`
    case 'triangle': return `M${cx},${cy - r}L${cx + r},${cy + r}L${cx - r},${cy + r}z`
    case 'diamond': return `M${cx},${cy - r}L${cx + r},${cy}L${cx},${cy + r}L${cx - r},${cy}z`
    case 'cross': return `M${cx - a},${cy - r}h${2 * a}v${r - a}h${r - a}v${2 * a}h${-(r - a)}v${r - a}h${-2 * a}v${-(r - a)}h${-(r - a)}v${-2 * a}h${r - a}z`
    case 'wye': return `M${cx - a},${cy + r}l${a},${-r}l${-r},${-a}l${a * 0.6},${-a * 0.9}l${r - a * 0.6},${a}l${r - a * 0.6},${-a}l${a * 0.6},${a * 0.9}l${-r},${a}l${a},${r}z`
    default: return `M${cx - r},${cy}a${r},${r} 0 1,0 ${2 * r},0a${r},${r} 0 1,0 ${-2 * r},0z` // circle
  }
}
function glyph(shape) { return shapePath(shape, 0, 0, 5) }

const { container, width: W, height: H } = useChartSize()
const padL = 52
const padR = 16
const padT = 12
const padB = 34
const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const dragStart = ref(null)
const viewportStyle = computed(() => ({
  transform: `translate(${pan.value.x}px, ${pan.value.y}px) scale(${zoom.value})`,
  transformOrigin: '0 0',
  transition: dragStart.value ? 'none' : 'transform 0.15s ease-out',
}))

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
const dragged = ref(false)
function onPointerDown(e) {
  dragStart.value = { x: e.clientX, y: e.clientY, panX: pan.value.x, panY: pan.value.y }
  dragged.value = false
  // NB: no setPointerCapture — capturing would retarget the click off the point
  // and break click-to-select. Panning still works via the move handler below.
}
function onPointerMove(e) {
  if (!dragStart.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  if (Math.abs(dx) + Math.abs(dy) > 4) dragged.value = true
  pan.value = { x: dragStart.value.panX + dx / zoom.value, y: dragStart.value.panY + dy / zoom.value }
}
function onPointerUp() { dragStart.value = null }
function onDotClick(pt) {
  // Ignore the click that ends a pan-drag; only a genuine click selects.
  if (!dragged.value && pt.obs) emit('select', pt.obs)
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

const sx = (v) => padL + ((v - xDom.value[0]) / (xDom.value[1] - xDom.value[0])) * (W.value - padL - padR)
const sy = (v) => (H.value - padB) - ((v - yDom.value[0]) / (yDom.value[1] - yDom.value[0])) * (H.value - padT - padB)

const scaled = computed(() => points.value.map((d) => {
  const cx = sx(d.x), cy = sy(d.y), r = Number.isFinite(d.r) ? d.r : 4
  return { ...d, cx, cy, color: d.color || SERIES_1, d: shapePath(d.shape || 'circle', cx, cy, r) }
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
.chart-viewport { position: absolute; inset: 0; }
svg { width: 100%; height: 100%; display: block; }

.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-y { text-anchor: end; }
.tick-x { text-anchor: middle; }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.dot { stroke: var(--surface); stroke-width: 1; opacity: 0.9; filter: var(--chart-glow); }
.dot:hover { stroke: var(--text); stroke-width: 1.5; }
.dot.selectable { cursor: pointer; }
.today-line { stroke: #b00020; stroke-width: 1.5; stroke-dasharray: 4 4; }
.today-label { fill: #b00020; font-size: 10px; font-weight: 600; }

.legend {
  position: absolute; top: 4px; right: 8px; display: flex; flex-wrap: wrap; gap: 4px 10px;
  font-size: 0.72rem; color: var(--text); max-width: 60%; justify-content: flex-end;
  background: color-mix(in srgb, var(--surface) 72%, transparent); padding: 3px 6px; border-radius: 6px;
}
.legend .lg { display: inline-flex; align-items: center; gap: 4px; }
.legend .sw { width: 10px; height: 10px; border-radius: 50%; border: 1px solid var(--border); }
.legend .sw-shape { width: 12px; height: 12px; flex: 0 0 auto; }
.legend .lg-sep { border-left: 1px solid var(--border); align-self: stretch; }
.legend .lg-size { gap: 2px; }

.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
</style>
