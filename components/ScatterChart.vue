<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="container" class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport">
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
          <defs>
            <clipPath :id="clipId"><rect :x="padL" :y="padT" :width="Math.max(0, W - padL - padR)" :height="Math.max(0, H - padT - padB)" /></clipPath>
          </defs>
          <!-- gridlines + axis ticks -->
          <g v-for="t in yTicks" :key="`y${t.v}`">
            <line :x1="padL" :y1="t.p" :x2="W - padR" :y2="t.p" class="grid" />
            <text :x="padL - 6" :y="t.p + 3" class="tick tick-y">{{ t.label }}</text>
          </g>
          <g v-for="t in xTicks" :key="`x${t.v}`">
            <line :x1="t.p" :y1="padT" :x2="t.p" :y2="H - padB" class="grid" />
            <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
          </g>

          <!-- points clipped to the plot box, so a zoom never pushes a mark off
               the chart where it can't be reached -->
          <g :clip-path="`url(#${clipId})`">
            <path v-for="(pt, i) in scaled" :key="i" :d="pt.d"
                  :fill="pt.color" :style="{ color: pt.color }" class="dot"
                  :class="{ selectable: pt.obs }" @mouseenter="active = pt"
                  @pointerdown="onDotDown($event, pt)" @pointerup="onDotUp($event, pt)"
                  @click="onDotClick(pt)" />

            <g v-if="todayX !== null && Number.isFinite(todayX) && inXDomain(todayX)">
              <line :x1="sx(todayX)" :y1="padT" :x2="sx(todayX)" :y2="H - padB" class="today-line" />
              <text :x="sx(todayX) + 4" :y="padT + 12" class="today-label">{{ todayLabel }}</text>
            </g>
          </g>

          <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
          <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>
        </svg>
      </div>

      <div class="zoombar">
        <button title="Zoom in" @click="zoomBy(1.6)">+</button>
        <button title="Zoom out" @click="zoomBy(1 / 1.6)">−</button>
        <button v-if="zoomed" title="Reset zoom" @click="resetZoom">⟲</button>
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
import { useId } from 'vue'
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ x, y, color?, label? }]
  data: { type: Array, required: true },
  xLabel: { type: String, default: 'x' },
  yLabel: { type: String, default: 'y' },
  xFormat: {
    type: Function,
    default: (v) => {
      if (!Number.isFinite(Number(v))) return ''
      const num = Number(v)
      if (Number.isInteger(num)) return Math.round(num).toLocaleString()
      return Math.abs(num) < 10 ? num.toFixed(2) : num.toFixed(1)
    },
  },
  yFormat: {
    type: Function,
    default: (v) => {
      if (!Number.isFinite(Number(v))) return ''
      const num = Number(v)
      if (Number.isInteger(num)) return Math.round(num).toLocaleString()
      return Math.abs(num) < 10 ? num.toFixed(2) : num.toFixed(1)
    },
  },
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
// useId (not Math.random) so the server and client agree on the clip-path id —
// a random one differs between the two renders and trips hydration.
const clipId = `plot-clip-${useId()}`

function clamp(v, min, max) { return Math.min(max, Math.max(min, v)) }
function domain(vals) {
  if (!vals.length) return [0, 1]
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.05
  return [lo - pad, hi + pad]
}

const points = computed(() => props.data.filter((d) => Number.isFinite(d.x) && Number.isFinite(d.y)))
const baseXDom = computed(() => domain(points.value.map((d) => d.x)))
const baseYDom = computed(() => domain(points.value.map((d) => d.y)))

// Zoom is expressed on the DATA domain (not a CSS transform), so points always
// render inside the axes and stay clickable. k = zoom factor, center = focal
// point in data coords.
const k = ref(1)
const center = ref(null) // { x, y } | null → base midpoint
const zoomed = computed(() => k.value > 1.001)

function viewDomain(base, c) {
  const [blo, bhi] = base
  const bw = bhi - blo
  const w = bw / k.value
  const mid = c ?? (blo + bhi) / 2
  let lo = mid - w / 2, hi = mid + w / 2
  if (lo < blo) { hi += blo - lo; lo = blo }
  if (hi > bhi) { lo -= hi - bhi; hi = bhi }
  return [Math.max(blo, lo), Math.min(bhi, hi)]
}
const xDom = computed(() => viewDomain(baseXDom.value, center.value?.x))
const yDom = computed(() => viewDomain(baseYDom.value, center.value?.y))

const sx = (v) => padL + ((v - xDom.value[0]) / (xDom.value[1] - xDom.value[0])) * (W.value - padL - padR)
const sy = (v) => (H.value - padB) - ((v - yDom.value[0]) / (yDom.value[1] - yDom.value[0])) * (H.value - padT - padB)
// Pixel → data (for cursor-anchored zoom and drag-to-pan).
const invX = (px) => xDom.value[0] + ((px - padL) / ((W.value - padL - padR) || 1)) * (xDom.value[1] - xDom.value[0])
const invY = (py) => yDom.value[0] + (((H.value - padB) - py) / ((H.value - padT - padB) || 1)) * (yDom.value[1] - yDom.value[0])
const inXDomain = (v) => v >= xDom.value[0] && v <= xDom.value[1]

function currentCenter() {
  return center.value ?? { x: (baseXDom.value[0] + baseXDom.value[1]) / 2, y: (baseYDom.value[0] + baseYDom.value[1]) / 2 }
}
function setZoom(next, focal) {
  const kv = clamp(next, 1, 60)
  if (kv <= 1.001) { k.value = 1; center.value = null; return }
  k.value = kv
  center.value = focal || currentCenter()
}
function zoomBy(factor) { setZoom(k.value * factor) }
function resetZoom() { k.value = 1; center.value = null }

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
const dragged = ref(false)
const dragStart = ref(null)
let holdTimer = null

function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}
function onWheel(e) {
  const r = e.currentTarget.getBoundingClientRect()
  const focal = { x: invX(e.clientX - r.left), y: invY(e.clientY - r.top) }
  setZoom(k.value * (e.deltaY > 0 ? 0.83 : 1.2), k.value * (e.deltaY > 0 ? 0.83 : 1.2) <= 1.001 ? null : focal)
}
function onPointerDown(e) {
  active.value = null
  dragStart.value = { x: e.clientX, y: e.clientY, center: currentCenter() }
  dragged.value = false
}
function onPointerMove(e) {
  if (!dragStart.value) return
  const dxPx = e.clientX - dragStart.value.x
  const dyPx = e.clientY - dragStart.value.y
  if (Math.abs(dxPx) + Math.abs(dyPx) > 4) { dragged.value = true; clearHold() }
  if (!zoomed.value) return // nothing to pan at full extent
  const xSpan = xDom.value[1] - xDom.value[0], ySpan = yDom.value[1] - yDom.value[0]
  const nx = dragStart.value.center.x - dxPx * (xSpan / ((W.value - padL - padR) || 1))
  const ny = dragStart.value.center.y + dyPx * (ySpan / ((H.value - padT - padB) || 1))
  center.value = { x: nx, y: ny }
}
function onPointerUp() { dragStart.value = null; clearHold() }

function clearHold() { if (holdTimer) { clearTimeout(holdTimer); holdTimer = null } }
// Press-and-hold on a point shows the hover tooltip on touch (same info a mouse
// gets on hover); a quick tap still selects.
function onDotDown(e, pt) {
  if (e.pointerType !== 'touch') return
  const r = container.value?.getBoundingClientRect()
  clearHold()
  holdTimer = setTimeout(() => {
    if (r) ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
    active.value = pt
    holdTimer = 'fired'
  }, 350)
}
function onDotUp() { if (holdTimer && holdTimer !== 'fired') clearHold() }
function onDotClick(pt) {
  // Ignore the click that ends a pan-drag or a long-press; only a genuine tap selects.
  if (holdTimer === 'fired') { holdTimer = null; return }
  if (!dragged.value && pt.obs) emit('select', pt.obs)
}

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
.today-line { stroke: var(--danger); stroke-width: 1.5; stroke-dasharray: 4 4; }
.today-label { fill: var(--danger); font-size: 10px; font-weight: 600; }

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

.zoombar { position: absolute; left: 8px; bottom: 8px; display: flex; flex-direction: column; gap: 4px; }
.zoombar button {
  width: 26px; height: 26px; border: 1px solid var(--border); background: var(--surface);
  color: var(--text); border-radius: 6px; font-size: 15px; line-height: 1; cursor: pointer;
  box-shadow: 0 1px 3px var(--shadow); display: grid; place-items: center; padding: 0;
}
.zoombar button:hover { background: var(--surface-2); }
</style>
