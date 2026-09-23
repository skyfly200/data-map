<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="container" class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
        <defs>
          <clipPath :id="clipId">
            <rect :x="padL" :y="padT" :width="W - padL - padR" :height="H - padT - padB" />
          </clipPath>
        </defs>

        <g v-for="t in yTicks" :key="`y${t.v}`">
          <line :x1="padL" :y1="t.p" :x2="W - padR" :y2="t.p" class="grid" />
          <text :x="padL - 6" :y="t.p + 3" class="tick tick-y">{{ t.label }}</text>
        </g>
        <g v-for="t in xTicks" :key="`x${t.v}`">
          <line :x1="t.p" :y1="padT" :x2="t.p" :y2="H - padB" class="grid" />
          <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
        </g>

        <g :clip-path="`url(#${clipId})`">
          <polygon v-for="(h, i) in hexes" :key="i"
                   :points="h.points"
                   :fill="h.fill"
                   class="hex"
                   @mouseenter="active = h" />
        </g>

        <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
        <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>
      </svg>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.count }} observations</strong>
        <span>{{ xLabel }}: {{ xFmt(active.cx) }}</span>
        <span>{{ yLabel }}: {{ yFmt(active.cy) }}</span>
      </div>

      <div class="colorbar">
        <span class="cb-lo">1</span>
        <svg width="80" height="10" class="cb-strip">
          <defs>
            <linearGradient id="hx-grad" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" :stop-color="colorLow" />
              <stop offset="100%" :stop-color="colorHigh" />
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="80" height="10" fill="url(#hx-grad)" rx="2" />
        </svg>
        <span class="cb-hi">{{ maxCount }}</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { boundsFor, clampDomain } from '~/composables/useChartFields'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ x, y }]
  data: { type: Array, required: true },
  xLabel: { type: String, default: '' },
  yLabel: { type: String, default: '' },
  xKey: { type: String, default: '' },
  yKey: { type: String, default: '' },
  xBounds: { type: Array, default: null },
  yBounds: { type: Array, default: null },
  xFormat: { type: Function, default: (v) => Number.isFinite(Number(v)) ? Number(v).toFixed(1) : '' },
  yFormat: { type: Function, default: (v) => Number.isFinite(Number(v)) ? Number(v).toFixed(1) : '' },
  bins: { type: Number, default: 20 },
})

const W = 640
const H = 400
const padL = 52
const padR = 24
const padT = 12
const padB = 36

const colorLow = 'rgba(99,179,237,0.3)'
const colorHigh = '#2563eb'

const clipId = `hx-clip-${Math.random().toString(36).slice(2)}`

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
const container = ref(null)
const dragStart = ref(null)
const dragged = ref(false)

function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}
function clampV(v, min, max) { return Math.min(max, Math.max(min, v)) }
function onWheel(e) { e.preventDefault() }
function onPointerDown(e) {
  dragStart.value = { x: e.clientX, y: e.clientY }
  dragged.value = false
}
function onPointerMove(e) {
  if (!dragStart.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  if (Math.abs(dx) + Math.abs(dy) > 4) dragged.value = true
}
function onPointerUp() { dragStart.value = null }

const xFmt = (v) => props.xFormat(v)
const yFmt = (v) => props.yFormat(v)

const xDomain = computed(() => {
  const xs = props.data.map((d) => d.x).filter((v) => Number.isFinite(v))
  if (!xs.length) return [0, 1]
  let lo = Math.min(...xs), hi = Math.max(...xs)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.04
  return clampDomain([lo - pad, hi + pad], props.xBounds ?? boundsFor(props.xKey), xs)
})

const yDomain = computed(() => {
  const ys = props.data.map((d) => d.y).filter((v) => Number.isFinite(v))
  if (!ys.length) return [0, 1]
  let lo = Math.min(...ys), hi = Math.max(...ys)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.04
  return clampDomain([lo - pad, hi + pad], props.yBounds ?? boundsFor(props.yKey), ys)
})

const sx = (v) => padL + ((v - xDomain.value[0]) / (xDomain.value[1] - xDomain.value[0] || 1)) * (W - padL - padR)
const sy = (v) => H - padB - ((v - yDomain.value[0]) / (yDomain.value[1] - yDomain.value[0] || 1)) * (H - padT - padB)

// Flat-top hex grid. radius = hex circumradius in SVG units.
const hexes = computed(() => {
  const n = Math.max(4, Math.min(40, props.bins))
  const plotW = W - padL - padR
  const plotH = H - padT - padB
  const r = Math.max(4, Math.min(plotW, plotH) / (n * 1.5))
  const w = r * 2
  const h = Math.sqrt(3) * r

  // Build hex grid in pixel space (flat-top orientation)
  const cols = Math.ceil(plotW / (w * 0.75)) + 1
  const rows = Math.ceil(plotH / h) + 1
  const bins = new Map()
  for (let col = 0; col <= cols; col++) {
    for (let row = 0; row <= rows; row++) {
      const cx = padL + col * w * 0.75
      const cy = padT + row * h + (col % 2 === 0 ? 0 : h / 2)
      bins.set(`${col},${row}`, { cx, cy, count: 0, col, row })
    }
  }

  // Assign each point to nearest hex center
  for (const { x, y } of props.data) {
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue
    const px = sx(x), py = sy(y)
    let best = null, bestD = Infinity
    for (const [, b] of bins) {
      const d = (px - b.cx) ** 2 + (py - b.cy) ** 2
      if (d < bestD) { bestD = d; best = b }
    }
    if (best) best.count++
  }

  const maxCount = Math.max(...[...bins.values()].map((b) => b.count), 1)

  // Build hex polygon points (flat-top)
  function hexPoints(cx, cy) {
    return Array.from({ length: 6 }, (_, i) => {
      const angle = (Math.PI / 180) * (60 * i)
      return `${(cx + r * Math.cos(angle)).toFixed(1)},${(cy + r * Math.sin(angle)).toFixed(1)}`
    }).join(' ')
  }

  // Interpolate color low→high
  function hexColor(count) {
    if (!count) return 'transparent'
    const t = count / maxCount
    const r1 = 99, g1 = 179, b1 = 237
    const r2 = 37, g2 = 99, b2 = 235
    const rr = Math.round(r1 + (r2 - r1) * t)
    const rg = Math.round(g1 + (g2 - g1) * t)
    const rb = Math.round(b1 + (b2 - b1) * t)
    return `rgba(${rr},${rg},${rb},${0.3 + 0.7 * t})`
  }

  return [...bins.values()]
    .filter((b) => b.count > 0)
    .map((b) => ({ ...b, points: hexPoints(b.cx, b.cy), fill: hexColor(b.count), maxCount }))
})

const maxCount = computed(() => Math.max(...hexes.value.map((h) => h.count), 1))

const xTicks = computed(() => {
  const [lo, hi] = xDomain.value
  return Array.from({ length: 5 }, (_, i) => {
    const v = lo + ((hi - lo) * i) / 4
    return { v, p: sx(v), label: props.xFormat(v) }
  })
})
const yTicks = computed(() => {
  const [lo, hi] = yDomain.value
  return Array.from({ length: 5 }, (_, i) => {
    const v = lo + ((hi - lo) * i) / 4
    return { v, p: sy(v), label: props.yFormat(v) }
  })
})
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area {
  position: relative; overflow: auto; max-width: 100%; border-radius: 8px;
  background: linear-gradient(180deg, rgba(148, 163, 184, 0.03), rgba(148, 163, 184, 0.01));
  user-select: none;
}
svg { width: 100%; height: auto; display: block; }
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.hex { cursor: default; transition: opacity 0.1s; }
.hex:hover { opacity: 0.8; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
.colorbar {
  display: flex; align-items: center; gap: 6px; margin-top: 6px; font-size: 0.72rem; color: var(--muted);
}
.cb-strip { border-radius: 2px; }
</style>
