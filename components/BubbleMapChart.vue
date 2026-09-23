<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="areaEl" class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
          <!-- Map background grid lines (lat/lng graticule) -->
          <g class="graticule">
            <line v-for="t in latLines" :key="`lat${t.v}`"
                  :x1="padL" :y1="t.p" :x2="W - padR" :y2="t.p" class="grid" />
            <text v-for="t in latLines" :key="`latl${t.v}`"
                  :x="padL - 4" :y="t.p + 3" class="tick tick-y">{{ t.label }}</text>
            <line v-for="t in lngLines" :key="`lng${t.v}`"
                  :x1="t.p" :y1="padT" :x2="t.p" :y2="H - padB" class="grid" />
            <text v-for="t in lngLines" :key="`lngl${t.v}`"
                  :x="t.p" :y="H - padB + 13" class="tick tick-x">{{ t.label }}</text>
          </g>

          <!-- Bubbles, sorted smallest to largest so big ones don't swallow small -->
          <circle v-for="(b, i) in sorted" :key="i"
                  :cx="b.px" :cy="b.py" :r="b.r"
                  :fill="b.color" fill-opacity="0.55"
                  stroke="currentColor" stroke-opacity="0.2" stroke-width="0.5"
                  class="bubble"
                  @mouseenter="active = b" />
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.label }}</strong>
        <span v-if="active.sizeVal !== null">{{ sizeLabel }}: {{ sizeFmt(active.sizeVal) }}</span>
        <span>{{ active.lat.toFixed(3) }}°, {{ active.lng.toFixed(3) }}°</span>
        <span v-if="active.n > 1">{{ active.n }} observations</span>
      </div>
    </div>

    <div v-if="legend.length" class="bblkey">
      <span v-for="l in legend" :key="l.label" class="k">
        <span class="sw" :style="{ background: l.color }"></span>{{ l.label }}
      </span>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useAppearance'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ lat, lng, label, color?, sizeVal: number|null, n? }]
  data: { type: Array, required: true },
  sizeLabel: { type: String, default: 'Value' },
  sizeFormat: { type: Function, default: (v) => Number.isFinite(v) ? Number(v).toFixed(1) : '' },
  legend: { type: Array, default: () => [] },
})

const W = 640
const H = 380
const padL = 36
const padR = 16
const padT = 12
const padB = 24

const zoom = ref(1)
const pan = ref({ x: 0, y: 0 })
const dragStart = ref(null)
const dragged = ref(false)
const viewportStyle = computed(() => ({
  width: `${W}px`,
  height: `${H}px`,
  transform: `translate(${pan.value.x}px, ${pan.value.y}px) scale(${zoom.value})`,
  transformOrigin: '0 0',
  transition: dragStart.value ? 'none' : 'transform 0.15s ease-out',
}))

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
const areaEl = ref(null)

function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}
function onWheel(e) {
  const delta = e.deltaY > 0 ? 0.85 : 1.18
  zoom.value = Math.min(8, Math.max(0.5, zoom.value * delta))
}
function onPointerDown(e) {
  dragStart.value = { x: e.clientX, y: e.clientY, panX: pan.value.x, panY: pan.value.y }
  dragged.value = false
}
function onPointerMove(e) {
  if (!dragStart.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  if (Math.abs(dx) + Math.abs(dy) > 4) dragged.value = true
  pan.value = { x: dragStart.value.panX + dx / zoom.value, y: dragStart.value.panY + dy / zoom.value }
}
function onPointerUp() { dragStart.value = null }

const sizeFmt = (v) => props.sizeFormat(v)

// Web Mercator projection scaled to SVG viewport
const lngDomain = computed(() => {
  const lngs = props.data.map((d) => d.lng).filter((v) => Number.isFinite(v))
  if (!lngs.length) return [-180, 180]
  let lo = Math.min(...lngs), hi = Math.max(...lngs)
  if (lo === hi) { lo -= 0.5; hi += 0.5 }
  const pad = (hi - lo) * 0.1
  return [lo - pad, hi + pad]
})

const latDomain = computed(() => {
  const lats = props.data.map((d) => d.lat).filter((v) => Number.isFinite(v))
  if (!lats.length) return [-60, 60]
  let lo = Math.min(...lats), hi = Math.max(...lats)
  if (lo === hi) { lo -= 0.5; hi += 0.5 }
  const pad = (hi - lo) * 0.12
  return [Math.max(-85, lo - pad), Math.min(85, hi + pad)]
})

function mercY(lat) {
  const rad = (lat * Math.PI) / 180
  return Math.log(Math.tan(Math.PI / 4 + rad / 2))
}

const mercYDomain = computed(() => latDomain.value.map(mercY))

const sx = (lng) => padL + ((lng - lngDomain.value[0]) / (lngDomain.value[1] - lngDomain.value[0] || 1)) * (W - padL - padR)
const sy = (lat) => {
  const my = mercY(lat)
  const [mlo, mhi] = mercYDomain.value
  return H - padB - ((my - mlo) / ((mhi - mlo) || 1)) * (H - padT - padB)
}

const sizeRange = computed(() => {
  const vals = props.data.map((d) => d.sizeVal).filter((v) => v !== null && Number.isFinite(v))
  if (!vals.length) return [0, 1]
  return [Math.min(...vals), Math.max(...vals)]
})

const minR = 3, maxR = 18

function radius(sizeVal) {
  if (sizeVal === null || !Number.isFinite(sizeVal)) return (minR + maxR) / 2
  const [lo, hi] = sizeRange.value
  if (lo === hi) return (minR + maxR) / 2
  return minR + (maxR - minR) * ((sizeVal - lo) / (hi - lo))
}

const bubbles = computed(() => props.data
  .filter((d) => Number.isFinite(d.lat) && Number.isFinite(d.lng))
  .map((d) => ({
    ...d,
    px: sx(d.lng),
    py: sy(d.lat),
    r: radius(d.sizeVal ?? null),
    color: d.color || SERIES_1,
  })))

const sorted = computed(() => [...bubbles.value].sort((a, b) => b.r - a.r))

function gridLines(lo, hi, count) {
  const step = (hi - lo) / count
  const out = []
  for (let i = 0; i <= count; i++) out.push(lo + i * step)
  return out
}

const latLines = computed(() => {
  const [lo, hi] = latDomain.value
  return gridLines(lo, hi, 4).map((v) => ({
    v, p: sy(v), label: `${v.toFixed(1)}°`,
  }))
})

const lngLines = computed(() => {
  const [lo, hi] = lngDomain.value
  return gridLines(lo, hi, 4).map((v) => ({
    v, p: sx(v), label: `${v.toFixed(1)}°`,
  }))
})
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area {
  position: relative; overflow: auto; max-width: 100%; border-radius: 8px;
  background: var(--surface-2, #f1f5f9);
  cursor: grab; user-select: none; touch-action: none;
}
.chart-area:active { cursor: grabbing; }
.chart-viewport { position: relative; display: block; }
svg { width: 100%; height: 100%; display: block; }
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 9px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; }
.bubble { cursor: default; }
.bubble:hover { fill-opacity: 0.8; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
.bblkey {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px;
  font-size: 0.75rem; color: var(--text); max-height: 72px; overflow-y: auto;
}
.bblkey .k { display: inline-flex; align-items: center; gap: 5px; }
.bblkey .sw { width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
