<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area" @mousemove="onMove" @mouseleave="active = null" @wheel.prevent="onWheel" @pointerdown="onPointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointerleave="onPointerUp">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" role="img" :aria-label="title">
          <!-- baseline / axis (recessive) -->
          <line v-if="horizontal" :x1="padL" :y1="padT" :x2="padL" :y2="H - padB" class="axis" />
          <line v-else :x1="padL" :y1="H - padB" :x2="W - padR" :y2="H - padB" class="axis" />

          <template v-for="(d, i) in scaled" :key="i">
            <path :d="d.path" :fill="d.color" :style="{ color: d.color }" class="bar" @mouseenter="active = d" />
            <!-- direct value label at the data end -->
            <text v-bind="d.valuePos" class="value">{{ d.valueLabel }}</text>
            <!-- category label -->
            <text v-bind="d.catPos" class="cat">{{ d.short }}</text>
          </template>
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.label }}</strong><span>{{ active.valueLabel }}</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, value, color?, short? }]
  data: { type: Array, required: true },
  horizontal: { type: Boolean, default: false },
  format: { type: Function, default: (v) => String(v) },
})

const labelFontSize = computed(() => {
  const n = props.data.length || 1
  return Math.max(8, 11 - Math.max(0, n - 8) * 0.35)
})
const W = computed(() => props.horizontal ? Math.max(640, (props.data.length || 1) * 120 + 180) : 640)
const H = computed(() => props.horizontal ? Math.max(120, props.data.length * 30 + 24) : 260)
const padL = computed(() => props.horizontal ? Math.max(84, 128 - Math.min(36, Math.max(0, (props.data.length || 1) - 6) * 4)) : 34)
const padR = 44
const padT = 20  // headroom so the tallest bar's value label isn't clipped
const padB = computed(() => props.horizontal ? 8 : 40)

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
  if (!value) return ''
  const safeMax = Math.max(5, maxLen)
  if (value.length <= safeMax) return value
  return `${value.slice(0, Math.max(0, safeMax - 1)).trimEnd()}…`
}
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

function clamp(v, min, max) { return Math.min(max, Math.max(min, v)) }

function onWheel(e) {
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  const next = clamp(zoom.value * delta, 0.7, 2.5)
  zoom.value = next
}

function onPointerDown(e) {
  dragStart.value = { x: e.clientX, y: e.clientY, panX: pan.value.x, panY: pan.value.y }
  e.currentTarget.setPointerCapture?.(e.pointerId)
}

function onPointerMove(e) {
  if (!dragStart.value) return
  const dx = e.clientX - dragStart.value.x
  const dy = e.clientY - dragStart.value.y
  pan.value = {
    x: dragStart.value.panX + dx / zoom.value,
    y: dragStart.value.panY + dy / zoom.value,
  }
}

function onPointerUp() {
  dragStart.value = null
}

function roundedRectPath(x, y, w, h, r, corner) {
  // corner: 'top' rounds y-min edge (vertical bars), 'right' rounds x-max edge.
  r = Math.max(0, Math.min(r, w / 2, h / 2))
  if (corner === 'top') {
    return `M${x},${y + h} L${x},${y + r} Q${x},${y} ${x + r},${y} `
      + `L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} L${x + w},${y + h} Z`
  }
  // 'right'
  return `M${x},${y} L${x + w - r},${y} Q${x + w},${y} ${x + w},${y + r} `
    + `L${x + w},${y + h - r} Q${x + w},${y + h} ${x + w - r},${y + h} L${x},${y + h} Z`
}

const scaled = computed(() => {
  const maxV = Math.max(1, ...props.data.map((d) => d.value || 0))
  const n = props.data.length || 1
  const hh = H.value

  if (props.horizontal) {
    const band = (hh - padT - padB.value) / n
    const barH = Math.min(18, band - 8)
    const trackW = W.value - padL.value - padR
    return props.data.map((d, i) => {
      const y = padT + i * band + (band - barH) / 2
      const w = (d.value / maxV) * trackW
      const labelLimit = Math.max(8, 18 - Math.max(0, props.data.length - 8))
      return {
        ...d,
        color: d.color || SERIES_1,
        path: roundedRectPath(padL.value, y, Math.max(0.5, w), barH, 4, 'right'),
        valueLabel: props.format(d.value),
        short: compactLabel(d.short || d.label, labelLimit),
        valuePos: { x: padL.value + w + 6, y: y + barH / 2 + 4, 'text-anchor': 'start' },
        catPos: { x: padL.value - 8, y: y + barH / 2 + 4, 'text-anchor': 'end' },
      }
    })
  }

  const band = (W.value - padL.value - padR) / n
  const barW = Math.min(46, band - 8)
  const trackH = hh - padT - padB.value
  return props.data.map((d, i) => {
    const x = padL.value + i * band + (band - barW) / 2
    const h = (d.value / maxV) * trackH
    const y = hh - padB.value - h
    return {
      ...d,
      color: d.color || SERIES_1,
      path: roundedRectPath(x, y, barW, Math.max(0.5, h), 4, 'top'),
      valueLabel: props.format(d.value),
      short: d.short || d.label,
      valuePos: { x: x + barW / 2, y: y - 5, 'text-anchor': 'middle' },
      catPos: { x: x + barW / 2, y: hh - padB.value + 16, 'text-anchor': 'middle' },
    }
  })
})
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

.axis { stroke: var(--border); stroke-width: 1; }
.bar { transition: opacity 0.1s; filter: var(--chart-glow); }
.bar:hover { opacity: 0.82; }
.value { fill: var(--text); font-size: 11px; font-variant-numeric: tabular-nums; }
.cat { fill: var(--muted); font-size: v-bind('`${labelFontSize}px`'); }

.tooltip {
  position: absolute; pointer-events: none; z-index: 10;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 4px 8px; border-radius: 6px;
  font-size: 0.78rem; white-space: nowrap; display: flex; gap: 8px; align-items: baseline;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip span { font-variant-numeric: tabular-nums; opacity: 0.85; }
</style>
