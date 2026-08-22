<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area" @mousemove="onMove" @mouseleave="active = null">
      <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="none" role="img" :aria-label="title">
        <!-- baseline / axis (recessive) -->
        <line v-if="horizontal" :x1="padL" :y1="padT" :x2="padL" :y2="H - padB" class="axis" />
        <line v-else :x1="padL" :y1="H - padB" :x2="W - padR" :y2="H - padB" class="axis" />

        <template v-for="(d, i) in scaled" :key="i">
          <path :d="d.path" :fill="d.color" class="bar" @mouseenter="active = d" />
          <!-- direct value label at the data end -->
          <text v-bind="d.valuePos" class="value">{{ d.valueLabel }}</text>
          <!-- category label -->
          <text v-bind="d.catPos" class="cat">{{ d.short }}</text>
        </template>
      </svg>

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

const W = 640
const H = computed(() => props.horizontal ? Math.max(120, props.data.length * 30 + 24) : 260)
const padL = computed(() => props.horizontal ? 128 : 34)
const padR = 44
const padT = 20  // headroom so the tallest bar's value label isn't clipped
const padB = computed(() => props.horizontal ? 8 : 40)

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
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
    const trackW = W - padL.value - padR
    return props.data.map((d, i) => {
      const y = padT + i * band + (band - barH) / 2
      const w = (d.value / maxV) * trackW
      return {
        ...d,
        color: d.color || SERIES_1,
        path: roundedRectPath(padL.value, y, Math.max(0.5, w), barH, 4, 'right'),
        valueLabel: props.format(d.value),
        short: d.short || d.label,
        valuePos: { x: padL.value + w + 6, y: y + barH / 2 + 4, 'text-anchor': 'start' },
        catPos: { x: padL.value - 8, y: y + barH / 2 + 4, 'text-anchor': 'end' },
      }
    })
  }

  const band = (W - padL.value - padR) / n
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
.chart-title { font-size: 0.95rem; font-weight: 600; color: #1f2933; margin-bottom: 6px; }
.chart-area { position: relative; }
svg { width: 100%; height: auto; display: block; }

.axis { stroke: #d1d5db; stroke-width: 1; }
.bar { transition: opacity 0.1s; }
.bar:hover { opacity: 0.82; }
.value { fill: #4b5563; font-size: 11px; font-variant-numeric: tabular-nums; }
.cat { fill: #6b7280; font-size: 11px; }

.tooltip {
  position: absolute; pointer-events: none; z-index: 10;
  background: #1f2933; color: #fff; padding: 4px 8px; border-radius: 6px;
  font-size: 0.78rem; white-space: nowrap; display: flex; gap: 8px; align-items: baseline;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip span { font-variant-numeric: tabular-nums; opacity: 0.85; }
</style>
