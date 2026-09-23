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

          <g v-for="(row, i) in rows" :key="i" class="striprow">
            <rect class="hit" x="0" :y="row.cy - rowH / 2" :width="W" :height="rowH" />
            <text :x="padL - 8" :y="row.cy + 4" class="tick tick-y" :style="{ fontSize: `${labelFontSize}px` }">{{ row.short }}</text>
            <circle v-for="(pt, j) in row.points" :key="j"
                    :cx="pt.cx" :cy="row.cy + pt.jitter"
                    :r="dotR" :fill="row.color" fill-opacity="0.65"
                    class="dot" @mouseenter="active = { row, pt }" />
          </g>

          <text :x="(padL + W - padR) / 2" :y="H - 2" class="axis-label">{{ xLabel }}</text>
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.row.label }}</strong>
        <span>{{ fmt(active.pt.val) }}</span>
      </div>
    </div>

    <div v-if="showKey && rows.length" class="stripkey">
      <span v-for="r in rows" :key="r.label" class="k">
        <span class="sw" :style="{ background: r.color }"></span>{{ r.label }}
      </span>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useAppearance'
import { boundsFor, clampDomain } from '~/composables/useChartFields'

const props = defineProps({
  title: { type: String, default: '' },
  data: { type: Array, required: true },
  xLabel: { type: String, default: '' },
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

const W = 640
const padL = computed(() => Math.max(90, 130 - Math.min(36, Math.max(0, (props.data.length || 1) - 5) * 3)))
const padR = 24
const padT = 12
const padB = 34
const rowH = 36
const dotR = 3

const labelFontSize = computed(() => {
  const n = props.data.length || 1
  return Math.max(8, 10 - Math.max(0, n - 6) * 0.4)
})
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
function clamp(v, min, max) { return Math.min(max, Math.max(min, v)) }
function onWheel(e) {
  const delta = e.deltaY > 0 ? 0.9 : 1.1
  zoom.value = clamp(zoom.value * delta, 0.7, 2.5)
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
const fmt = (v) => props.format(v)

// Deterministic jitter using a simple LCG seeded per-row so the chart is
// stable across re-renders (no random() calls in computed).
function jitter(seed, halfH) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return ((s / 0x100000000) - 0.5) * halfH * 1.6
  }
}

const domain = computed(() => {
  const all = props.data.flatMap((d) => (d.values || []).filter((v) => Number.isFinite(v)))
  if (!all.length) return [0, 1]
  let lo = Math.min(...all), hi = Math.max(...all)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.04
  return clampDomain([lo - pad, hi + pad], props.bounds ?? boundsFor(props.valueKey), all)
})

const sx = (v) => padL.value + ((v - domain.value[0]) / (domain.value[1] - domain.value[0] || 1)) * (W - padL.value - padR)

const MAX_PER_ROW = 300
const rows = computed(() => {
  const halfH = rowH / 2 - dotR - 2
  return props.data.map((d, i) => {
    const vals = (d.values || []).filter((v) => Number.isFinite(v))
    const sample = vals.length > MAX_PER_ROW
      ? vals.filter((_, idx) => idx % Math.ceil(vals.length / MAX_PER_ROW) === 0)
      : vals
    const next = jitter(i * 31337, halfH)
    return {
      label: d.label,
      short: compactLabel(d.label || d.short, Math.max(8, 18 - Math.max(0, props.data.length - 6))),
      color: d.color || SERIES_1,
      cy: padT + i * rowH + rowH / 2,
      points: sample.map((val) => ({ val, cx: sx(val), jitter: next() })),
    }
  })
})

const xTicks = computed(() => {
  const [lo, hi] = domain.value
  return Array.from({ length: 5 }, (_, i) => {
    const v = lo + ((hi - lo) * i) / 4
    return { v, p: sx(v), label: props.format(v) }
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
.chart-viewport { position: relative; display: block; min-width: 100%; min-height: 100%; }
svg { width: 100%; height: 100%; display: block; }
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; fill: var(--text); }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.dot { cursor: default; }
.hit { fill: transparent; pointer-events: all; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
.stripkey {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px;
  font-size: 0.75rem; color: var(--text); max-height: 92px; overflow-y: auto;
}
.stripkey .k { display: inline-flex; align-items: center; gap: 5px; }
.stripkey .sw { width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
