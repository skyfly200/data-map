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

          <g v-for="(v, i) in violins" :key="i" class="violinrow" @mouseenter="active = v" @click="onViolinTap($event, v)">
            <rect class="hit" x="0" :y="v.cy - rowH / 2" :width="W" :height="rowH" />
            <text :x="padL - 8" :y="v.cy + 4" class="tick tick-y" :style="{ fontSize: `${labelFontSize}px` }">{{ v.short }}</text>

            <!-- violin shape: top half path then mirrored bottom half -->
            <path :d="v.topPath" :fill="v.color" fill-opacity="0.55" stroke="none" />
            <path :d="v.bottomPath" :fill="v.color" fill-opacity="0.55" stroke="none" />

            <!-- IQR box -->
            <rect :x="v.q1x" :y="v.cy - bh / 2" :width="Math.max(1, v.q3x - v.q1x)" :height="bh" rx="2"
                  :fill="v.color" fill-opacity="0.9" />
            <!-- median line -->
            <line :x1="v.medx" :y1="v.cy - bh / 2" :x2="v.medx" :y2="v.cy + bh / 2" class="median" />
          </g>

          <text :x="(padL + W - padR) / 2" :y="H - 2" class="axis-label">{{ xLabel }}</text>
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.label }} (n={{ active.n }})</strong>
        <span>median {{ fmt(active.medVal) }}</span>
        <span>{{ fmt(active.q1Val) }} – {{ fmt(active.q3Val) }} (IQR)</span>
        <span>{{ fmt(active.minVal) }} – {{ fmt(active.maxVal) }} (range)</span>
      </div>
    </div>

    <div v-if="showKey && violins.length" class="violinkey">
      <span v-for="v in violins" :key="v.label" class="k" :class="{ on: active && active.label === v.label }"
            @click="active = v">
        <span class="sw" :style="{ background: v.color }"></span>{{ v.label }}
      </span>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useAppearance'
import { boundsFor, clampDomain } from '~/composables/useChartFields'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, values: number[], color? }]
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

const labelFontSize = computed(() => {
  const n = props.data.length || 1
  return Math.max(8, 10 - Math.max(0, n - 6) * 0.4)
})
const W = 640
const padL = computed(() => Math.max(90, 130 - Math.min(36, Math.max(0, (props.data.length || 1) - 5) * 3)))
const padR = 24
const padT = 12
const padB = 34
const bh = 10
const rowH = 40

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
function onViolinTap(e, v) {
  if (dragged.value) return
  const r = areaEl.value?.getBoundingClientRect()
  if (r) ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
  active.value = v
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

function quantile(sorted, p) {
  const idx = (sorted.length - 1) * p
  const lo = Math.floor(idx), hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

// Kernel density estimate using a Gaussian kernel
function kde(sorted, bandwidth, evalPoints) {
  const n = sorted.length
  if (!n) return evalPoints.map((x) => ({ x, y: 0 }))
  const bw = bandwidth || (1.06 * Math.sqrt(sorted.reduce((s, v) => s + (v - sorted[Math.floor(n / 2)]) ** 2, 0) / n) * n ** -0.2) || 1
  return evalPoints.map((x) => ({
    x,
    y: sorted.reduce((s, v) => s + Math.exp(-0.5 * ((x - v) / bw) ** 2), 0) / (n * bw * Math.sqrt(2 * Math.PI)),
  }))
}

const stats = computed(() => props.data
  .map((d) => ({ ...d, values: (d.values || []).filter((v) => Number.isFinite(v)).sort((a, b) => a - b) }))
  .filter((d) => d.values.length >= 3)
  .map((d) => {
    const vals = d.values
    return {
      label: d.label,
      short: compactLabel(d.label || d.short, Math.max(8, 18 - Math.max(0, props.data.length - 6))),
      color: d.color || SERIES_1,
      n: vals.length,
      minVal: vals[0],
      maxVal: vals[vals.length - 1],
      q1Val: quantile(vals, 0.25),
      medVal: quantile(vals, 0.5),
      q3Val: quantile(vals, 0.75),
      values: vals,
    }
  }))

const domain = computed(() => {
  const all = stats.value.flatMap((d) => [d.minVal, d.maxVal])
  if (!all.length) return [0, 1]
  let lo = Math.min(...all), hi = Math.max(...all)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.06
  return clampDomain([lo - pad, hi + pad], props.bounds ?? boundsFor(props.valueKey), all)
})

const sx = (v) => padL.value + ((v - domain.value[0]) / (domain.value[1] - domain.value[0] || 1)) * (W - padL.value - padR)

// Build a smooth SVG path for one half of a violin, given kde points.
// halfSign: -1 = top half, +1 = bottom half (cy-offset direction)
function violinPath(kdePoints, maxDensity, cy, halfSign, halfH) {
  if (!kdePoints.length) return ''
  const scale = maxDensity > 0 ? halfH / maxDensity : 1

  // Start at leftmost data point on the center line
  const first = kdePoints[0]
  const last = kdePoints[kdePoints.length - 1]
  let d = `M ${sx(first.x).toFixed(1)} ${cy.toFixed(1)}`

  // Draw the outline with cubic bezier smoothing
  for (let i = 0; i < kdePoints.length; i++) {
    const p = kdePoints[i]
    const offset = (p.y * scale * halfSign).toFixed(2)
    if (i === 0) {
      d += ` L ${sx(p.x).toFixed(1)} ${(cy + parseFloat(offset)).toFixed(1)}`
    } else {
      const prev = kdePoints[i - 1]
      const cpx = ((sx(prev.x) + sx(p.x)) / 2).toFixed(1)
      const prevOff = (prev.y * scale * halfSign).toFixed(2)
      d += ` C ${cpx} ${(cy + parseFloat(prevOff)).toFixed(1)} ${cpx} ${(cy + parseFloat(offset)).toFixed(1)} ${sx(p.x).toFixed(1)} ${(cy + parseFloat(offset)).toFixed(1)}`
    }
  }

  // Close back to center line
  d += ` L ${sx(last.x).toFixed(1)} ${cy.toFixed(1)} Z`
  return d
}

const violins = computed(() => {
  const [lo, hi] = domain.value
  const EVAL_N = 80
  const evalPoints = Array.from({ length: EVAL_N }, (_, i) => lo + ((hi - lo) * i) / (EVAL_N - 1))
  const halfH = (rowH / 2) - 4

  return stats.value.map((d, i) => {
    const cy = padT + i * rowH + rowH / 2
    const kdePoints = kde(d.values, null, evalPoints)
    const maxDensity = Math.max(...kdePoints.map((p) => p.y), 1e-10)

    return {
      ...d,
      cy,
      q1x: sx(d.q1Val),
      q3x: sx(d.q3Val),
      medx: sx(d.medVal),
      topPath: violinPath(kdePoints, maxDensity, cy, -1, halfH),
      bottomPath: violinPath(kdePoints, maxDensity, cy, 1, halfH),
    }
  })
})

function ticks() {
  const [lo, hi] = domain.value
  const span = hi - lo
  const out = []
  for (let i = 0; i <= 4; i++) {
    const v = lo + (span * i) / 4
    out.push({ v, p: sx(v), label: props.format(v) })
  }
  return out
}
const xTicks = computed(ticks)
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
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; fill: var(--text); }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.median { stroke: #fff; stroke-width: 2; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }

.violinrow { cursor: pointer; }
.hit { fill: transparent; pointer-events: all; }
.violinkey {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px;
  font-size: 0.75rem; color: var(--text); max-height: 92px; overflow-y: auto;
}
.violinkey .k { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; opacity: 0.9; }
.violinkey .k:hover, .violinkey .k.on { opacity: 1; font-weight: 600; }
.violinkey .sw { width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
