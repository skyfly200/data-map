<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" role="img" :aria-label="title">
          <!-- Axis -->
          <line :x1="padL" :y1="H - padB" :x2="W - padR" :y2="H - padB" class="axis" />
          <line :x1="padL" :y1="padT" :x2="padL" :y2="H - padB" class="axis" />
          <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
          <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>

          <g v-for="t in xTicks" :key="`x${t.v}`">
            <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
          </g>

          <!-- Importance Bars -->
          <g v-for="(d, i) in scaled" :key="i">
            <rect :x="padL" :y="cy(i)" :width="d.w" :height="ch - 2" rx="2"
                  :fill="d.color" class="bar" @mouseenter="active = d" />
            <text v-bind="d.valPos" class="value">{{ d.valueLabel }}</text>
            <text v-bind="d.catPos" class="cat">{{ d.label }}</text>
          </g>
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.label }}</strong>
        <span>: {{ active.valueLabel }}</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'
import { scaleContribution } from '~/composables/maxentViz'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, value }]
  data: { type: Array, required: true },
  xLabel: { type: String, default: 'Contribution' },
  yLabel: { type: String, default: 'Predictors' },
  color: { type: String, default: SERIES_1 },
})

const padL = 120
const padR = 44
const padT = 20
const padB = 34
const W = 640
const H = computed(() => Math.max(260, props.data.length * 35 + 40))
const ch = 30

const maxV = computed(() => Math.max(1, ...props.data.map(d => d.value || 0)))

const sx = (v) => scaleContribution(v, maxV.value, { W, H: H.value, padL, padR, padT, padB })
const cy = (i) => padT + i * ch

const scaled = computed(() => props.data.map((d, i) => {
  const w = sx(d.value)
  return {
    ...d,
    color: props.color,
    w,
    valueLabel: (d.value * 100).toFixed(1) + '%',
    valPos: { x: padL + w + 6, y: cy(i) + ch / 2 + 4, 'text-anchor': 'start' },
    catPos: { x: padL - 8, y: cy(i) + ch / 2 + 4, 'text-anchor': 'end' },
  }
}))

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

const viewportStyle = computed(() => ({
  width: `${W}px`,
  height: `${H.value}px`,
  transform: 'none',
  transition: 'none',
}))

const xTicks = computed(() => {
  const out = []
  for (let i = 0; i <= 4; i++) {
    const v = (i / 4) * maxV.value
    out.push({ v, p: sx(v), label: (v * 100).toFixed(0) + '%' })
  }
  return out
})
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area {
  position: relative; overflow: auto; max-width: 100%; border-radius: 8px;
  background: linear-gradient(180deg, rgba(148, 163, 184, 0.03), rgba(148, 163, 184, 0.01));
  cursor: crosshair;
}
.chart-viewport {
  position: relative; display: block; min-width: 100%; min-height: 100%;
}
svg { width: 100%; height: 100%; display: block; }
.axis { stroke: var(--border); stroke-width: 1; }
.grid { stroke: var(--border-soft); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-x { text-anchor: middle; }
.tick-y { text-anchor: end; }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.bar { transition: opacity 0.1s; filter: var(--chart-glow); }
.bar:hover { opacity: 0.82; }
.value { fill: var(--text); font-size: 11px; font-variant-numeric: tabular-nums; }
.cat { fill: var(--muted); font-size: 11px; text-anchor: end; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 4px 8px; border-radius: 6px;
  font-size: 0.78rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
</style>
