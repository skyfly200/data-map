<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="chart-area">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" role="img" :aria-label="title">
          <line :x1="padL" :y1="padT" :x2="padL" :y2="H - padB" class="axis" />
          <line :x1="padL" :y1="H - padB" :x2="W - padR" :y2="H - padB" class="axis" />
          <text :x="(padL + W - padR) / 2" :y="H - 3" class="axis-label">{{ xLabel }}</text>
          <text :x="-(padT + H - padB) / 2" :y="12" transform="rotate(-90)" class="axis-label">{{ yLabel }}</text>

          <g v-for="t in xTicks" :key="`x${t.v}`">
            <text :x="t.p" :y="H - padB + 14" class="tick tick-x">{{ t.label }}</text>
          </g>

          <polyline :points="polyline" class="line" :style="{ stroke: color, color: color }" />
          <circle v-for="pt in points" :key="pt.x" :cx="pt.cx" :cy="pt.cy" r="3" class="dot" :style="{ fill: color, color: color }" />
        </svg>
      </div>

      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <span v-if="active.label">{{ active.label }}: </span>
        <strong>{{ active.valueLabel }}</strong>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useAppearance'
import { scalePoint, generateTicks, DEFAULT_DIMS } from '~/composables/maxentViz'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ x, y }] ordered by x
  data: { type: Array, required: true },
  xLabel: { type: String, default: 'True Presence' },
  yLabel: { type: String, default: 'Predicted Probability' },
  color: { type: String, default: SERIES_1 },
})

const { padL, padR, padT, padB, W, H } = DEFAULT_DIMS

const xDom = computed(() => {
  const vals = props.data.map(d => d.x)
  return [Math.min(...vals), Math.max(...vals)]
})
const yDom = [0, 1]

const points = computed(() => props.data.map(d => scalePoint({ x: d.x, y: d.y }, xDom.value, yDom)))
const polyline = computed(() => points.value.map(p => `${p.cx},${p.cy}`).join(' '))

const xTicks = computed(() => generateTicks(xDom.value, 4, true))
const yTicks = computed(() => generateTicks(yDom, 4, false))

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

const viewportStyle = computed(() => ({
  width: `${W}px`,
  height: `${H}px`,
  transform: 'none',
  transition: 'none',
}))
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
.line { fill: none; stroke-width: 2; filter: var(--chart-glow); }
.dot { stroke: var(--surface); stroke-width: 1; filter: var(--chart-glow); }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 4px 8px; border-radius: 6px;
  font-size: 0.78rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
</style>
