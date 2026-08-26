<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div class="pie-wrap">
      <svg :viewBox="`0 0 ${size} ${size}`" role="img" :aria-label="title">
        <g :transform="`translate(${size / 2}, ${size / 2})`">
          <path v-for="(a, i) in arcs" :key="i" :d="a.d" :fill="a.color">
            <title>{{ a.label }}: {{ format(a.value) }} ({{ a.pct }}%)</title>
          </path>
          <circle :r="inner" fill="#fff" />
          <text class="center-num" text-anchor="middle" dy="-2">{{ format(total) }}</text>
          <text class="center-lbl" text-anchor="middle" dy="14">total</text>
        </g>
      </svg>
      <div class="legend">
        <span v-for="(a, i) in arcs" :key="i" class="lg">
          <span class="sw" :style="{ background: a.color }"></span>{{ a.label }}
          <span class="lg-val">{{ a.pct }}%</span>
        </span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { PALETTE, UNCLUSTERED } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  data: { type: Array, required: true }, // [{ label, value, color? }]
  format: { type: Function, default: (v) => String(Math.round(v)) },
})

const size = 220
const radius = 100
const inner = 56

const total = computed(() => props.data.reduce((s, d) => s + (d.value || 0), 0))

const arcs = computed(() => {
  const t = total.value || 1
  let angle = -Math.PI / 2 // start at top
  return props.data.map((d, i) => {
    const frac = (d.value || 0) / t
    const start = angle
    const end = angle + frac * Math.PI * 2
    angle = end
    const large = end - start > Math.PI ? 1 : 0
    const x0 = Math.cos(start) * radius, y0 = Math.sin(start) * radius
    const x1 = Math.cos(end) * radius, y1 = Math.sin(end) * radius
    const d3 = `M 0 0 L ${x0.toFixed(2)} ${y0.toFixed(2)} A ${radius} ${radius} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)} Z`
    return {
      d: d3,
      color: d.color || PALETTE[i % PALETTE.length] || UNCLUSTERED,
      label: d.label,
      value: d.value,
      pct: Math.round(frac * 100),
    }
  })
})
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: #1f2933; margin-bottom: 10px; }
.pie-wrap { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; justify-content: center; }
svg { width: 260px; max-width: 100%; height: auto; }
.center-num { font-size: 22px; font-weight: 700; fill: #1f2933; }
.center-lbl { font-size: 11px; fill: #9aa0a6; }
.legend { display: flex; flex-direction: column; gap: 5px; font-size: 0.8rem; color: #4b5563; }
.legend .lg { display: inline-flex; align-items: center; gap: 6px; }
.legend .sw { width: 11px; height: 11px; border-radius: 3px; flex: 0 0 auto; }
.lg-val { color: #9aa0a6; font-variant-numeric: tabular-nums; }
</style>
