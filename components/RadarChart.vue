<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="container" class="chart-area">
      <svg :viewBox="`0 0 ${size} ${size}`" role="img" :aria-label="title" preserveAspectRatio="xMidYMid meet">
        <!-- Rings, with the outermost labelled so the shape has a scale. A radar
             without one is a decoration. -->
        <g class="rings">
          <circle v-for="r in rings" :key="r.t" :cx="cx" :cy="cy" :r="r.r" class="ring" />
          <text :x="cx + 3" :y="cy - radius + 10" class="ring-label">{{ format(max) }}</text>
          <text :x="cx + 3" :y="cy - radius / 2 + 10" class="ring-label">{{ format(max / 2) }}</text>
        </g>

        <!-- One spoke per category, labelled around the outside. -->
        <g class="spokes">
          <line v-for="p in points" :key="`s-${p.label}`"
                :x1="cx" :y1="cy" :x2="p.ax" :y2="p.ay" class="spoke" />
          <text v-for="p in points" :key="`t-${p.label}`"
                :x="p.lx" :y="p.ly" class="spoke-label"
                :text-anchor="p.anchor" dominant-baseline="middle">{{ p.short }}</text>
        </g>

        <polygon :points="polygon" class="area" />
        <g>
          <circle v-for="p in points" :key="`v-${p.label}`" :cx="p.x" :cy="p.y" r="3" class="vertex" />
        </g>

        <!-- Values sit at the vertices: the enclosed area is not the quantity,
             and reading the shape instead of the points is how radar misleads. -->
        <g v-if="points.length <= 12">
          <text v-for="p in points" :key="`n-${p.label}`" :x="p.x" :y="p.y - 7"
                class="value" text-anchor="middle">{{ format(p.value) }}</text>
        </g>
      </svg>
    </div>
  </figure>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, value, color? }]
  data: { type: Array, default: () => [] },
  format: { type: Function, default: (v) => String(Math.round(v)) },
})

// A fixed square canvas the SVG scales into, so the shape stays circular.
const size = 320
const cx = size / 2
const cy = size / 2
const radius = 108

// Beyond a dozen spokes the labels collide and the shape becomes a blob.
const MAX_SPOKES = 12
const items = computed(() => props.data.filter((d) => Number.isFinite(Number(d.value))).slice(0, MAX_SPOKES))
const max = computed(() => Math.max(1, ...items.value.map((d) => Number(d.value))))

const rings = computed(() => [0.25, 0.5, 0.75, 1].map((t) => ({ t, r: radius * t })))

const points = computed(() => {
  const n = items.value.length
  return items.value.map((d, i) => {
    // Start at twelve o'clock and go clockwise, which is how these are read.
    const a = (i / n) * Math.PI * 2 - Math.PI / 2
    const t = Number(d.value) / max.value
    const label = String(d.label)
    const cos = Math.cos(a)
    return {
      label,
      short: label.length > 14 ? `${label.slice(0, 13)}…` : label,
      value: Number(d.value),
      x: cx + Math.cos(a) * radius * t,
      y: cy + Math.sin(a) * radius * t,
      ax: cx + Math.cos(a) * radius,
      ay: cy + Math.sin(a) * radius,
      lx: cx + Math.cos(a) * (radius + 14),
      ly: cy + Math.sin(a) * (radius + 14),
      // Labels on the left of the circle read outward, so they hang left.
      anchor: Math.abs(cos) < 0.15 ? 'middle' : (cos > 0 ? 'start' : 'end'),
    }
  })
})

const polygon = computed(() => points.value.map((p) => `${p.x},${p.y}`).join(' '))
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 10px; }
.chart-area { position: relative; height: 100%; min-height: 260px; overflow: hidden; }
svg { width: 100%; height: 100%; display: block; }

.ring { fill: none; stroke: var(--border); stroke-width: 1; }
.ring-label { fill: var(--muted); font-size: 9px; }
.spoke { stroke: var(--border); stroke-width: 1; }
.spoke-label { fill: var(--muted); font-size: 9px; }
.area { fill: var(--accent, #2a78d6); fill-opacity: 0.25; stroke: var(--accent, #2a78d6); stroke-width: 2; }
.vertex { fill: var(--accent, #2a78d6); }
.value { fill: var(--text); font-size: 9px; font-weight: 600; }
</style>
