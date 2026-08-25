<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div v-if="!values.length" class="empty">
      Needs slope aspect — run the DEM/terrain step to populate this.
    </div>
    <div v-else class="chart-area" @mousemove="onMove" @mouseleave="active = null">
      <svg :viewBox="`0 0 ${W} ${W}`" role="img" :aria-label="title">
        <circle v-for="r in rings" :key="r" :cx="c" :cy="c" :r="r" class="ring" />
        <text v-for="(lbl, k) in ['N', 'E', 'S', 'W']" :key="lbl"
              :x="c + Math.sin(k * Math.PI / 2) * (R + 12)"
              :y="c - Math.cos(k * Math.PI / 2) * (R + 12) + 4" class="compass">{{ lbl }}</text>
        <path v-for="(s, i) in sectors" :key="i" :d="s.path" class="sector"
              @mouseenter="active = s" />
      </svg>
      <div v-if="active" class="tooltip" :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ active.dir }}</strong><span>{{ active.count }} obs</span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { SERIES_1 } from '~/composables/useObservations'

const props = defineProps({
  title: { type: String, default: '' },
  values: { type: Array, default: () => [] }, // aspect degrees 0–360
  bins: { type: Number, default: 8 },
})

const W = 320
const c = W / 2
const R = W / 2 - 26
const rings = [R / 3, (2 * R) / 3, R]

const DIRS8 = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

const counts = computed(() => {
  const n = props.bins
  const arr = new Array(n).fill(0)
  const step = 360 / n
  for (const v of props.values) {
    if (!Number.isFinite(v)) continue
    const idx = (Math.round(((v % 360) + 360) % 360 / step)) % n
    arr[idx] += 1
  }
  return arr
})
const maxCount = computed(() => Math.max(1, ...counts.value))

const active = ref(null)
const ptr = ref({ x: 0, y: 0 })
function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

const sectors = computed(() => {
  const n = props.bins
  const step = (2 * Math.PI) / n
  return counts.value.map((count, i) => {
    const r = (count / maxCount.value) * R
    const a0 = i * step - step / 2 - Math.PI / 2
    const a1 = i * step + step / 2 - Math.PI / 2
    const p0 = [c + Math.cos(a0) * r, c + Math.sin(a0) * r]
    const p1 = [c + Math.cos(a1) * r, c + Math.sin(a1) * r]
    const path = `M${c},${c} L${p0[0]},${p0[1]} A${r},${r} 0 0 1 ${p1[0]},${p1[1]} Z`
    return { path, count, dir: DIRS8[Math.round((i * 360 / n) / 45) % 8], fill: SERIES_1 }
  })
})
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: #1f2933; margin-bottom: 6px; }
.chart-area { position: relative; }
svg { width: 100%; max-width: 320px; height: auto; display: block; margin: 0 auto; }
.ring { fill: none; stroke: #eef0f2; stroke-width: 1; }
.compass { fill: #9aa0a6; font-size: 11px; text-anchor: middle; font-weight: 600; }
.sector { fill: #2a78d6; fill-opacity: 0.8; stroke: #fff; stroke-width: 1; }
.sector:hover { fill-opacity: 1; }
.empty { color: #9aa0a6; font-size: 0.85rem; padding: 24px 8px; text-align: center; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: #1f2933; color: #fff; padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
</style>
