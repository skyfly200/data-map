<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="container" class="chart-area">
      <div class="chart-viewport" :style="viewportStyle">
        <svg :viewBox="`0 0 ${W} ${H}`" preserveAspectRatio="xMidYMid meet" role="img" :aria-label="title">
          <g v-for="(g, gi) in scaled" :key="g.label">
            <text v-if="horizontal" :x="padL - 6" :y="g.centre + 3" class="tick label" text-anchor="end">{{ g.short }}</text>
            <text v-else :x="g.centre" :y="H - padB + 14" class="tick label" text-anchor="middle">{{ g.short }}</text>
            <rect v-for="seg in g.segments" :key="seg.key"
                  :x="seg.x" :y="seg.y" :width="seg.w" :height="seg.h"
                  :fill="seg.color" class="seg">
              <title>{{ g.label }} · {{ seg.key }}: {{ format(seg.value) }}{{ normalise ? ` (${seg.pct}%)` : '' }}</title>
            </rect>
            <text v-if="!normalise" :class="horizontal ? 'total h' : 'total'"
                  :x="horizontal ? g.end + 4 : g.centre"
                  :y="horizontal ? g.centre + 3 : g.end - 4"
                  :text-anchor="horizontal ? 'start' : 'middle'">{{ format(g.total) }}</text>
            <text v-if="gi === -1" /> <!-- keeps the key stable when groups change -->
          </g>
        </svg>
      </div>

      <div class="legend">
        <span v-for="k in keys" :key="k" class="lg">
          <span class="sw" :style="{ background: colors[k] }"></span>{{ k }}
        </span>
        <span v-if="hasOther" class="lg">
          <span class="sw" :style="{ background: OTHER_COLOR }"></span>Other
        </span>
      </div>
    </div>
  </figure>
</template>

<script setup>
import { computed } from 'vue'
import { useChartSize } from '~/composables/useChartSize'

const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, short, values: number[], other: number, total: number }]
  groups: { type: Array, default: () => [] },
  keys: { type: Array, default: () => [] },
  colors: { type: Object, default: () => ({}) },
  horizontal: { type: Boolean, default: false },
  // Stretch every bar to full length, so the chart compares composition rather
  // than size — the two questions a stacked bar gets asked.
  normalise: { type: Boolean, default: false },
  format: { type: Function, default: (v) => String(Math.round(v)) },
})

const OTHER_COLOR = '#9aa0a6'
const { container, width: W } = useChartSize()

const padL = computed(() => (props.horizontal ? 130 : 44))
const padR = 44
const padT = 10
const padB = computed(() => (props.horizontal ? 24 : 48))

const BAND = 26
// Grows with the number of groups when horizontal, so labels never overlap.
const H = computed(() => (props.horizontal
  ? Math.max(220, padT + padB.value + props.groups.length * BAND)
  : 300))

const hasOther = computed(() => props.groups.some((g) => (g.other || 0) > 0))
const max = computed(() => Math.max(1, ...props.groups.map((g) => g.total || 0)))

const viewportStyle = computed(() => (props.horizontal ? { height: `${H.value}px` } : {}))

const scaled = computed(() => {
  const n = props.groups.length || 1
  const innerW = Math.max(10, W.value - padL.value - padR)
  const innerH = H.value - padT - padB.value
  const band = (props.horizontal ? innerH : innerW) / n
  const thickness = Math.min(34, band * 0.72)

  return props.groups.map((g, i) => {
    const centre = (props.horizontal ? padT : padL.value) + band * (i + 0.5)
    const total = g.total || 0
    // Normalised bars divide by their own total, so an empty group stays empty
    // rather than dividing by zero into NaN geometry.
    const scaleOf = (v) => {
      if (props.normalise) return total ? v / total : 0
      return max.value ? v / max.value : 0
    }
    const full = props.horizontal ? innerW : innerH

    const parts = [...props.keys.map((k, ki) => ({ key: k, value: g.values[ki] || 0, color: props.colors[k] }))]
    if ((g.other || 0) > 0) parts.push({ key: 'Other', value: g.other, color: OTHER_COLOR })

    let run = 0
    const segments = parts.filter((s) => s.value > 0).map((s) => {
      const len = scaleOf(s.value) * full
      const start = run
      run += len
      return {
        ...s,
        pct: total ? Math.round((s.value / total) * 100) : 0,
        x: props.horizontal ? padL.value + start : centre - thickness / 2,
        y: props.horizontal ? centre - thickness / 2 : (H.value - padB.value) - start - len,
        w: props.horizontal ? len : thickness,
        h: props.horizontal ? thickness : len,
      }
    })
    return {
      ...g,
      centre,
      segments,
      end: props.horizontal ? padL.value + run : (H.value - padB.value) - run,
    }
  })
})
</script>

<style scoped>
.chart { margin: 0; display: flex; flex-direction: column; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 10px; }
.chart-area { position: relative; overflow: auto; max-width: 100%; border-radius: 8px; }
.chart-viewport { position: relative; display: block; min-width: 100%; min-height: 100%; }
svg { width: 100%; height: 100%; display: block; }

.seg { stroke: var(--surface); stroke-width: 0.5; }
.tick { fill: var(--muted); font-size: 10px; }
.label { font-size: 10px; }
.total { fill: var(--text); font-size: 10px; font-weight: 600; }

.legend {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px;
  font-size: 0.78rem; color: var(--text); max-height: 74px; overflow-y: auto;
}
.legend .lg { display: inline-flex; align-items: center; gap: 6px; }
.legend .sw { width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
