<template>
  <figure class="chart">
    <figcaption v-if="title" class="chart-title">{{ title }}</figcaption>
    <div ref="areaEl" class="chart-area" @mousemove="onMove" @mouseleave="hoveredLine = null">
      <svg :viewBox="`0 0 ${W} ${H}`" role="img" :aria-label="title">
        <!-- Axis lines -->
        <g v-for="ax in axes" :key="ax.key">
          <line :x1="ax.x" :y1="padT" :x2="ax.x" :y2="H - padB" class="axis-line" />
          <text :x="ax.x" :y="padT - 4" class="axis-label-top">{{ ax.label }}</text>
          <text v-for="t in ax.ticks" :key="t.v" :x="ax.x - 4" :y="t.py" class="tick tick-y">{{ t.label }}</text>
        </g>

        <!-- Lines: unlit first, then lit on top -->
        <g class="lines-dim">
          <path v-for="(ln, i) in scaledLines" :key="i"
                v-show="hoveredLine === null || hoveredLine !== i"
                :d="ln.d" fill="none" :stroke="ln.color" stroke-width="0.8" stroke-opacity="0.25"
                class="pline" @mouseenter="hoveredLine = i" />
        </g>
        <g class="lines-hot">
          <path v-if="hoveredLine !== null && scaledLines[hoveredLine]"
                :d="scaledLines[hoveredLine].d"
                fill="none" :stroke="scaledLines[hoveredLine].color"
                stroke-width="1.8" stroke-opacity="0.95"
                class="pline-hot" />
        </g>
      </svg>

      <div v-if="hoveredLine !== null && scaledLines[hoveredLine]" class="tooltip"
           :style="{ left: `${ptr.x + 12}px`, top: `${ptr.y + 8}px` }">
        <strong>{{ scaledLines[hoveredLine].label }}</strong>
        <span v-for="ax in axes" :key="ax.key">{{ ax.label }}: {{ ax.fmt(lines[hoveredLine]?.vals[ax.key]) }}</span>
      </div>
    </div>

    <div v-if="legend.length" class="pckey">
      <span v-for="l in legend" :key="l.label" class="k">
        <span class="sw" :style="{ background: l.color }"></span>{{ l.label }}
      </span>
    </div>
  </figure>
</template>

<script setup>
const props = defineProps({
  title: { type: String, default: '' },
  // [{ label, color, vals: { [fieldKey]: number } }]
  lines: { type: Array, required: true },
  // [{ key, label, format }]
  axes: { type: Array, required: true },
  // [{ label, color }]
  legend: { type: Array, default: () => [] },
})

const W = 640
const H = 320
const padT = 32
const padB = 24
const padL = 32
const padR = 32

const areaEl = ref(null)
const hoveredLine = ref(null)
const ptr = ref({ x: 0, y: 0 })

function onMove(e) {
  const r = e.currentTarget.getBoundingClientRect()
  ptr.value = { x: e.clientX - r.left, y: e.clientY - r.top }
}

// Per-axis domain from data
const axisDomains = computed(() => {
  const out = {}
  for (const ax of props.axes) {
    const vals = props.lines.map((l) => l.vals[ax.key]).filter((v) => Number.isFinite(v))
    if (!vals.length) { out[ax.key] = [0, 1]; continue }
    let lo = Math.min(...vals), hi = Math.max(...vals)
    if (lo === hi) { lo -= 1; hi += 1 }
    const pad = (hi - lo) * 0.05
    out[ax.key] = [lo - pad, hi + pad]
  }
  return out
})

const axesWithX = computed(() => {
  const n = props.axes.length
  if (!n) return []
  const usable = W - padL - padR
  return props.axes.map((ax, i) => {
    const [lo, hi] = axisDomains.value[ax.key] || [0, 1]
    const x = padL + (n > 1 ? (usable * i) / (n - 1) : usable / 2)
    const sy = (v) => padT + (H - padT - padB) * (1 - (v - lo) / ((hi - lo) || 1))
    const ticks = Array.from({ length: 5 }, (_, j) => {
      const v = lo + ((hi - lo) * j) / 4
      return { v, py: sy(v), label: (ax.format || ((v) => Number.isFinite(v) ? Number(v).toFixed(1) : ''))(v) }
    })
    return { ...ax, x, sy, ticks, fmt: ax.format || ((v) => Number.isFinite(v) ? Number(v).toFixed(1) : '') }
  })
})

// Expose axes with x for the template (override computed axes prop)
const axes = axesWithX

const scaledLines = computed(() => {
  return props.lines.map((ln) => {
    const pts = axesWithX.value
      .map((ax) => {
        const v = ln.vals[ax.key]
        if (!Number.isFinite(v)) return null
        return { x: ax.x, y: ax.sy(v) }
      })
      .filter(Boolean)

    if (pts.length < 2) return null

    let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`
    for (let i = 1; i < pts.length; i++) {
      const cp = ((pts[i - 1].x + pts[i].x) / 2).toFixed(1)
      d += ` C ${cp} ${pts[i - 1].y.toFixed(1)} ${cp} ${pts[i].y.toFixed(1)} ${pts[i].x.toFixed(1)} ${pts[i].y.toFixed(1)}`
    }
    return { d, color: ln.color, label: ln.label }
  }).filter(Boolean)
})
</script>

<style scoped>
.chart { margin: 0; }
.chart-title { font-size: 0.95rem; font-weight: 600; color: var(--text); margin-bottom: 6px; }
.chart-area {
  position: relative; overflow: auto; max-width: 100%; border-radius: 8px;
  background: linear-gradient(180deg, rgba(148, 163, 184, 0.03), rgba(148, 163, 184, 0.01));
  user-select: none;
}
svg { width: 100%; height: auto; display: block; }
.axis-line { stroke: var(--border); stroke-width: 1.5; }
.axis-label-top { fill: var(--text); font-size: 10px; text-anchor: middle; font-weight: 600; }
.tick { fill: var(--muted); font-size: 9px; text-anchor: end; }
.pline { cursor: default; }
.pline-hot { pointer-events: none; }
.tooltip {
  position: absolute; pointer-events: none; z-index: 10; display: flex; flex-direction: column;
  background: var(--tooltip-bg); color: var(--tooltip-fg); padding: 5px 8px; border-radius: 6px;
  font-size: 0.75rem; white-space: nowrap; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
}
.tooltip strong { margin-bottom: 2px; }
.pckey {
  display: flex; flex-wrap: wrap; gap: 4px 12px; margin-top: 8px;
  font-size: 0.75rem; color: var(--text); max-height: 72px; overflow-y: auto;
}
.pckey .k { display: inline-flex; align-items: center; gap: 5px; }
.pckey .sw { width: 11px; height: 11px; border-radius: 3px; border: 1px solid var(--border); flex: 0 0 auto; }
</style>
