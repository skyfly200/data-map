<template>
  <div class="leadup">
    <!-- Rain: daily precipitation over the 7 days up to the observation -->
    <div v-if="rain.some((d) => d.value !== null)" class="lc">
      <div class="lc-title">Rain leading up (mm)</div>
      <svg :viewBox="`0 0 ${W} ${H}`" class="lc-svg">
        <g v-for="t in rainTicks" :key="`rt${t.v}`">
          <line :x1="axL" :y1="t.y" :x2="W - padR" :y2="t.y" class="grid" />
          <text :x="axL - 5" :y="t.y + 3" class="lc-y">{{ t.label }}</text>
        </g>
        <g v-for="(d, i) in rain" :key="i">
          <rect v-if="d.value !== null"
                :x="x(i) - bw / 2" :y="rainY(d.value)" :width="bw"
                :height="Math.max(1, yB - rainY(d.value))" rx="2" class="rain-bar">
            <title>{{ d.label }}: {{ d.value.toFixed(1) }} mm</title>
          </rect>
          <text :x="x(i)" :y="H - 3" class="lc-cat">{{ d.short }}</text>
        </g>
      </svg>
    </div>

    <!-- Temperature: daily min–max band + high/low lines -->
    <div v-if="hasTempHistory" class="lc">
      <div class="lc-title">Temp leading up (°{{ tempUnit }}, min–max)</div>
      <svg :viewBox="`0 0 ${W} ${H}`" class="lc-svg">
        <g v-for="t in tempTicks" :key="`tt${t.v}`">
          <line :x1="axL" :y1="t.y" :x2="W - padR" :y2="t.y" class="grid" />
          <text :x="axL - 5" :y="t.y + 3" class="lc-y">{{ t.label }}</text>
        </g>
        <polygon :points="bandPoints" class="temp-band" />
        <polyline :points="highLine" class="temp-high" />
        <polyline :points="lowLine" class="temp-low" />
        <text v-for="(d, i) in temp" :key="i" :x="x(i)" :y="H - 3" class="lc-cat">{{ d.short }}</text>
      </svg>
      <div class="lc-legend"><span class="k high"></span>high <span class="k low"></span>low · days before</div>
    </div>

    <!-- Fallback: observation-day temp only (daily history not yet in the data) -->
    <div v-else-if="hasDayTemp" class="lc">
      <div class="lc-title">Temperature (observation day)</div>
      <div class="day-temp">
        <span>low {{ tempLabel(p.tmin) }}</span>
        <span class="avg">avg {{ tempLabel(p.tavg) }}</span>
        <span>high {{ tempLabel(p.tmax) }}</span>
      </div>
      <div class="lc-note">Daily temperature history fills in on the next pipeline run.</div>
    </div>
  </div>
</template>

<script setup>
import { useUnits } from '~/composables/useUnits'

const props = defineProps({ p: { type: Object, required: true } })
const p = computed(() => props.p)
const { tempUnit, tempValue, tempLabel } = useUnits()

const W = 264
const H = 100
const padT = 8   // top padding
const axB = 16   // bottom gutter (day labels)
const axL = 34   // left gutter (y labels)
const padR = 8   // right padding
const bw = 16
const n = 7

const yT = padT
const yB = H - axB

// x position for day column i (i = 0 is 6 days before … i = 6 is the day-of)
function x(i) { return axL + (i * (W - axL - padR)) / (n - 1) }

const has = (v) => v !== null && v !== undefined && v !== ''

// Rain: prcp_d6 (oldest) … prcp_d0 (observation day)
const rain = computed(() => Array.from({ length: n }, (_, i) => {
  const offset = n - 1 - i
  const v = p.value[`prcp_d${offset}`]
  return { value: has(v) ? Number(v) : null, label: dayLabel(offset), short: offset === 0 ? '0' : `-${offset}` }
}))
const rainMax = computed(() => Math.max(1, ...rain.value.map((d) => d.value ?? 0)))
function rainY(v) { return yB - (v / rainMax.value) * (yB - yT) }
const rainTicks = computed(() => {
  const m = rainMax.value
  return [0, m / 2, m].map((v) => ({ v, y: rainY(v), label: m >= 10 ? String(Math.round(v)) : v.toFixed(1) }))
})

// Temperature history: tmax_d6..d0 / tmin_d6..d0, converted to the display unit.
const temp = computed(() => Array.from({ length: n }, (_, i) => {
  const offset = n - 1 - i
  return {
    hi: tempValue(p.value[`tmax_d${offset}`]),
    lo: tempValue(p.value[`tmin_d${offset}`]),
    short: offset === 0 ? '0' : `-${offset}`,
  }
}))
const hasTempHistory = computed(() => temp.value.some((d) => d.hi !== null))
const hasDayTemp = computed(() => has(p.value.tmax) || has(p.value.tavg))

const tMin = computed(() => Math.min(...temp.value.flatMap((d) => (d.lo === null ? [] : [d.lo]))))
const tMax = computed(() => Math.max(...temp.value.flatMap((d) => (d.hi === null ? [] : [d.hi]))))
function tempY(v) {
  const span = (tMax.value - tMin.value) || 1
  return yT + (1 - (v - tMin.value) / span) * (yB - yT)
}
const tempTicks = computed(() => {
  const lo = tMin.value, hi = tMax.value, mid = (lo + hi) / 2
  return [hi, mid, lo].map((v) => ({ v, y: tempY(v), label: `${Math.round(v)}°` }))
})
const highLine = computed(() => temp.value.map((d, i) => (d.hi === null ? null : `${x(i)},${tempY(d.hi)}`)).filter(Boolean).join(' '))
const lowLine = computed(() => temp.value.map((d, i) => (d.lo === null ? null : `${x(i)},${tempY(d.lo)}`)).filter(Boolean).join(' '))
const bandPoints = computed(() => {
  const top = temp.value.map((d, i) => (d.hi === null ? null : `${x(i)},${tempY(d.hi)}`)).filter(Boolean)
  const bot = temp.value.map((d, i) => (d.lo === null ? null : `${x(i)},${tempY(d.lo)}`)).filter(Boolean).reverse()
  return [...top, ...bot].join(' ')
})

function dayLabel(offset) { return offset === 0 ? 'observation day' : `${offset} day${offset > 1 ? 's' : ''} before` }
</script>

<style scoped>
.leadup { display: flex; flex-direction: column; gap: 14px; }
.lc-title { font-size: 0.78rem; font-weight: 600; color: #374151; margin-bottom: 3px; }
.lc-svg { width: 100%; height: auto; display: block; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.rain-bar { fill: #2a78d6; }
.temp-band { fill: rgba(235, 104, 52, 0.15); stroke: none; }
.temp-high { fill: none; stroke: #eb6834; stroke-width: 2; }
.temp-low { fill: none; stroke: #2a78d6; stroke-width: 1.5; stroke-dasharray: 3 2; }
.lc-cat { fill: #9aa0a6; font-size: 9px; text-anchor: middle; }
.lc-y { fill: #9aa0a6; font-size: 9px; text-anchor: end; }
.lc-legend { font-size: 0.72rem; color: #6b7280; display: flex; align-items: center; gap: 5px; }
.lc-legend .k { display: inline-block; width: 12px; height: 3px; border-radius: 2px; }
.lc-legend .k.high { background: #eb6834; }
.lc-legend .k.low { background: #2a78d6; }
.day-temp { display: flex; gap: 12px; font-size: 0.85rem; color: #374151; }
.day-temp .avg { font-weight: 600; }
.lc-note { font-size: 0.72rem; color: #9aa0a6; margin-top: 4px; }
</style>
