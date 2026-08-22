<template>
  <div class="leadup">
    <!-- Rain: daily precipitation over the 7 days up to the observation -->
    <div v-if="rain.some((d) => d.value !== null)" class="lc">
      <div class="lc-title">Rain leading up (mm)</div>
      <svg :viewBox="`0 0 ${W} ${Hr}`" class="lc-svg" preserveAspectRatio="none">
        <line :x1="0" :y1="Hr - axB" :x2="W" :y2="Hr - axB" class="axis" />
        <g v-for="(d, i) in rain" :key="i">
          <rect v-if="d.value !== null"
                :x="x(i) - bw / 2" :y="rainY(d.value)" :width="bw"
                :height="Math.max(1, (Hr - axB) - rainY(d.value))" rx="2" class="rain-bar">
            <title>{{ d.label }}: {{ d.value.toFixed(1) }} mm</title>
          </rect>
          <text :x="x(i)" :y="Hr - 3" class="lc-cat">{{ d.short }}</text>
        </g>
      </svg>
    </div>

    <!-- Temperature: daily min–max band + high line -->
    <div v-if="hasTempHistory" class="lc">
      <div class="lc-title">Temp leading up (°C, min–max)</div>
      <svg :viewBox="`0 0 ${W} ${Ht}`" class="lc-svg" preserveAspectRatio="none">
        <polygon :points="bandPoints" class="temp-band" />
        <polyline :points="highLine" class="temp-high" />
        <polyline :points="lowLine" class="temp-low" />
        <g v-for="(d, i) in temp" :key="i">
          <text :x="x(i)" :y="Ht - 3" class="lc-cat">{{ d.short }}</text>
        </g>
      </svg>
      <div class="lc-scale">{{ tMin.toFixed(0) }}° – {{ tMax.toFixed(0) }}°</div>
    </div>

    <!-- Fallback: observation-day temp only (daily history not yet in the data) -->
    <div v-else-if="hasDayTemp" class="lc">
      <div class="lc-title">Temperature (observation day)</div>
      <div class="day-temp">
        <span>low {{ fmt(p.tmin) }}°</span>
        <span class="avg">avg {{ fmt(p.tavg) }}°</span>
        <span>high {{ fmt(p.tmax) }}°</span>
      </div>
      <div class="lc-note">Daily temperature history fills in on the next pipeline run.</div>
    </div>
  </div>
</template>

<script setup>
const props = defineProps({ p: { type: Object, required: true } })
const p = computed(() => props.p)

const W = 240
const Hr = 90
const Ht = 90
const axB = 16
const bw = 20
const n = 7

// x position for day column i (i = 0 is 6 days before … i = 6 is the day-of)
function x(i) { return 14 + (i * (W - 28)) / (n - 1) }

const has = (v) => v !== null && v !== undefined && v !== ''
const fmt = (v) => (has(v) ? Number(v).toFixed(0) : '—')

// Rain: prcp_d6 (oldest) … prcp_d0 (observation day)
const rain = computed(() => Array.from({ length: n }, (_, i) => {
  const offset = n - 1 - i // i=0 -> d6, i=6 -> d0
  const v = p.value[`prcp_d${offset}`]
  return { value: has(v) ? Number(v) : null, label: dayLabel(offset), short: offset === 0 ? '0' : `-${offset}` }
}))
const rainMax = computed(() => Math.max(1, ...rain.value.map((d) => d.value ?? 0)))
function rainY(v) { return (Hr - axB) - (v / rainMax.value) * (Hr - axB - 6) }

// Temperature history: tmax_d6..d0 / tmin_d6..d0
const temp = computed(() => Array.from({ length: n }, (_, i) => {
  const offset = n - 1 - i
  const hi = p.value[`tmax_d${offset}`]
  const lo = p.value[`tmin_d${offset}`]
  return { hi: has(hi) ? Number(hi) : null, lo: has(lo) ? Number(lo) : null, short: offset === 0 ? '0' : `-${offset}` }
}))
const hasTempHistory = computed(() => temp.value.some((d) => d.hi !== null))
const hasDayTemp = computed(() => has(p.value.tmax) || has(p.value.tavg))

const tMin = computed(() => Math.min(...temp.value.flatMap((d) => [d.lo].filter((v) => v !== null))))
const tMax = computed(() => Math.max(...temp.value.flatMap((d) => [d.hi].filter((v) => v !== null))))
function tempY(v) {
  const lo = tMin.value, hi = tMax.value
  const span = hi - lo || 1
  return 8 + (1 - (v - lo) / span) * (Ht - axB - 8)
}
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
.leadup { display: flex; flex-direction: column; gap: 12px; }
.lc-title { font-size: 0.78rem; font-weight: 600; color: #374151; margin-bottom: 3px; }
.lc-svg { width: 100%; height: 90px; display: block; }
.axis { stroke: #d1d5db; stroke-width: 1; }
.rain-bar { fill: #2a78d6; }
.temp-band { fill: rgba(235, 104, 52, 0.15); stroke: none; }
.temp-high { fill: none; stroke: #eb6834; stroke-width: 2; }
.temp-low { fill: none; stroke: #2a78d6; stroke-width: 1.5; stroke-dasharray: 3 2; }
.lc-cat { fill: #9aa0a6; font-size: 9px; text-anchor: middle; }
.lc-scale { font-size: 0.72rem; color: #6b7280; text-align: right; }
.day-temp { display: flex; gap: 12px; font-size: 0.85rem; color: #374151; }
.day-temp .avg { font-weight: 600; }
.lc-note { font-size: 0.72rem; color: #9aa0a6; margin-top: 4px; }
</style>
