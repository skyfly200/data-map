<template>
  <div class="coverage">
    <div class="head">
      <div>
        <h2>Raster coverage</h2>
        <p class="sub">Which environmental layers are cached, for what dates, and over what area.</p>
      </div>
      <span v-if="cov" class="generated">updated {{ cov.generated.replace('T', ' ').replace('Z', ' UTC') }}</span>
    </div>

    <p v-if="error" class="msg">No coverage summary found. Run <code>python raster_coverage.py</code> after the pipeline.</p>
    <p v-else-if="!cov" class="msg">Loading…</p>

    <template v-else>
      <div class="stats">
        <div class="stat"><span class="n">{{ cov.layers.length }}</span><span class="l">layers</span></div>
        <div class="stat"><span class="n">{{ datedCount }}</span><span class="l">dated snapshots</span></div>
        <div class="stat"><span class="n">{{ totalFiles }}</span><span class="l">files</span></div>
        <div class="stat"><span class="n">{{ mb(cov.total_bytes) }}</span><span class="l">on disk</span></div>
      </div>

      <div class="cards">
        <div v-for="l in cov.layers" :key="l.key" class="card">
          <div class="card-top">
            <span class="dot" :style="{ background: colorOf(l.key) }"></span>
            <h3>{{ l.label }}</h3>
          </div>
          <dl>
            <div><dt>Files</dt><dd>{{ l.count }}</dd></div>
            <div><dt>Dates</dt><dd>{{ l.date_range ? `${l.date_range[0]} → ${l.date_range[1]}` : 'static' }}</dd></div>
            <div><dt>Size</dt><dd>{{ mb(l.total_bytes) }}</dd></div>
            <div><dt>Extent</dt><dd class="extent">{{ bboxLabel(l.bbox) }}</dd></div>
          </dl>
        </div>
      </div>

      <template v-if="datedLayers.length && matrixDates.length">
        <h3 class="section">Coverage by date</h3>
        <p class="sub">A filled cell means that layer has data for that date.</p>
        <div class="matrix-wrap">
          <table class="matrix">
            <thead>
              <tr>
                <th class="date-col">Date</th>
                <th v-for="l in datedLayers" :key="l.key">
                  <span class="dot sm" :style="{ background: colorOf(l.key) }"></span>{{ l.label }}
                </th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="d in matrixDates" :key="d">
                <td class="date-col">{{ d }}</td>
                <td v-for="l in datedLayers" :key="l.key" class="cell">
                  <span v-if="has(l.key, d)" class="mark" :style="{ background: colorOf(l.key) }" :title="`${l.label} · ${d}`"></span>
                  <span v-else class="gap" title="no data">·</span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </template>
  </div>
</template>

<script setup>
import { PALETTE, UNCLUSTERED } from '~/composables/useObservations'

useHead({ title: 'Raster coverage · data-map' })

const cov = ref(null)
const error = ref(false)

onMounted(async () => {
  try {
    const res = await fetch('/data/coverage.json')
    if (!res.ok) throw new Error('not found')
    cov.value = await res.json()
  } catch {
    error.value = true
  }
})

const colors = {}
function colorOf(key) {
  if (!(key in colors)) colors[key] = PALETTE[Object.keys(colors).length % PALETTE.length] || UNCLUSTERED
  return colors[key]
}

const datedLayers = computed(() => (cov.value?.layers || []).filter((l) => l.dated && l.dates.length))
const datedCount = computed(() => Object.keys(cov.value?.date_index || {}).length)
const totalFiles = computed(() => (cov.value?.layers || []).reduce((s, l) => s + l.count, 0))
const matrixDates = computed(() => Object.keys(cov.value?.date_index || {}))

const dateSets = computed(() => {
  const m = {}
  for (const l of datedLayers.value) m[l.key] = new Set(l.dates)
  return m
})
function has(key, date) { return dateSets.value[key]?.has(date) }

function mb(bytes) {
  if (!bytes) return '0 MB'
  const v = bytes / 1e6
  return v >= 1000 ? `${(v / 1000).toFixed(1)} GB` : `${v.toFixed(1)} MB`
}
function bboxLabel(b) {
  if (!b) return '—'
  const ns = (v) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'N' : 'S'}`
  const ew = (v) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'E' : 'W'}`
  return `${ns(b[1])}–${ns(b[3])}, ${ew(b[0])}–${ew(b[2])}`
}
</script>

<style scoped>
.coverage { padding: 16px 18px; max-width: 960px; margin: 0 auto; }
.head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 0; color: #6b7280; font-size: 0.82rem; }
.generated { color: #9aa0a6; font-size: 0.76rem; white-space: nowrap; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 20px; }
.stat { border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px 14px; background: #fff; display: flex; flex-direction: column; gap: 2px; }
.stat .n { font-size: 1.3rem; font-weight: 700; color: #1f2933; }
.stat .l { font-size: 0.76rem; color: #6b7280; }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 28px; }
.card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 16px; background: #fff; }
.card-top { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.card-top h3 { margin: 0; font-size: 0.92rem; }
.dot { width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto; }
.dot.sm { width: 8px; height: 8px; display: inline-block; margin-right: 5px; }
dl { margin: 0; display: flex; flex-direction: column; gap: 5px; }
dl > div { display: flex; justify-content: space-between; gap: 10px; font-size: 0.82rem; }
dt { color: #6b7280; }
dd { margin: 0; color: #1f2933; font-variant-numeric: tabular-nums; text-align: right; }
.extent { font-size: 0.76rem; }

.section { margin: 4px 0 2px; font-size: 1rem; }
.matrix-wrap { overflow: auto; border: 1px solid #e5e7eb; border-radius: 10px; margin-top: 10px; max-height: 60vh; }
.matrix { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
.matrix th, .matrix td { padding: 6px 12px; border-bottom: 1px solid #f1f2f4; white-space: nowrap; }
.matrix thead th { position: sticky; top: 0; background: #f7f8fa; text-align: left; color: #374151; border-bottom: 1px solid #e5e7eb; }
.date-col { font-variant-numeric: tabular-nums; color: #4b5563; position: sticky; left: 0; background: #fff; }
.matrix thead .date-col { background: #f7f8fa; z-index: 1; }
.cell { text-align: center; }
.mark { display: inline-block; width: 12px; height: 12px; border-radius: 3px; }
.gap { color: #d1d5db; }
.msg { padding: 16px; color: #555; }
.msg code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }
</style>
