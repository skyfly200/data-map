<template>
  <div class="dash-widget now-fruiting">
    <div class="widget-head">
      <h3 class="widget-title"><span aria-hidden="true">🍄 </span>Now Fruiting</h3>
      <div class="widget-actions">
        <button v-if="selectedSpecies" class="clear-btn" @click="selectedSpecies = null">✕ {{ selectedSpecies }}</button>
        <NuxtLink :to="mapLink" class="widget-link">Map ›</NuxtLink>
        <NuxtLink to="/modeling/maxent" class="widget-link">Model ›</NuxtLink>
      </div>
    </div>

    <p class="season-label">{{ seasonLabel }}</p>

    <div v-if="loading" class="nf-loading">Loading…</div>

    <p v-else-if="!topSpecies.length" class="nf-empty">
      No observations found for this time of year.
    </p>

    <ul v-else class="nf-list">
      <li v-for="s in topSpecies" :key="s.name" class="nf-item"
          :class="{ selected: selectedSpecies === s.name, 'has-model': s.hasModel }"
          @click="toggleSpecies(s.name)">
        <span class="nf-name">{{ s.name }}</span>
        <span class="nf-meta">
          <span v-if="s.hasModel" class="nf-model-badge" title="MaxEnt model available">ML</span>
          <span class="nf-peak" :title="`Peak ${s.peakLabel} from today · ${s.iqr}d IQR`">{{ s.peakLabel }}</span>
          <span class="nf-iqr" :title="`Fruiting window width (IQR): ${s.iqr} days`">{{ s.iqr }}d</span>
          <span v-if="s.elevBand" class="nf-elev"
                :title="`Typical elevation band (IQR): ${elevLabel(s.elevBand.loM)}–${elevLabel(s.elevBand.hiM)}`">
            {{ elevLabel(s.elevBand.loM) }}–{{ elevLabel(s.elevBand.hiM) }}
          </span>
          <span class="nf-count">{{ s.count.toLocaleString() }}</span>
        </span>
      </li>
    </ul>

    <p v-if="latestModel" class="nf-model">
      <span class="nf-model-label">Latest model:</span>
      <NuxtLink :to="mapLink" class="nf-model-link">{{ latestModel.title }}</NuxtLink>
    </p>
    <p v-else class="nf-model">
      <NuxtLink to="/modeling/maxent" class="nf-model-link">No models yet — run one ›</NuxtLink>
    </p>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useMaxEnt } from '~/composables/useMaxEnt'
import { useUnits } from '~/composables/useUnits'

const { rows, load } = useObservations()
const { models, fetchModels } = useMaxEnt()
const { elevLabel } = useUnits()

const loading = ref(true)
const selectedSpecies = ref(null)

const WINDOW = 30
const TOP_N = 8

const today = new Date()
const todayDoy = dayOfYear(today)

function dayOfYear(date) {
  const start = new Date(date.getFullYear(), 0, 0)
  return Math.floor((date - start) / 86400000)
}

function doyDist(a, b) {
  const d = Math.abs(a - b)
  return d > 182 ? 365 - d : d
}

function obsDoy(r) {
  if (r.date) {
    const d = new Date(r.date)
    if (!isNaN(d)) return dayOfYear(d)
  }
  const month = r.month ?? null
  if (month) return dayOfYear(new Date(2001, month - 1, 15))
  return null
}

// Species names that have at least one model
const modelSpeciesSet = computed(() => {
  const s = new Set()
  for (const m of models.value || []) {
    if (m.species) s.add(m.species)
    if (m.target_species) s.add(m.target_species)
  }
  return s
})

const topSpecies = computed(() => {
  const buckets = new Map()
  for (const r of rows.value || []) {
    if (!r.species) continue
    const doy = obsDoy(r)
    if (doy === null || doyDist(doy, todayDoy) > WINDOW) continue
    if (!buckets.has(r.species)) buckets.set(r.species, { count: 0, doys: [], elevs: [] })
    const b = buckets.get(r.species)
    b.count++
    b.doys.push(doy)
    const elev = Number(r.elevation)
    if (Number.isFinite(elev)) b.elevs.push(elev)
  }

  return [...buckets.entries()]
    .map(([name, { count, doys, elevs }]) => {
      const sorted = [...doys].sort((a, b) => a - b)
      const n = sorted.length
      const mid = Math.floor(n / 2)
      const medianDoy = n % 2 === 0
        ? Math.round((sorted[mid - 1] + sorted[mid]) / 2)
        : sorted[mid]
      const q1 = sorted[Math.floor(n * 0.25)]
      const q3 = sorted[Math.floor(n * 0.75)]
      const iqr = q3 - q1
      const dist = doyDist(medianDoy, todayDoy)
      const score = dist + iqr * 0.5
      const peakLabel = dist === 0 ? 'today' : `±${dist}d`
      let elevBand = null
      if (elevs.length >= 3) {
        const es = [...elevs].sort((a, b) => a - b)
        elevBand = { loM: es[Math.floor(es.length * 0.25)], hiM: es[Math.floor(es.length * 0.75)] }
      }
      const hasModel = modelSpeciesSet.value.has(name)
      return { name, count, dist, iqr, score, peakLabel, elevBand, hasModel }
    })
    .sort((a, b) => a.score - b.score || b.count - a.count)
    .slice(0, TOP_N)
})

function toggleSpecies(name) {
  selectedSpecies.value = selectedSpecies.value === name ? null : name
}

const latestModel = computed(() => {
  const base = (models.value || [])
  const filtered = selectedSpecies.value
    ? base.filter((m) => m.species === selectedSpecies.value || m.target_species === selectedSpecies.value)
    : base
  return [...filtered]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0] || null
})

const mapLink = computed(() => {
  const m = latestModel.value
  const base = m ? `/map?layer=maxent:${m.id}` : '/map'
  return selectedSpecies.value
    ? `${base}${base.includes('?') ? '&' : '?'}species=${encodeURIComponent(selectedSpecies.value)}`
    : base
})

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const seasonLabel = computed(() => {
  const m = today.getMonth()
  return `${MONTH_NAMES[(m + 11) % 12]} – ${MONTH_NAMES[m]} – ${MONTH_NAMES[(m + 1) % 12]}`
})

onMounted(async () => {
  await Promise.all([load(), fetchModels()])
  loading.value = false
})
</script>

<style scoped>
.now-fruiting { height: 100%; display: flex; flex-direction: column; gap: 0.6rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.4rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.clear-btn {
  font: inherit; font-size: 0.7rem; border: 1px solid var(--accent, #2a78d6);
  background: color-mix(in srgb, var(--accent, #2a78d6) 12%, transparent);
  color: var(--accent, #2a78d6); border-radius: 999px;
  padding: 0.1rem 0.5rem; cursor: pointer; max-width: 160px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.season-label { margin: 0; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted, #888); }

.nf-loading, .nf-empty { font-size: 0.85rem; color: var(--muted, #888); }

.nf-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; flex: 1; overflow-y: auto; }
.nf-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.3rem 0.5rem; background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px;
  cursor: pointer; transition: border-color 0.15s, background 0.15s;
}
.nf-item:hover { border-color: var(--accent, #2a78d6); }
.nf-item.selected { border-color: var(--accent, #2a78d6); background: color-mix(in srgb, var(--accent, #2a78d6) 8%, var(--surface-2, #f5f5f5)); }
.nf-item.has-model .nf-name { color: var(--accent, #2a78d6); }

.nf-name { font-size: 0.82rem; color: var(--text, #222); font-style: italic; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nf-meta { display: flex; gap: 0.5rem; align-items: center; flex-shrink: 0; }
.nf-model-badge {
  font-size: 0.6rem; font-weight: 700; color: #fff;
  background: var(--accent, #2a78d6); border-radius: 3px; padding: 0 3px;
  font-style: normal;
}
.nf-peak { font-size: 0.7rem; color: var(--accent, #2a78d6); font-variant-numeric: tabular-nums; white-space: nowrap; }
.nf-iqr { font-size: 0.7rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; white-space: nowrap; }
.nf-elev { font-size: 0.68rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; white-space: nowrap; opacity: 0.8; }
.nf-count { font-size: 0.75rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }

.nf-model {
  margin: 0.2rem 0 0; font-size: 0.76rem; color: var(--muted, #888);
  border-top: 1px solid var(--border-soft, #eee); padding-top: 0.5rem;
}
.nf-model-label { margin-right: 0.3rem; }
.nf-model-link { color: var(--accent, #2a78d6); text-decoration: none; }
.nf-model-link:hover { text-decoration: underline; }
</style>
