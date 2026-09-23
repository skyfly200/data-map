<template>
  <div class="dash-widget now-fruiting">
    <div class="widget-head">
      <h3 class="widget-title">🍄 Now Fruiting</h3>
      <div class="widget-actions">
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
      <li v-for="s in topSpecies" :key="s.name" class="nf-item">
        <span class="nf-name">{{ s.name }}</span>
        <span class="nf-meta">
          <span class="nf-peak" :title="`Peak ${s.peakLabel} days from today`">{{ s.peakLabel }}</span>
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

const { rows, load } = useObservations()
const { models, fetchModels } = useMaxEnt()

const loading = ref(true)

// ±30-day window around today's DOY, year-wrapped.
const WINDOW = 30
const TOP_N = 8

const today = new Date()
const todayDoy = dayOfYear(today)

function dayOfYear(date) {
  const start = new Date(date.getFullYear(), 0, 0)
  return Math.floor((date - start) / 86400000)
}

// Circular distance between two DOYs (0–365).
function doyDist(a, b) {
  const d = Math.abs(a - b)
  return d > 182 ? 365 - d : d
}

function obsDoy(r) {
  if (r.date) {
    const d = new Date(r.date)
    if (!isNaN(d)) return dayOfYear(d)
  }
  // Fall back to month-only: use the 15th as representative.
  const month = r.month ?? null
  if (month) return dayOfYear(new Date(2001, month - 1, 15))
  return null
}

const topSpecies = computed(() => {
  // Bucket observations by species, keeping only those inside the window.
  const buckets = new Map() // name -> { count, doys[] }
  for (const r of rows.value || []) {
    if (!r.species) continue
    const doy = obsDoy(r)
    if (doy === null || doyDist(doy, todayDoy) > WINDOW) continue
    if (!buckets.has(r.species)) buckets.set(r.species, { count: 0, doys: [] })
    const b = buckets.get(r.species)
    b.count++
    b.doys.push(doy)
  }

  // For each species compute the median DOY of windowed observations, then
  // rank by proximity of that median to today (peak overlap first), breaking
  // ties by observation count.
  return [...buckets.entries()]
    .map(([name, { count, doys }]) => {
      const sorted = [...doys].sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      const medianDoy = sorted.length % 2 === 0
        ? Math.round((sorted[mid - 1] + sorted[mid]) / 2)
        : sorted[mid]
      const dist = doyDist(medianDoy, todayDoy)
      const peakLabel = dist === 0 ? 'today' : dist <= 3 ? `±${dist}d` : `±${dist}d`
      return { name, count, dist, peakLabel }
    })
    .sort((a, b) => a.dist - b.dist || b.count - a.count)
    .slice(0, TOP_N)
})

const latestModel = computed(() => {
  return [...(models.value || [])]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0] || null
})

// Always link to map; activate the latest model layer when one has an asset.
const mapLink = computed(() => {
  const m = latestModel.value
  return m?.suitability_asset_path ? `/map?layer=maxent:${m.id}` : '/map'
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
.widget-head { display: flex; align-items: baseline; justify-content: space-between; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.6rem; }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }

.season-label { margin: 0; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted, #888); }

.nf-loading, .nf-empty { font-size: 0.85rem; color: var(--muted, #888); }

.nf-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; }
.nf-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.3rem 0.5rem; background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px;
}
.nf-name { font-size: 0.82rem; color: var(--text, #222); font-style: italic; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nf-meta { display: flex; gap: 0.5rem; align-items: center; flex-shrink: 0; }
.nf-peak { font-size: 0.7rem; color: var(--accent, #2a78d6); font-variant-numeric: tabular-nums; white-space: nowrap; }
.nf-count { font-size: 0.75rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }

.nf-model {
  margin: 0.2rem 0 0; font-size: 0.76rem; color: var(--muted, #888);
  border-top: 1px solid var(--border-soft, #eee); padding-top: 0.5rem;
}
.nf-model-label { margin-right: 0.3rem; }
.nf-model-link { color: var(--accent, #2a78d6); text-decoration: none; }
.nf-model-link:hover { text-decoration: underline; }
</style>
