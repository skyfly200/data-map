<template>
  <div class="dash-widget now-fruiting">
    <div class="widget-head">
      <h3 class="widget-title">🍄 Now Fruiting</h3>
      <NuxtLink :to="modelLink" class="widget-link">{{ modelLinkLabel }} ›</NuxtLink>
    </div>

    <p class="season-label">{{ seasonLabel }}</p>

    <div v-if="loading" class="nf-loading">Loading…</div>

    <p v-else-if="!topSpecies.length" class="nf-empty">
      No observations found for this time of year.
    </p>

    <ul v-else class="nf-list">
      <li v-for="s in topSpecies" :key="s.name" class="nf-item">
        <span class="nf-name">{{ s.name }}</span>
        <span class="nf-count">{{ s.count.toLocaleString() }}</span>
      </li>
    </ul>

    <p v-if="latestModel" class="nf-model">
      <span class="nf-model-label">Latest model:</span>
      <NuxtLink :to="modelLink" class="nf-model-link">{{ latestModel.title }}</NuxtLink>
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

// Window of ±30 days around today's day-of-year, wrapping at year boundaries.
const WINDOW = 30
const TOP_N = 8

function dayOfYear(date) {
  const start = new Date(date.getFullYear(), 0, 0)
  return Math.floor((date - start) / 86400000)
}

const today = new Date()
const todayDoy = dayOfYear(today)

function isInWindow(month, day) {
  if (!month || !day) return false
  // Reconstruct a representative DOY from month/day (non-leap).
  const probe = new Date(2001, month - 1, day)
  const doy = dayOfYear(probe)
  let diff = Math.abs(doy - todayDoy)
  if (diff > 182) diff = 365 - diff  // wrap
  return diff <= WINDOW
}

const topSpecies = computed(() => {
  const counts = new Map()
  for (const r of rows.value || []) {
    if (!r.species) continue
    const month = r.month ?? (r.date ? new Date(r.date).getMonth() + 1 : null)
    const day = r.date ? new Date(r.date).getDate() : null
    if (!isInWindow(month, day)) continue
    counts.set(r.species, (counts.get(r.species) || 0) + 1)
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, TOP_N)
})

const latestModel = computed(() => {
  const sorted = [...(models.value || [])].sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
  )
  return sorted[0] || null
})

// Link to map (activating latest model layer) if a model with an asset exists,
// otherwise fall back to the modeling page.
const modelLink = computed(() => {
  if (latestModel.value?.suitability_asset_path) {
    return `/map?layer=maxent:${latestModel.value.id}`
  }
  return '/modeling/maxent'
})

const modelLinkLabel = computed(() =>
  latestModel.value?.suitability_asset_path ? 'View on map' : 'Run a model',
)

const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const seasonLabel = computed(() => {
  const m = today.getMonth()
  const prev = MONTH_NAMES[(m + 11) % 12]
  const curr = MONTH_NAMES[m]
  const next = MONTH_NAMES[(m + 1) % 12]
  return `${prev} – ${curr} – ${next}`
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
.nf-name { font-size: 0.82rem; color: var(--text, #222); font-style: italic; }
.nf-count { font-size: 0.75rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }

.nf-model {
  margin: 0.2rem 0 0; font-size: 0.76rem; color: var(--muted, #888);
  border-top: 1px solid var(--border-soft, #eee); padding-top: 0.5rem;
}
.nf-model-label { margin-right: 0.3rem; }
.nf-model-link { color: var(--accent, #2a78d6); text-decoration: none; }
.nf-model-link:hover { text-decoration: underline; }
</style>
