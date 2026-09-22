<template>
  <div class="dash-widget overview">
    <div class="widget-head">
      <h3 class="widget-title">📌 Overview</h3>
      <NuxtLink to="/map" class="widget-link">Open map ›</NuxtLink>
    </div>

    <div class="tiles">
      <NuxtLink to="/data" class="tile">
        <span class="tile-value">{{ total.toLocaleString() }}</span>
        <span class="tile-label">Observations</span>
      </NuxtLink>
      <NuxtLink to="/charts" class="tile">
        <span class="tile-value">{{ speciesCount.toLocaleString() }}</span>
        <span class="tile-label">Species</span>
      </NuxtLink>
      <NuxtLink to="/jobs" class="tile">
        <span class="tile-value">{{ jobCount.toLocaleString() }}</span>
        <span class="tile-label">Jobs run</span>
      </NuxtLink>
      <div class="tile">
        <span class="tile-value small">{{ lastActivity }}</span>
        <span class="tile-label">Last activity</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useEeJobs } from '~/composables/useEeJobs'

const { rows, load } = useObservations()
const { jobs, refresh } = useEeJobs()

const total = computed(() => (rows.value || []).length)
const speciesCount = computed(() => {
  const set = new Set()
  for (const r of rows.value || []) if (r.species) set.add(r.species)
  return set.size
})
const jobCount = computed(() => (jobs.value || []).length)

const lastActivity = computed(() => {
  const at = (jobs.value || []).map((j) => j.created_at).filter(Boolean).sort().pop()
  if (!at) return '—'
  const days = Math.floor((Date.now() - new Date(at).getTime()) / 86400000)
  if (!Number.isFinite(days)) return '—'
  if (days <= 0) return 'today'
  if (days === 1) return 'yesterday'
  if (days < 30) return `${days}d ago`
  return `${Math.floor(days / 30)}mo ago`
})

onMounted(() => { load(); refresh() })
</script>

<style scoped>
.overview { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 0.6rem; }
.tile {
  display: flex; flex-direction: column; gap: 0.15rem; padding: 0.7rem 0.8rem; text-decoration: none;
  background: var(--surface-2, #f5f5f5); border: 1px solid var(--border-soft, #eee); border-radius: 8px;
}
a.tile:hover { border-color: var(--accent, #2a78d6); }
.tile-value { font-size: 1.35rem; font-weight: 700; color: var(--text, #111); font-variant-numeric: tabular-nums; }
.tile-value.small { font-size: 1rem; }
.tile-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted, #777); }
</style>
