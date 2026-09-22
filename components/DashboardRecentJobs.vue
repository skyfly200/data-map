<template>
  <div class="dash-widget recent-jobs">
    <div class="widget-head">
      <h3 class="widget-title">⚙️ Recent jobs</h3>
      <NuxtLink to="/jobs" class="widget-link">All jobs ›</NuxtLink>
    </div>

    <p v-if="loading && !jobs.length" class="widget-note">Loading…</p>
    <p v-else-if="!recent.length" class="widget-note">No jobs yet. Start one from the map or the jobs page.</p>

    <ul v-else class="job-list">
      <li v-for="job in recent" :key="job.id" class="job-row">
        <span class="dot" :class="job.status"></span>
        <span class="job-name">{{ job.title || describe(job) }}</span>
        <span class="job-status" :class="job.status">{{ label(job) }}</span>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useEeJobs } from '~/composables/useEeJobs'

const { jobs, loading, refresh } = useEeJobs()

const recent = computed(() => (jobs.value || []).slice(0, 5))

function label(job) {
  if (job.status === 'running' || job.status === 'queued') return `${Math.round((job.progress || 0) * 100)}%`
  if (job.status === 'succeeded') return 'Done'
  if (job.status === 'cancelled') return 'Cancelled'
  return 'Failed'
}

function describe(job) {
  const p = job.params || {}
  if (p.kind === 'model') return `Suitability model`
  return `${p.stages?.length || 0}-layer enrichment`
}

onMounted(() => { refresh() })
</script>

<style scoped>
.recent-jobs { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }
.job-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; }
.job-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
.job-name { flex: 1 1 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text, #222); }
.job-status { flex: 0 0 auto; font-size: 0.72rem; color: var(--muted, #777); }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; background: var(--muted, #999); }
.dot.running, .dot.queued { background: #2a78d6; }
.dot.succeeded { background: #3d8b5f; }
.dot.failed { background: #b3492f; }
.job-status.succeeded { color: #3d8b5f; }
.job-status.failed { color: #b3492f; }
</style>
