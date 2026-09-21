<template>
  <div class="dashboard-recent-jobs">
    <h3 class="widget-title">⚙️ Recent Jobs</h3>
    <div v-if="jobs.length === 0" class="no-jobs">
      <p>No recent jobs.</p>
      <NuxtLink to="/jobs" class="view-jobs-link">View all jobs</NuxtLink>
    <div class="widget-head">
      <h3 class="widget-title">⚙️ Recent Jobs</h3>
      <NuxtLink to="/jobs" class="view-jobs-link">All jobs ›</NuxtLink>
    </div>

    <div v-if="loading && !jobs.length" class="jobs-loading">
      <p>Loading jobs…</p>
    </div>

    <div v-else-if="jobs.length === 0" class="no-jobs">
      <p>No enrichment jobs yet.</p>
      <NuxtLink to="/jobs" class="submit-job-link">Run an enrichment job</NuxtLink>
    </div>

    <div v-else class="jobs-list">
      <div v-for="job in jobs" :key="job.id" class="job-item">
      <div v-for="job in jobList" :key="job.id" class="job-item">
        <div class="job-header">
          <span class="job-name">{{ job.name || 'Unnamed Job' }}</span>
          <span class="job-name" :title="job.title || job.kind || 'Pipeline job'">
            {{ job.title || job.kind || 'Pipeline job' }}
          </span>
          <span class="job-status" :class="job.status">{{ job.status }}</span>
        </div>
        <div class="job-meta">
          <span v-if="job.created_at" class="job-date">{{ formatDate(job.created_at) }}</span>
          <span v-if="job.progress" class="job-progress">{{ job.progress }}%</span>
          <span class="job-date">{{ formatDate(job.created_at) }}</span>
          <span v-if="job.status === 'running' && job.progress" class="job-progress">
            {{ Math.round(job.progress * 100) }}%
          </span>
          <span v-else-if="job.result_meta?.features" class="job-features">
            {{ job.result_meta.features.toLocaleString() }} records
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { computed, onMounted } from 'vue'
import { useEeJobs } from '~/composables/useEeJobs'

const jobs = ref([])
const { jobs, loading, refresh } = useEeJobs()

const jobList = computed(() => (jobs.value || []).slice(0, 5))

function formatDate(dateStr) {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}

onMounted(() => {
  // Load recent jobs from localStorage or API
  if (import.meta.client) {
    const saved = localStorage.getItem('recent-jobs')
    if (saved) {
      try {
        jobs.value = JSON.parse(saved).slice(0, 5) || []
      } catch {
        jobs.value = []
      }
    }
  }
  refresh()
})
</script>

<style scoped>
.dashboard-recent-jobs {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.widget-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.widget-title {
  margin: 0 0 1rem 0;
  margin: 0;
  font-size: 1rem;
  color: var(--text, #222);
}

.view-jobs-link,
.submit-job-link {
  font-size: 0.8rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
}

.view-jobs-link:hover,
.submit-job-link:hover {
  text-decoration: underline;
}

.jobs-loading,
.no-jobs {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted, #999);
}

.view-jobs-link {
.submit-job-link {
  display: inline-block;
  margin-top: 0.75rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
  margin-top: 0.5rem;
  font-weight: 500;
}

.view-jobs-link:hover {
  text-decoration: underline;
}

.jobs-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
  gap: 0.5rem;
}

.job-item {
  padding: 0.75rem;
  padding: 0.65rem 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 6px;
  border: 1px solid var(--border-soft, #eee);
}

.job-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
  gap: 0.5rem;
  margin-bottom: 0.35rem;
}

.job-name {
  font-weight: 600;
  color: var(--text, #222);
  font-size: 0.9rem;
  font-size: 0.85rem;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.job-status {
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
  font-size: 0.7rem;
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
  text-transform: uppercase;
  font-weight: 600;
  letter-spacing: 0.04em;
  flex-shrink: 0;
}

.job-status.completed {
.job-status.succeeded {
  background: #d4edda;
  color: #155724;
}

.job-status.running {
  background: #cce5ff;
  color: #004085;
}

.job-status.failed {
  background: #f8d7da;
  color: #721c24;
}

.job-status.queued,
.job-status.pending {
  background: #fff3cd;
  color: #856404;
}

.job-meta {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: var(--muted, #666);
}
</style>
