<template>
  <div class="dashboard-recent-jobs">
    <h3 class="widget-title">⚙️ Recent Jobs</h3>
    <div v-if="jobs.length === 0" class="no-jobs">
      <p>No recent jobs.</p>
      <NuxtLink to="/jobs" class="view-jobs-link">View all jobs</NuxtLink>
    </div>
    <div v-else class="jobs-list">
      <div v-for="job in jobs" :key="job.id" class="job-item">
        <div class="job-header">
          <span class="job-name">{{ job.name || 'Unnamed Job' }}</span>
          <span class="job-status" :class="job.status">{{ job.status }}</span>
        </div>
        <div class="job-meta">
          <span v-if="job.created_at" class="job-date">{{ formatDate(job.created_at) }}</span>
          <span v-if="job.progress" class="job-progress">{{ job.progress }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const jobs = ref([])

function formatDate(dateStr) {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
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
})
</script>

<style scoped>
.dashboard-recent-jobs {
  height: 100%;
}

.widget-title {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text, #222);
}

.no-jobs {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted, #999);
}

.view-jobs-link {
  display: inline-block;
  margin-top: 0.75rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
  font-weight: 500;
}

.view-jobs-link:hover {
  text-decoration: underline;
}

.jobs-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.job-item {
  padding: 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 6px;
}

.job-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.job-name {
  font-weight: 600;
  color: var(--text, #222);
  font-size: 0.9rem;
}

.job-status {
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  text-transform: uppercase;
  font-weight: 600;
}

.job-status.completed {
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
