<template>
  <div class="dash-widget quota-widget">
    <div class="widget-head">
      <h3 class="widget-title">⚡ Quota usage</h3>
      <NuxtLink to="/jobs" class="widget-link">Jobs ›</NuxtLink>
    </div>

    <p v-if="loadingProfile" class="widget-note">Loading…</p>
    <p v-else-if="!profile" class="widget-note">Sign in to view quota.</p>

    <div v-else class="quota-body">
      <div class="quota-tier">
        <span class="tier-badge" :class="`tier-${tier}`">{{ TIER_LABELS[tier] || tier }}</span>
        <span v-if="profile.member_until && !neverExpires(profile.member_until)" class="tier-until">
          until {{ formatDate(profile.member_until) }}
        </span>
      </div>

      <div class="quota-rows">
        <div class="quota-row">
          <span class="quota-label">Monthly GEE jobs</span>
          <div class="quota-bar-wrap">
            <div class="quota-bar">
              <div class="quota-fill" :style="{ width: `${monthlyPct}%` }" :class="fillClass(monthlyPct)"></div>
            </div>
            <span class="quota-nums">{{ jobsThisMonth }} / {{ profile.ee_quota_monthly }}</span>
          </div>
        </div>

        <div class="quota-row">
          <span class="quota-label">Jobs today</span>
          <div class="quota-bar-wrap">
            <div class="quota-bar">
              <div class="quota-fill" :style="{ width: `${dailyPct}%` }" :class="fillClass(dailyPct)"></div>
            </div>
            <span class="quota-nums">{{ jobsToday }} / {{ profile.ee_jobs_per_day }}</span>
          </div>
        </div>
      </div>

      <div class="quota-limits">
        <div class="limit-item">
          <span class="limit-label">Max points</span>
          <span class="limit-val">{{ profile.ee_max_points.toLocaleString() }}</span>
        </div>
        <div class="limit-item">
          <span class="limit-label">Concurrent</span>
          <span class="limit-val">{{ profile.ee_max_concurrent }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useMembership } from '~/composables/useMembership'
import { useEeJobs } from '~/composables/useEeJobs'

const { profile, tier, loadProfile } = useMembership()
const { jobs, refresh: refreshJobs } = useEeJobs()
const loadingProfile = ref(true)

const TIER_LABELS = { free: 'Free', member: 'Member', admin: 'Admin' }

// neverExpires mirrors the server-side helper: a far-future date means no expiry.
function neverExpires(d) {
  if (!d) return false
  return new Date(d).getFullYear() >= 9000
}

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
function formatDate(d) {
  if (!d) return ''
  const dt = new Date(d)
  return isNaN(dt) ? '' : `${MONTHS[dt.getMonth()]} ${dt.getDate()}, ${dt.getFullYear()}`
}

const now = new Date()
const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1)
const startOfDay = new Date(now.getFullYear(), now.getMonth(), now.getDate())

const jobsThisMonth = computed(() =>
  (jobs.value || []).filter((j) => j.created_at && new Date(j.created_at) >= startOfMonth).length
)
const jobsToday = computed(() =>
  (jobs.value || []).filter((j) => j.created_at && new Date(j.created_at) >= startOfDay).length
)

const monthlyPct = computed(() =>
  profile.value ? Math.min(100, Math.round((jobsThisMonth.value / profile.value.ee_quota_monthly) * 100)) : 0
)
const dailyPct = computed(() =>
  profile.value ? Math.min(100, Math.round((jobsToday.value / profile.value.ee_jobs_per_day) * 100)) : 0
)

function fillClass(pct) {
  if (pct >= 90) return 'fill-danger'
  if (pct >= 60) return 'fill-warn'
  return 'fill-ok'
}

onMounted(async () => {
  await Promise.all([loadProfile(), refreshJobs()])
  loadingProfile.value = false
})
</script>

<style scoped>
.quota-widget { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }

.quota-body { display: flex; flex-direction: column; gap: 0.75rem; flex: 1; }

.quota-tier { display: flex; align-items: center; gap: 0.6rem; }
.tier-badge {
  font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.06em; padding: 2px 8px; border-radius: 999px;
}
.tier-free { background: var(--surface-2, #eee); color: var(--muted, #666); }
.tier-member { background: #e8f4fd; color: #1a6aad; }
.tier-admin { background: #f0e8ff; color: #6a1aad; }
.tier-until { font-size: 0.72rem; color: var(--muted, #aaa); }

.quota-rows { display: flex; flex-direction: column; gap: 0.55rem; }
.quota-row { display: flex; flex-direction: column; gap: 0.25rem; }
.quota-label { font-size: 0.72rem; color: var(--muted, #666); text-transform: uppercase; letter-spacing: 0.04em; }
.quota-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
.quota-bar { flex: 1; height: 8px; background: var(--surface-2, #eee); border-radius: 4px; overflow: hidden; }
.quota-fill { height: 100%; border-radius: 4px; transition: width 0.4s; }
.fill-ok { background: #27ae60; }
.fill-warn { background: #e67e22; }
.fill-danger { background: #e74c3c; }
.quota-nums { font-size: 0.72rem; color: var(--muted, #666); font-variant-numeric: tabular-nums; white-space: nowrap; }

.quota-limits { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.limit-item {
  flex: 1; min-width: 80px; display: flex; flex-direction: column; gap: 0.1rem;
  padding: 0.5rem; background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px;
}
.limit-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted, #888); }
.limit-val { font-size: 1rem; font-weight: 700; color: var(--text, #222); font-variant-numeric: tabular-nums; }
</style>
