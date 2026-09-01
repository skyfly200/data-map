<template>
  <!-- Silent when there is no account to sync with: local-only is a complete,
       valid state, not something to nag about. -->
  <div v-if="cloud.enabled.value || cloud.status.value === 'error'" class="sync">
    <button class="sy-btn" :class="cloud.status.value" :title="tooltip" :disabled="busy"
            @click="resync">
      <span class="dot" :class="cloud.status.value"></span>{{ label }}
    </button>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const cloud = useCloudSync()
const saved = useSavedCharts()
const appearance = useAppearance()
const layout = useChartLayout()
const overlays = useMapOverlays()
const busy = ref(false)

const label = computed(() => ({
  syncing: 'Syncing…',
  synced: 'Synced',
  error: 'Sync failed',
}[cloud.status.value] || 'Sync'))

const tooltip = computed(() => {
  if (cloud.status.value === 'error') return `${cloud.error.value} — click to retry`
  if (cloud.lastSynced.value) {
    return `Settings and charts saved to your account · last synced ${
      new Date(cloud.lastSynced.value).toLocaleTimeString()} · click to sync now`
  }
  return 'Settings and charts follow your account — click to sync now'
})

/**
 * Pull from the account and re-read every composable that caches its state.
 * Without the re-read, the new values would sit in localStorage unseen until
 * the next reload.
 */
async function resync() {
  if (busy.value) return
  busy.value = true
  try {
    await cloud.sync()
    appearance.loadFromStorage()
    layout.loadFromStorage()
    overlays.loadFromStorage()
    saved.loadFromStorage()
  } finally {
    busy.value = false
  }
}

// Sync when a session appears — signing in on a new device should bring
// everything with it without the viewer asking.
const { isAuthed } = useAuth()
watch(isAuthed, (authed, was) => {
  if (authed && !was) resync()
}, { immediate: true })
</script>

<style scoped>
.sync { display: inline-flex; }
.sy-btn {
  display: inline-flex; align-items: center; gap: 5px;
  border: 1px solid var(--border); background: transparent; color: var(--muted);
  border-radius: 6px; padding: 4px 8px; font-size: 0.74rem; font-weight: 600; cursor: pointer;
}
.sy-btn:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.sy-btn:disabled { cursor: default; }
.sy-btn.error { color: var(--danger); border-color: var(--danger); }

.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
.dot.synced { background: var(--accent); }
.dot.error { background: var(--danger); }
.dot.syncing { background: var(--accent); animation: pulse 1s ease-in-out infinite; }
@keyframes pulse { 50% { opacity: 0.3; } }
</style>
