<template>
  <!-- Silent when there is no account to sync with: local-only is a complete,
       valid state, not something to nag about. -->
  <div v-if="cloud.enabled.value || cloud.status.value === 'error'" class="sync">
    <button class="sy-btn" :class="cloud.status.value" :title="tooltip" :disabled="busy"
            @click="onClick">
      <span class="dot" :class="cloud.status.value"></span>{{ label }}
    </button>

    <!-- A failure needs to be readable, not hidden in a tooltip: the usual cause
         is a setup step that has not been run, and nobody hovers a button to
         find that out. -->
    <div v-if="showDetail && cloud.status.value === 'error'" class="sy-detail">
      <button class="sy-close" aria-label="Dismiss" @click="showDetail = false">×</button>
      <p class="sy-msg">{{ cloud.error.value }}</p>
      <p v-if="cloud.errorHint.value" class="sy-hint">{{ cloud.errorHint.value }}</p>
      <p class="sy-note">
        Your settings and charts are safe in this browser — only syncing to the
        account is affected.
      </p>
      <details v-if="cloud.errorRaw.value" class="sy-raw">
        <summary>Technical detail</summary>
        <code>{{ cloud.errorRaw.value }}</code>
      </details>
      <button class="sy-retry" :disabled="busy" @click="resync">
        {{ busy ? 'Retrying…' : 'Retry' }}
      </button>
    </div>
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

const showDetail = ref(false)

// Clicking retries when things are fine, and opens the explanation when they
// are not — retrying blind against a missing table just fails again.
function onClick() {
  if (cloud.status.value === 'error') showDetail.value = !showDetail.value
  else resync()
}

const label = computed(() => ({
  syncing: 'Syncing…',
  synced: 'Synced',
  error: 'Sync failed',
}[cloud.status.value] || 'Sync'))

const tooltip = computed(() => {
  if (cloud.status.value === 'error') return `${cloud.error.value} — click for detail`
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
watch(isAuthed, async (authed, was) => {
  if (!authed || was) return
  await resync()
  if (cloud.status.value === 'error') showDetail.value = true
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

.sy-detail {
  position: absolute; top: calc(100% + 6px); right: 0; z-index: 900; width: 300px;
  background: var(--surface); border: 1px solid var(--danger); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 12px; text-align: left;
  font-size: 0.8rem; color: var(--text);
}
.sync { position: relative; }
.sy-close {
  position: absolute; top: 6px; right: 8px; border: 0; background: transparent;
  color: var(--muted); font-size: 1.1rem; line-height: 1; cursor: pointer; padding: 0;
}
.sy-msg { margin: 0 18px 6px 0; font-weight: 600; }
.sy-hint { margin: 0 0 6px; color: var(--muted); line-height: 1.4; }
.sy-note { margin: 0 0 8px; color: var(--muted); font-size: 0.74rem; line-height: 1.4; }
.sy-raw { margin-bottom: 8px; }
.sy-raw summary { cursor: pointer; color: var(--muted); font-size: 0.74rem; }
.sy-raw code {
  display: block; margin-top: 5px; font-size: 0.7rem; line-height: 1.4;
  background: var(--surface-2); border-radius: 4px; padding: 5px 6px; word-break: break-word;
}
.sy-retry {
  width: 100%; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 6px; padding: 5px; font-size: 0.78rem;
  font-weight: 600; cursor: pointer;
}
.sy-retry:hover:not(:disabled) { background: var(--surface-3); }
</style>
