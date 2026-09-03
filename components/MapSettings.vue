<template>
  <div ref="root" class="settings">
    <button class="set-btn" :class="{ on: open }"
            :title="tip('Map display and data options', ',')" aria-label="Map settings"
            :aria-expanded="String(open)" @click="open = !open">
      <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
        <path fill="currentColor" d="M19.4 13a7.5 7.5 0 0 0 0-2l2-1.6-2-3.4-2.4 1a7.6 7.6 0 0 0-1.7-1l-.4-2.6h-4l-.4 2.6c-.6.2-1.2.6-1.7 1l-2.4-1-2 3.4L6.6 11a7.5 7.5 0 0 0 0 2l-2 1.6 2 3.4 2.4-1c.5.4 1.1.8 1.7 1l.4 2.6h4l.4-2.6c.6-.2 1.2-.6 1.7-1l2.4 1 2-3.4-2-1.6zM12 15.5A3.5 3.5 0 1 1 12 8.5a3.5 3.5 0 0 1 0 7z" />
      </svg>
    </button>

    <div v-if="open" class="set-panel">
      <div class="set-head">
        <span>Map settings</span>
        <button class="set-close" aria-label="Close" @click="open = false">×</button>
      </div>

      <div class="set-group">
        <span class="set-label">Display</span>
        <label class="set-row">
          <input type="checkbox" :checked="modelValue"
                 @change="$emit('update:modelValue', $event.target.checked)" />
          <span class="set-text">Show observation points</span>
          <HelpLink option="map-show-points" keys="p" />
        </label>
        <p class="set-note">Hide them to read an overlay on its own.</p>
      </div>

      <div class="set-group">
        <span class="set-label">Data</span>
        <label class="set-row">
          <input v-model="showFiltered" type="checkbox" />
          <span class="set-text">Include excluded water / non-terrestrial rows</span>
          <HelpLink option="data-show-filtered" />
        </label>
        <p class="set-note">
          Observations the pipeline flagged as landing on water, ice or built-up
          ground.
          <template v-if="!excludedCount">
            None are present in this dataset — the pipeline drops them before
            export, so this reveals nothing here.
          </template>
          <template v-else>{{ excludedCount.toLocaleString() }} in this dataset.</template>
        </p>
      </div>

      <div class="set-group">
        <span class="set-label">Offline <HelpLink option="map-offline" /></span>
        <ClientOnly>
          <OfflineControls :bounds="bounds" :templates="templates" :dataset-label="datasetLabel" />
        </ClientOnly>
      </div>

    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'

// The options that were eating the control bar: things you SET, not things you
// do. The two actions that briefly lived here — my location, save image — moved
// back out as icons, because a button you press to make something happen does
// not belong behind a menu of preferences.
defineProps({
  // Whether the observation points are drawn. v-model, because the map owns it.
  modelValue: { type: Boolean, default: true },
  // What is on screen, so "save this view" can mean the view. The map owns the
  // viewport and the active tile templates; this panel only passes them on.
  bounds: { type: Object, default: null },
  templates: { type: Array, default: () => [] },
  datasetLabel: { type: String, default: '' },
})
defineEmits(['update:modelValue'])

const { data, showFiltered } = useObservations()

// Whether the excluded rows exist at all. The pipeline drops them before export,
// so on the published dataset the toggle has nothing to reveal — better to say
// so here than to let someone tick it and wonder why nothing happened.
const excludedCount = computed(() =>
  (data.value?.features || []).reduce((n, f) => n + (f.properties?.water_mask ? 1 : 0), 0))

const shortcuts = useShortcuts()
const tip = (text, keys) => shortcuts.withKey(text, keys)

const open = ref(false)
shortcuts.register([
  { scope: 'Map', keys: ',', label: 'Map settings', run: () => { open.value = !open.value } },
])

// A panel this size covers the map, so it closes on an outside click as well as
// on the button — unlike the smaller popovers beside it.
const root = ref(null)
function onDocClick(e) {
  if (open.value && root.value && !root.value.contains(e.target)) open.value = false
}
onMounted(() => document.addEventListener('click', onDocClick))
onBeforeUnmount(() => document.removeEventListener('click', onDocClick))

defineExpose({ close: () => { open.value = false } })
</script>

<style scoped>
.settings { position: relative; }

/* Matches the other on-map controls rather than the in-page panels. */
.set-btn {
  background: rgba(255, 255, 255, 0.95); border: 1px solid #ddd; border-radius: 8px;
  width: 34px; height: 34px; padding: 0; color: #333; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.set-btn:hover, .set-btn.on { background: #fff; }

.set-panel {
  position: absolute; top: calc(100% + 6px); left: 0; z-index: 900; width: 280px;
  max-height: 70vh; overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 12px; font-size: 0.82rem; color: var(--text);
}
/* The offline panel is the tallest thing in here, and the one worth widening
   for: its rows are a label, an explanation and a button on one line. */
.set-panel { width: 320px; }
.set-head {
  display: flex; align-items: center; justify-content: space-between;
  font-weight: 700; margin-bottom: 10px;
}
.set-close {
  border: 0; background: transparent; color: var(--muted);
  font-size: 1.3rem; line-height: 1; cursor: pointer; padding: 0 2px;
}
.set-close:hover { color: var(--text); }

.set-group { margin-bottom: 14px; }
.set-group:last-child { margin-bottom: 0; }
.set-label {
  display: block; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--muted); font-weight: 700; margin-bottom: 6px;
}
.set-row {
  display: flex; align-items: flex-start; gap: 7px; cursor: pointer; line-height: 1.4;
}
.set-row input { margin: 2px 0 0; accent-color: var(--accent); flex: 0 0 auto; }
.set-text { flex: 1 1 auto; min-width: 0; }
.set-note { margin: 5px 0 0 22px; color: var(--muted); font-size: 0.74rem; line-height: 1.45; }

.set-err { margin: 0 0 8px; color: var(--danger); font-size: 0.74rem; line-height: 1.4; }

@media (max-width: 640px) {
  .set-panel { width: min(280px, calc(100vw - 40px)); }
}
</style>
