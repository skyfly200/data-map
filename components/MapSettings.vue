<template>
  <div ref="root" class="settings">
    <button class="set-btn" :class="{ on: open }"
            :title="tip('Map display and data options', ',')"
            :aria-expanded="String(open)" @click="open = !open">
      <span class="gear" aria-hidden="true">⚙</span> Settings
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
        <span class="set-label">Actions</span>
        <button class="set-action" :class="{ busy: locating }" :disabled="locating"
                :title="locateError || tip('Centre the map on where you are', 'l')"
                @click="$emit('locate')">
          <span class="dot-icon"></span>{{ locating ? 'Locating…' : 'My location' }}
          <HelpLink option="map-locate" />
        </button>
        <p v-if="locateError" class="set-err">{{ locateError }}</p>

        <button class="set-action" :disabled="saving"
                :title="saveError || tip('Save the map, basemap and all, as a PNG', 'e')"
                @click="$emit('save')">
          ⤓ {{ saving ? 'Saving…' : 'Save image' }}
          <HelpLink option="map-save-image" keys="e" />
        </button>
        <p v-if="saveError" class="set-err">{{ saveError }}</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'

// The options that were eating the control bar. None of them are things you
// reach for every minute — the encodings and the overlay are — so they belong
// behind one button rather than spread across the top of the map.
defineProps({
  // Whether the observation points are drawn. v-model, because the map owns it.
  modelValue: { type: Boolean, default: true },
  locating: { type: Boolean, default: false },
  locateError: { type: String, default: '' },
  saving: { type: Boolean, default: false },
  saveError: { type: String, default: '' },
})
defineEmits(['update:modelValue', 'locate', 'save'])

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
  padding: 7px 10px; font: 600 13px system-ui, sans-serif; color: #333; cursor: pointer;
  display: inline-flex; gap: 6px; align-items: center; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
}
.set-btn:hover, .set-btn.on { background: #fff; }
.gear { font-size: 14px; line-height: 1; }

.set-panel {
  position: absolute; top: calc(100% + 6px); left: 0; z-index: 900; width: 280px;
  max-height: 70vh; overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 4px 16px var(--shadow); padding: 12px; font-size: 0.82rem; color: var(--text);
}
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

.set-action {
  display: flex; align-items: center; gap: 7px; width: 100%; margin-bottom: 6px;
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 6px; padding: 6px 10px; font-size: 0.82rem; font-weight: 600; cursor: pointer;
  text-align: left;
}
.set-action:hover:not(:disabled) { background: var(--surface-3); }
.set-action:disabled { opacity: 0.7; cursor: progress; }
.set-action .dot-icon {
  width: 11px; height: 11px; border-radius: 50%; background: #2a78d6; border: 2px solid #fff;
  box-shadow: 0 0 0 1px #2a78d6; flex: 0 0 auto;
}
.set-err { margin: 0 0 8px; color: var(--danger); font-size: 0.74rem; line-height: 1.4; }

@media (max-width: 640px) {
  .set-panel { width: min(280px, calc(100vw - 40px)); }
}
</style>
