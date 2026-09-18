<template>
  <div class="hmc">
    <div class="pop-field">
      <label :for="`${uid}-mode`">Show <HelpLink :option="docId" /></label>
      <select :id="`${uid}-mode`" v-model="mode" :title="tipText">
        <option value="">None</option>
        <optgroup v-for="g in groupedModes" :key="g.label" :label="g.label">
          <option v-for="o in g.modes" :key="o.key" :value="o.key">{{ o.label }}</option>
        </optgroup>
      </select>
    </div>

    <div v-if="mode" class="pop-field">
      <label :for="`${uid}-cell`">Cell size <HelpLink option="map-cell-size" /></label>
      <select :id="`${uid}-cell`" v-model.number="cellSize"
              title="Ground size of each grid cell. Smaller is more precise and noisier.">
        <option v-for="c in CELL_SIZES" :key="c.value" :value="c.value">{{ c.label }}</option>
      </select>
    </div>

    <!-- Only the seasonal modes use a date window, so it appears with them
         rather than being a permanent control that does nothing. -->
    <template v-if="mode === 'season' || mode === 'hotspots'">
      <div class="pop-field">
        <label :for="`${uid}-day`">
          Date <strong>{{ seasonLabel }}</strong> <HelpLink option="map-season-day" keys="[" />
          <button class="today-btn" :disabled="seasonDay === todayDay"
                  title="Centre the window on today" @click="seasonDay = todayDay">Today</button>
        </label>
        <input :id="`${uid}-day`" v-model.number="seasonDay" type="range" min="1" max="365" step="1"
               :title="tip(`Centre of the date window: currently ${seasonLabel}`, '[')" />
      </div>
      <div class="pop-field">
        <label :for="`${uid}-window`">
          Window <strong>±{{ seasonWindow }} days</strong> <HelpLink option="map-season-window" />
        </label>
        <input :id="`${uid}-window`" v-model.number="seasonWindow" type="range"
               min="3" max="60" step="1"
               title="How wide a window counts as 'in season'. Wider is smoother and less specific." />
      </div>
      <p class="slider-note">{{ windowSpan }}</p>
    </template>
  </div>
</template>

<script setup>
// The heatmap's controls, on their own so they can live in two places.
//
// On a wide screen they are a dropdown on the map's control bar. On a phone
// that bar was eight buttons across a 390px screen — at the exact edge of
// fitting, and past it as soon as anything changed — so there they move inside
// the layer window instead, with the basemap, under a heading.
//
// No props for any of it: every value comes from useMapHeatmaps, which is
// shared state. Threading a dozen v-models through two hosts would be two
// places to forget one.

import { computed, useId } from 'vue'

import { useMapHeatmaps } from '~/composables/useMapHeatmaps'
import { docAnchor } from '~/composables/optionDocs'

const heatmaps = useMapHeatmaps()
const {
  mode, cellSize, seasonDay, seasonWindow, activeMode, groupedModes, CELL_SIZES, todayOfYear,
} = heatmaps

// Ids have to differ between the two hosts, or a label points at the other
// copy's control and tapping it does nothing.
const uid = useId()

const shortcuts = useShortcuts()
const tip = (text, keys) => shortcuts.withKey(text, keys)

const todayDay = computed(() => todayOfYear())
const tipText = computed(() => activeMode.value?.note || 'A grid summary drawn under the points.')
const docId = computed(() => (mode.value ? `map-heatmap-${mode.value}` : 'map-heatmap'))

const DAY_MS = 86400000
const dayLabel = (day) => {
  const d = new Date(Date.UTC(2001, 0, 1) + (Math.max(1, day) - 1) * DAY_MS)
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', timeZone: 'UTC' })
}
const seasonLabel = computed(() => dayLabel(seasonDay.value))
const windowSpan = computed(() =>
  `${dayLabel(seasonDay.value - seasonWindow.value)} – ${dayLabel(seasonDay.value + seasonWindow.value)}`)
</script>

<style scoped>
.hmc { display: grid; gap: 8px; }
.pop-field { display: grid; gap: 4px; }
.pop-field label {
  display: flex; align-items: center; gap: 5px;
  font-size: 0.74rem; color: var(--muted); font-weight: 600;
}
.pop-field select {
  width: 100%; box-sizing: border-box;
  background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 6px; font: inherit; font-size: 0.8rem;
}
.pop-field input[type="range"] { width: 100%; margin: 0; accent-color: var(--accent); }
.today-btn {
  margin-left: auto; background: none; border: 0; padding: 0; cursor: pointer;
  font: inherit; font-size: 0.7rem; color: var(--accent); text-decoration: underline;
}
.today-btn:disabled { opacity: 0.4; cursor: default; text-decoration: none; }
.slider-note { margin: 0; font-size: 0.7rem; color: var(--muted); }
</style>
