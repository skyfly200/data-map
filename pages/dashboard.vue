<template>
  <div v-if="!isAuthed" class="dash-guard">
    <h2>Dashboard</h2>
    <p>Sign in to see and customise your dashboard.</p>
    <NuxtLink to="/login" class="btn primary">Log in</NuxtLink>
  </div>

  <div v-else class="dashboard">
    <header class="dash-header">
      <div>
        <h1>Dashboard</h1>
        <p class="sub">Your observations, jobs, taxa and saved views at a glance.</p>
      </div>
      <div class="dash-actions">
        <button v-if="!editing" class="btn" @click="editing = true">✏️ Customise</button>
        <template v-else>
          <button class="btn" @click="addDefaults">Reset to default</button>
          <button class="btn primary" @click="editing = false">Done</button>
        </template>
      </div>
    </header>

    <!-- The palette of widgets to add, only while editing. -->
    <div v-if="editing" class="palette">
      <button v-for="w in WIDGET_TYPES" :key="w.type" class="palette-item"
              :disabled="hasType(w.type)" @click="add({ type: w.type, label: w.label })">
        <span class="pi-icon">{{ w.icon }}</span>{{ w.label }}
        <span v-if="hasType(w.type)" class="pi-on">on</span>
      </button>
    </div>

    <p v-if="!orderedWidgets.length" class="dash-empty">
      Nothing on the dashboard yet. {{ editing ? 'Add a widget above.' : 'Press Customise to add widgets.' }}
    </p>

    <div v-else class="dash-grid">
      <section v-for="w in orderedWidgets" :key="w.id"
               class="dash-cell"
               :class="cellClass(w, editing)"
               :draggable="editing"
               @dragstart="onDragStart(w.id)" @dragover.prevent @drop="onDrop(w.id)">
        <div v-if="editing" class="cell-bar">
          <span class="drag" title="Drag to reorder">⋮⋮</span>
          <div class="size-btns" title="Resize widget">
            <button :class="{ on: !w.settings.colspan || w.settings.colspan === 1 }"
                    title="1 column" @click="updateSettings(w.id, { colspan: 1 })">▪</button>
            <button :class="{ on: w.settings.colspan === 2 }"
                    title="2 columns" @click="updateSettings(w.id, { colspan: 2 })">▪▪</button>
            <button :class="{ on: w.settings.colspan === 'full' }"
                    title="Full width" @click="updateSettings(w.id, { colspan: 'full' })">▬</button>
            <button :class="{ on: w.settings.tall === 1 }"
                    title="Tall (1.5×)" @click="updateSettings(w.id, { tall: w.settings.tall === 1 ? false : 1 })">↕</button>
            <button :class="{ on: w.settings.tall === 2 }"
                    title="Extra tall (2×)" @click="updateSettings(w.id, { tall: w.settings.tall === 2 ? false : 2 })">⤢</button>
          </div>
          <button class="remove" title="Remove" @click="remove(w.id)">×</button>
        </div>
        <!-- Widgets below the fold build as they scroll in, so opening the
             dashboard does not fetch and compute every one at once. -->
        <LazyVisible min-height="140px">
          <component :is="componentFor(w.type)" :widget="w" :is-editing="editing" />
        </LazyVisible>
      </section>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { useAuth } from '~/composables/useAuth'
import { useDashboardState } from '~/composables/useDashboardState'
import DashboardOverview from '~/components/DashboardOverview.vue'
import DashboardRecentJobs from '~/components/DashboardRecentJobs.vue'
import DashboardSpeciesList from '~/components/DashboardSpeciesList.vue'
import DashboardEnvStats from '~/components/DashboardEnvStats.vue'
import DashboardSavedCharts from '~/components/DashboardSavedCharts.vue'
import DashboardQuickFilters from '~/components/DashboardQuickFilters.vue'
import DashboardNowFruiting from '~/components/DashboardNowFruiting.vue'
import DashboardPhenology from '~/components/DashboardPhenology.vue'
import DashboardModelLeaderboard from '~/components/DashboardModelLeaderboard.vue'
import DashboardMapPreview from '~/components/DashboardMapPreview.vue'
import DashboardDataQuality from '~/components/DashboardDataQuality.vue'
import DashboardFavorites from '~/components/DashboardFavorites.vue'
import DashboardWeather from '~/components/DashboardWeather.vue'
import DashboardChart from '~/components/DashboardChart.vue'
import DashboardDatasets from '~/components/DashboardDatasets.vue'
import DashboardQuota from '~/components/DashboardQuota.vue'
import DashboardWeatherForecast from '~/components/DashboardWeatherForecast.vue'
import DashboardRecentRainTemp from '~/components/DashboardRecentRainTemp.vue'

const { isAuthed } = useAuth()
const { orderedWidgets, load, add, remove, reorder, reset, updateSettings } = useDashboardState()

// The widget catalogue: what a member can put on the dashboard, and what draws it.
const WIDGET_TYPES = [
  { type: 'overview', label: 'Overview', icon: '📌', component: DashboardOverview },
  { type: 'recent-jobs', label: 'Recent jobs', icon: '⚙️', component: DashboardRecentJobs },
  { type: 'species-list', label: 'Top species', icon: '🍄', component: DashboardSpeciesList },
  { type: 'env-stats', label: 'Data Stats', icon: '📊', component: DashboardEnvStats },
  { type: 'saved-charts', label: 'Saved charts', icon: '📊', component: DashboardSavedCharts },
  { type: 'quick-filters', label: 'Quick filters', icon: '🔍', component: DashboardQuickFilters },
  { type: 'now-fruiting', label: 'Now fruiting', icon: '🍄', component: DashboardNowFruiting },
  { type: 'phenology', label: 'Phenology calendar', icon: '📅', component: DashboardPhenology },
  { type: 'model-leaderboard', label: 'Model leaderboard', icon: '🏆', component: DashboardModelLeaderboard },
  { type: 'map-preview', label: 'Map preview', icon: '🗺️', component: DashboardMapPreview },
  { type: 'data-quality', label: 'Data quality', icon: '🔬', component: DashboardDataQuality },
  { type: 'favorites', label: 'Favorites', icon: '⭐', component: DashboardFavorites },
  { type: 'weather', label: 'Weather correlation', icon: '🌧️', component: DashboardWeather },
  { type: 'chart', label: 'Custom chart', icon: '📊', component: DashboardChart },
  { type: 'datasets', label: 'Datasets', icon: '🗄️', component: DashboardDatasets },
  { type: 'quota', label: 'Quota usage', icon: '⚡', component: DashboardQuota },
  { type: 'weather-forecast', label: 'Weather & radar', icon: '🌤️', component: DashboardWeatherForecast },
  { type: 'recent-rain-temp', label: 'Recent rain & temps', icon: '🌡️', component: DashboardRecentRainTemp },
]
const COMPONENTS = Object.fromEntries(WIDGET_TYPES.map((w) => [w.type, w.component]))
function componentFor(type) { return COMPONENTS[type] || DashboardOverview }

// The layout a new member lands on. Sizes chosen so the default grid reads well
// at common viewport widths without wasted space.
const DEFAULTS = {
  widgets: [
    { id: 'def-overview',     type: 'overview',     settings: {},                   order: 0 },
    { id: 'def-now-fruiting', type: 'now-fruiting',  settings: {},                   order: 1 },
    { id: 'def-map-preview',  type: 'map-preview',   settings: { colspan: 2, tall: true }, order: 2 },
    { id: 'def-phenology',    type: 'phenology',     settings: { colspan: 2 },       order: 3 },
    { id: 'def-recent-jobs',  type: 'recent-jobs',   settings: {},                   order: 4 },
    { id: 'def-species-list', type: 'species-list',  settings: {},                   order: 5 },
    { id: 'def-env-stats',    type: 'env-stats',     settings: { colspan: 2 },       order: 6 },
  ],
  order: ['def-overview', 'def-now-fruiting', 'def-map-preview', 'def-phenology',
          'def-recent-jobs', 'def-species-list', 'def-env-stats'],
}

function cellClass(w, isEditing) {
  const span = w.settings?.colspan
  const tall = w.settings?.tall
  return {
    editing:    isEditing,
    'span-2':   span === 2,
    'span-full': span === 'full',
    tall:       tall === true || tall === 1,
    'tall-2':   tall === 2,
  }
}

const editing = ref(false)

function hasType(type) { return orderedWidgets.value.some((w) => w.type === type) }
function addDefaults() { reset(structuredClone(DEFAULTS)) }

// Drag to reorder: remember what is picked up, drop it onto a target's slot.
const dragging = ref('')
function onDragStart(id) { dragging.value = id }
function onDrop(targetId) {
  if (dragging.value && dragging.value !== targetId) reorder(dragging.value, targetId)
  dragging.value = ''
}

onMounted(() => { load(DEFAULTS) })
</script>

<style scoped>
.dash-guard { max-width: 420px; margin: 4rem auto; text-align: center; display: grid; gap: 0.8rem; }
.dashboard { padding: 1rem 1.5rem 3rem; }
.dash-header { display: flex; align-items: flex-start; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
.dash-header h1 { margin: 0; font-size: 1.5rem; }
.sub { margin: 0.2rem 0 0; color: var(--muted, #777); font-size: 0.9rem; }
.dash-actions { display: flex; gap: 0.5rem; }
.btn {
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff); color: var(--text, #222);
  border-radius: 8px; padding: 0.5rem 0.9rem; font: inherit; font-size: 0.85rem; cursor: pointer;
}
.btn.primary { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }
.btn:hover { filter: brightness(0.98); }

.palette { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; padding: 0.8rem; background: var(--surface-2, #f6f6f6); border-radius: 10px; }
.palette-item {
  display: inline-flex; align-items: center; gap: 0.4rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); border-radius: 999px; padding: 0.35rem 0.8rem; font: inherit;
  font-size: 0.82rem; cursor: pointer;
}
.palette-item:disabled { opacity: 0.5; cursor: default; }
.pi-icon { font-size: 1rem; }
.pi-on { font-size: 0.62rem; text-transform: uppercase; color: var(--muted, #888); }

.dash-empty { text-align: center; color: var(--muted, #888); padding: 3rem 1rem; }

/* Grid layout.
   - 3 columns on wide screens, 2 on medium, 1 on mobile.
   - .span-2 spans 2 cols; .span-full spans all.
   - .tall makes the cell a fixed taller height with inner content stretching to fill.
   - align-items: stretch so tall cells fill their row height visually. */
.dash-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
  align-items: start;
}
.dash-cell {
  position: relative; background: var(--surface, #fff); border: 1px solid var(--border, #e5e5e5);
  border-radius: 12px; padding: 1rem; min-height: 160px;
  display: flex; flex-direction: column;
  transition: box-shadow 0.15s;
}
/* LazyVisible wrapper and the .dash-widget inside both stretch to fill the cell */
.dash-cell :deep(.lazy-visible) { flex: 1; min-height: 0; }
.dash-cell :deep(.dash-widget) { flex: 1; min-height: 0; }

.dash-cell.editing { border-style: dashed; cursor: grab; }
.dash-cell.editing:hover { box-shadow: 0 0 0 2px var(--accent, #2a78d6); }
.dash-cell.span-2 { grid-column: span 2; }
.dash-cell.span-full { grid-column: 1 / -1; }
.dash-cell.tall { height: 520px; min-height: unset; }
.dash-cell.tall-2 { height: 760px; min-height: unset; }

.cell-bar { display: flex; justify-content: space-between; align-items: center; margin: -0.3rem -0.3rem 0.4rem; gap: 0.3rem; }
.drag { color: var(--muted, #aaa); cursor: grab; user-select: none; }

.size-btns { display: flex; gap: 2px; margin: 0 auto 0 0.5rem; }
.size-btns button {
  border: 1px solid var(--border, #ddd); background: var(--surface-2, #f5f5f5); color: var(--muted, #888);
  border-radius: 4px; padding: 2px 6px; font-size: 0.7rem; cursor: pointer; line-height: 1.4;
}
.size-btns button:hover { background: var(--surface-3, #eee); color: var(--text, #222); }
.size-btns button.on { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }

.remove {
  border: 0; background: none; color: var(--muted, #999); font-size: 1.2rem; line-height: 1; cursor: pointer;
}
.remove:hover { color: #b3492f; }

@media (max-width: 600px) {
  .dash-grid { grid-template-columns: 1fr; }
  .dash-cell.span-2, .dash-cell.span-full { grid-column: 1; }
}
@media (min-width: 601px) and (max-width: 960px) {
  .dash-grid { grid-template-columns: repeat(2, 1fr); }
  .dash-cell.span-full { grid-column: 1 / -1; }
  .dash-cell.span-2 { grid-column: span 2; }
}
@media (min-width: 961px) {
  .dash-cell.span-full { grid-column: 1 / -1; }
}
</style>
