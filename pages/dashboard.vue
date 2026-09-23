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
      <section v-for="w in orderedWidgets" :key="w.id" class="dash-cell" :class="{ editing }"
               :draggable="editing"
               @dragstart="onDragStart(w.id)" @dragover.prevent @drop="onDrop(w.id)">
        <div v-if="editing" class="cell-bar">
          <span class="drag" title="Drag to reorder">⋮⋮</span>
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

const { isAuthed } = useAuth()
const { orderedWidgets, load, add, remove, reorder, reset } = useDashboardState()

// The widget catalogue: what a member can put on the dashboard, and what draws it.
const WIDGET_TYPES = [
  { type: 'overview', label: 'Overview', icon: '📌', component: DashboardOverview },
  { type: 'recent-jobs', label: 'Recent jobs', icon: '⚙️', component: DashboardRecentJobs },
  { type: 'species-list', label: 'Top species', icon: '🍄', component: DashboardSpeciesList },
  { type: 'env-stats', label: 'Environmental stats', icon: '🌡️', component: DashboardEnvStats },
  { type: 'saved-charts', label: 'Saved charts', icon: '📊', component: DashboardSavedCharts },
  { type: 'quick-filters', label: 'Quick filters', icon: '🔍', component: DashboardQuickFilters },
  { type: 'now-fruiting', label: 'Now fruiting', icon: '🍄', component: DashboardNowFruiting },
  { type: 'phenology', label: 'Phenology calendar', icon: '📅', component: DashboardPhenology },
]
const COMPONENTS = Object.fromEntries(WIDGET_TYPES.map((w) => [w.type, w.component]))
function componentFor(type) { return COMPONENTS[type] || DashboardOverview }

// The layout a new member lands on: an overview and the three most-answered
// panels, so the page reads as something rather than an empty grid.
const DEFAULTS = {
  widgets: ['overview', 'now-fruiting', 'phenology', 'recent-jobs', 'species-list', 'env-stats'].map((type, i) => ({
    id: `def-${type}`, type, settings: {}, order: i,
  })),
  order: ['def-overview', 'def-now-fruiting', 'def-phenology', 'def-recent-jobs', 'def-species-list', 'def-env-stats'],
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
.dashboard { max-width: 1100px; margin: 0 auto; padding: 1rem 1rem 3rem; }
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

.dash-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 1rem; align-items: start; }
.dash-cell {
  position: relative; background: var(--surface, #fff); border: 1px solid var(--border, #e5e5e5);
  border-radius: 12px; padding: 1rem; min-height: 120px;
}
.dash-cell.editing { border-style: dashed; cursor: grab; }
.cell-bar { display: flex; justify-content: space-between; align-items: center; margin: -0.3rem -0.3rem 0.4rem; }
.drag { color: var(--muted, #aaa); cursor: grab; user-select: none; }
.remove {
  border: 0; background: none; color: var(--muted, #999); font-size: 1.2rem; line-height: 1; cursor: pointer;
}
.remove:hover { color: #b3492f; }

@media (max-width: 560px) { .dash-grid { grid-template-columns: 1fr; } }
</style>
