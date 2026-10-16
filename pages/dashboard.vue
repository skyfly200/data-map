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

    <!-- Pages: tabs that each hold their own widgets. Shown once there is more
         than one, or while customising so a second can be added. Dropping a
         widget onto a tab moves it to that page. -->
    <nav v-if="pages.length > 1 || editing" class="page-tabs" role="tablist" aria-label="Dashboard pages">
      <button v-for="p in pages" :key="p.id"
              type="button" role="tab" class="page-tab"
              :aria-selected="p.id === activePageId"
              :class="{ active: p.id === activePageId, 'drop-into': tabDrop === p.id }"
              @click="setActivePage(p.id)"
              @dragover.prevent="onTabDragOver(p.id)"
              @dragleave="onTabDragLeave(p.id)"
              @drop="onTabDrop(p.id)">
        {{ p.name }}<span v-if="editing" class="tab-count">{{ p.order.length }}</span>
      </button>
      <button v-if="editing" type="button" class="page-tab add" @click="addPage()">＋ Page</button>
    </nav>

    <div v-if="editing && activePage" class="page-tools">
      <label class="page-name">
        <span>Page name</span>
        <input :value="activePage.name" maxlength="40"
               @change="renamePage(activePage.id, $event.target.value); $event.target.value = activePage.name"
               @keydown.enter="$event.target.blur()" />
      </label>
      <div class="page-tool-btns">
        <button class="btn sm" :disabled="pageIndex <= 0" title="Move page left" @click="movePage(activePage.id, -1)">◀</button>
        <button class="btn sm" :disabled="pageIndex >= pages.length - 1" title="Move page right" @click="movePage(activePage.id, 1)">▶</button>
        <button class="btn sm danger" :disabled="pages.length <= 1" @click="deletePage">Delete page</button>
      </div>
    </div>

    <!-- The palette of widgets to add, only while editing. -->
    <div v-if="editing" class="palette"
         @dragover.prevent="onPaletteDragOver" @drop="onPaletteDrop">
      <div
        v-for="w in WIDGET_TYPES" :key="w.type"
        class="palette-item"
        :class="{ disabled: hasType(w.type) }"
        :draggable="!hasType(w.type)"
        @dragstart="onPaletteDragStart(w)"
        @click="!hasType(w.type) && add({ type: w.type, label: w.label })"
      >
        <span class="pi-icon">{{ w.icon }}</span>{{ w.label }}
        <span v-if="hasType(w.type)" class="pi-on">on</span>
      </div>
    </div>

    <p v-if="!orderedWidgets.length" class="dash-empty"
       :class="{ 'drop-target': editing && dragFromPalette }"
       @dragover.prevent @drop="onGridEndDrop">
      Nothing on {{ pages.length > 1 ? 'this page' : 'the dashboard' }} yet. {{ editing ? 'Drag or click a widget above to add it.' : 'Press Customise to add widgets.' }}
    </p>

    <div v-else class="dash-grid">
      <section v-for="w in orderedWidgets" :key="w.id"
               class="dash-cell"
               :class="[cellClass(w, editing), { 'drop-before': dropTarget === w.id }]"
               :draggable="editing"
               @dragstart="onDragStart(w.id)"
               @dragover.prevent="onCellDragOver(w.id)"
               @dragleave="onCellDragLeave(w.id)"
               @drop="onDrop(w.id)">
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
          <select v-if="pages.length > 1" class="move-page" title="Move to another page" aria-label="Move to another page"
                  @change="moveWidgetToPage(w.id, $event.target.value); $event.target.value = ''">
            <option value="">Move…</option>
            <option v-for="p in pages.filter((x) => x.id !== activePageId)" :key="p.id" :value="p.id">{{ p.name }}</option>
          </select>
          <button class="remove" title="Remove" @click="remove(w.id)">×</button>
        </div>
        <!-- Insert-before drop indicator -->
        <div v-if="editing && dropTarget === w.id && dragFromPalette" class="drop-before-bar"></div>
        <!-- Widgets below the fold build as they scroll in, so opening the
             dashboard does not fetch and compute every one at once. -->
        <RenderWhenVisible min-height="140px">
          <component :is="componentFor(w.type)" :widget="w" :is-editing="editing" />
        </RenderWhenVisible>
      </section>

      <!-- Trailing drop zone: append after all existing widgets -->
      <div v-if="editing && dragFromPalette" class="grid-end-drop"
           :class="{ active: dropTarget === '__end__' }"
           @dragover.prevent="onCellDragOver('__end__')"
           @dragleave="onCellDragLeave('__end__')"
           @drop="onGridEndDrop">
        <span>Drop here to add at end</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
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
import DashboardSeasonOutlook from '~/components/DashboardSeasonOutlook.vue'

const { isAuthed } = useAuth()
const {
  orderedWidgets, load, add, addBefore, remove, reorder, reset, updateSettings,
  pages, activePage, activePageId, addPage, renamePage, removePage, movePage, setActivePage, moveWidgetToPage,
} = useDashboardState()
const pageIndex = computed(() => pages.value.findIndex((p) => p.id === activePageId.value))

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
  { type: 'season-outlook', label: 'Season outlook', icon: '📆', component: DashboardSeasonOutlook },
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

// The layout a new member lands on: three pages, each sized to read well on a
// 3-column grid and to stack in a sensible order on a phone.
//   Today   — what to go out for, and where: fruiting now, weather, then the
//             map across the full width so its height never strands a short cell.
//   Seasons — the longer view: the outlook, the phenology calendar, favourites.
//   Data    — the collection and the models built on it.
const DEFAULT_PAGES = [
  { id: 'page-today', name: 'Today', widgets: [
    ['now-fruiting', {}],
    ['weather-forecast', {}],
    ['recent-rain-temp', {}],
    ['map-preview', { colspan: 'full', tall: 1 }],
    ['species-list', {}],
  ] },
  { id: 'page-seasons', name: 'Seasons', widgets: [
    ['season-outlook', {}],
    ['favorites', {}],
    ['quick-filters', {}],
    ['phenology', { colspan: 'full' }],
  ] },
  { id: 'page-data', name: 'Data', widgets: [
    ['overview', {}],
    ['datasets', { colspan: 2 }],
    ['env-stats', { colspan: 2 }],
    ['data-quality', {}],
    ['model-leaderboard', {}],
    ['recent-jobs', {}],
    ['quota', {}],
  ] },
]
const DEFAULTS = {
  widgets: DEFAULT_PAGES.flatMap((p) => p.widgets.map(([type, settings]) => ({ id: `def-${type}`, type, settings }))),
  order: DEFAULT_PAGES.flatMap((p) => p.widgets.map(([type]) => `def-${type}`)),
  pages: DEFAULT_PAGES.map((p) => ({ id: p.id, name: p.name, order: p.widgets.map(([type]) => `def-${type}`) })),
  activePage: 'page-today',
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
    'is-map':   w.type === 'map-preview',
  }
}

const editing = ref(false)

function hasType(type) { return orderedWidgets.value.some((w) => w.type === type) }
function addDefaults() {
  if (window.confirm('Replace every page with the default layout?')) reset(structuredClone(DEFAULTS))
}
function deletePage() {
  const page = activePage.value
  if (!page || pages.value.length <= 1) return
  const n = page.order.length
  if (n && !window.confirm(`Delete “${page.name}” and its ${n} widget${n === 1 ? '' : 's'}?`)) return
  removePage(page.id)
}

// ── Drag state ────────────────────────────────────────────────────────────────
const dragging = ref('')          // id of an existing widget being reordered
const dragFromPalette = ref(null) // { type, label } when dragging from palette
const dropTarget = ref('')        // id of cell showing the drop-before indicator

// Dropping an existing widget onto a page tab moves it to that page.
const tabDrop = ref('')
function onTabDragOver(pageId) {
  if (dragging.value && pageId !== activePageId.value) tabDrop.value = pageId
}
function onTabDragLeave(pageId) { if (tabDrop.value === pageId) tabDrop.value = '' }
function onTabDrop(pageId) {
  if (dragging.value) moveWidgetToPage(dragging.value, pageId)
  dragging.value = ''
  tabDrop.value = ''
}

// Existing widget reorder
function onDragStart(id) {
  dragging.value = id
  dragFromPalette.value = null
}

// Palette drag
function onPaletteDragStart(w) {
  dragFromPalette.value = { type: w.type, label: w.label }
  dragging.value = ''
}
function onPaletteDragOver() { /* allow drop back onto palette to cancel */ }
function onPaletteDrop() {
  dragFromPalette.value = null
  dropTarget.value = ''
}

// Cell hover while dragging
function onCellDragOver(id) {
  if (dragFromPalette.value) dropTarget.value = id
}
function onCellDragLeave(id) {
  if (dropTarget.value === id) dropTarget.value = ''
}

function onDrop(targetId) {
  if (dragFromPalette.value) {
    // Insert the new widget before the target cell.
    const def = dragFromPalette.value
    if (!hasType(def.type)) addBefore({ type: def.type, label: def.label }, targetId)
    dragFromPalette.value = null
    dropTarget.value = ''
    return
  }
  if (dragging.value && dragging.value !== targetId) reorder(dragging.value, targetId)
  dragging.value = ''
  dropTarget.value = ''
}

function onGridEndDrop() {
  if (dragFromPalette.value) {
    const def = dragFromPalette.value
    if (!hasType(def.type)) add({ type: def.type, label: def.label })
    dragFromPalette.value = null
    dropTarget.value = ''
  }
}

onMounted(() => { load(DEFAULTS) })
</script>

<style scoped>
.dash-guard { max-width: 420px; margin: 4rem auto; text-align: center; display: grid; gap: 0.8rem; }
.dashboard { padding: 1rem 1rem 3rem; }
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
.btn.sm { padding: 0.3rem 0.6rem; font-size: 0.8rem; }
.btn.danger { color: #b3492f; }
.btn:disabled { opacity: 0.45; cursor: default; }

.page-tabs {
  display: flex; gap: 0.25rem; margin: 0 0 1rem; border-bottom: 1px solid var(--border, #e5e5e5);
  overflow-x: auto; scrollbar-width: none; -webkit-overflow-scrolling: touch;
}
.page-tabs::-webkit-scrollbar { display: none; }
.page-tab {
  flex: none; border: 0; border-bottom: 2px solid transparent; background: none; color: var(--muted, #777);
  font: inherit; font-size: 0.92rem; padding: 0.5rem 0.9rem; margin-bottom: -1px; cursor: pointer;
  display: inline-flex; align-items: center; gap: 0.35rem; border-radius: 6px 6px 0 0;
}
.page-tab:hover { color: var(--text, #222); }
.page-tab.active { color: var(--text, #222); border-bottom-color: var(--accent, #2a78d6); font-weight: 600; }
.page-tab.drop-into { background: color-mix(in srgb, var(--accent, #2a78d6) 12%, transparent); color: var(--accent, #2a78d6); }
.page-tab.add { color: var(--accent, #2a78d6); }
.tab-count {
  font-size: 0.68rem; font-weight: 500; background: var(--surface-2, #eee); color: var(--muted, #777);
  border-radius: 999px; padding: 0 0.4rem; line-height: 1.5;
}

.page-tools { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 0.5rem 0.8rem; margin-bottom: 0.8rem; }
.page-name { display: flex; flex-direction: column; gap: 0.2rem; font-size: 0.75rem; color: var(--muted, #777); flex: 1 1 180px; max-width: 280px; }
.page-name input {
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff); color: var(--text, #222);
  border-radius: 6px; padding: 0.35rem 0.5rem; font: inherit; font-size: 0.88rem;
}
.page-tool-btns { display: flex; gap: 0.35rem; }

.palette { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; padding: 0.8rem; background: var(--surface-2, #f6f6f6); border-radius: 10px; }
.palette-item {
  display: inline-flex; align-items: center; gap: 0.4rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); border-radius: 999px; padding: 0.35rem 0.8rem;
  font: inherit; font-size: 0.82rem; cursor: grab; user-select: none;
  transition: border-color 0.12s, box-shadow 0.12s;
}
.palette-item:not(.disabled):hover { border-color: var(--accent, #2a78d6); box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent, #2a78d6) 20%, transparent); }
.palette-item.disabled { opacity: 0.5; cursor: default; }
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
  display: flex; flex-direction: column; min-width: 0; overflow: hidden;
  transition: box-shadow 0.15s;
}
/* RenderWhenVisible wrapper and the .dash-widget inside both stretch to fill the cell */
.dash-cell :deep(.render-when-visible) { flex: 1; min-height: 0; }
.dash-cell :deep(.dash-widget) { flex: 1; min-height: 0; }

.dash-cell.editing { border-style: dashed; cursor: grab; }
.dash-cell.editing:hover { box-shadow: 0 0 0 2px var(--accent, #2a78d6); }
.dash-cell.span-2 { grid-column: span 2; }
.dash-cell.span-full { grid-column: 1 / -1; }
.dash-cell.tall { height: 520px; min-height: unset; }
.dash-cell.tall-2 { height: 760px; min-height: unset; }
/* The map preview is only useful with room to see the Front Range, so it gets
   its own, taller steps. Classes apply to saved layouts too, not just new ones. */
.dash-cell.is-map { height: 440px; min-height: unset; }
.dash-cell.is-map.tall { height: 680px; }
.dash-cell.is-map.tall-2 { height: 900px; }

.cell-bar { display: flex; justify-content: space-between; align-items: center; margin: -0.3rem -0.3rem 0.4rem; gap: 0.3rem; }
.drag { color: var(--muted, #aaa); cursor: grab; user-select: none; }

.size-btns { display: flex; gap: 2px; margin: 0 auto 0 0.5rem; }
.size-btns button {
  border: 1px solid var(--border, #ddd); background: var(--surface-2, #f5f5f5); color: var(--muted, #888);
  border-radius: 4px; padding: 2px 6px; font-size: 0.7rem; cursor: pointer; line-height: 1.4;
}
.size-btns button:hover { background: var(--surface-3, #eee); color: var(--text, #222); }
.size-btns button.on { background: var(--accent, #2a78d6); border-color: var(--accent, #2a78d6); color: #fff; }

.move-page {
  border: 1px solid var(--border, #ddd); background: var(--surface-2, #f5f5f5); color: var(--muted, #888);
  border-radius: 4px; font: inherit; font-size: 0.7rem; padding: 1px 2px; max-width: 6.5rem; cursor: pointer;
}
.remove {
  border: 0; background: none; color: var(--muted, #999); font-size: 1.2rem; line-height: 1; cursor: pointer;
}
.remove:hover { color: #b3492f; }

/* Drop-before indicator on a cell */
.dash-cell.drop-before { box-shadow: -4px 0 0 var(--accent, #2a78d6), 0 0 0 2px color-mix(in srgb, var(--accent, #2a78d6) 25%, transparent); }
.drop-before-bar {
  position: absolute; top: 0; left: -2px; bottom: 0; width: 4px;
  background: var(--accent, #2a78d6); border-radius: 2px; z-index: 10;
  pointer-events: none;
}

/* End-of-grid drop zone, only visible while dragging from palette */
.grid-end-drop {
  min-height: 80px; border: 2px dashed var(--border, #ccc);
  border-radius: 12px; display: flex; align-items: center; justify-content: center;
  color: var(--muted, #aaa); font-size: 0.82rem; transition: border-color 0.15s, background 0.15s;
  grid-column: span 1;
}
.grid-end-drop.active {
  border-color: var(--accent, #2a78d6);
  background: color-mix(in srgb, var(--accent, #2a78d6) 6%, transparent);
  color: var(--accent, #2a78d6);
}

/* Empty-state drop target */
.dash-empty.drop-target {
  border: 2px dashed var(--accent, #2a78d6);
  background: color-mix(in srgb, var(--accent, #2a78d6) 5%, transparent);
  color: var(--accent, #2a78d6);
  border-radius: 12px; padding: 2rem;
}

@media (max-width: 600px) {
  .dashboard { padding: 0.75rem 0.75rem 2rem; }
  .dash-grid { grid-template-columns: 1fr; gap: 0.75rem; }
  .dash-cell.span-2, .dash-cell.span-full { grid-column: 1; }
  /* Leave a strip of the next widget in sight so the page still reads as scrollable. */
  .dash-cell.is-map { max-height: 75svh; }
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
