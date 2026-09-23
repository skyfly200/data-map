<template>
  <div v-if="open" ref="win" class="lm" :class="{ docked }" role="dialog" aria-label="Layer manager">
    <header class="lm-head">
      <div class="lm-title">
        <strong>Layers</strong>
        <span v-if="activeList.length" class="lm-count">{{ activeList.length }} on</span>
      </div>
      <div class="lm-head-acts">
        <button v-if="solo" class="lm-text-btn solo-off" title="Draw every layer again"
                @click="$emit('solo', '')">Un-solo</button>
        <button v-if="activeList.length" class="lm-text-btn" :title="allVisible ? 'Hide all layers' : 'Show all layers'"
                @click="toggleAllVisibility">
          <span class="lm-eye-icon" aria-hidden="true">{{ allVisible ? '👁' : '👁‍🗨' }}</span>
        </button>
        <button v-if="activeList.length" class="lm-text-btn" title="Switch every overlay off"
                @click="$emit('clear')">Clear</button>
        <button class="lm-close" aria-label="Close the layer manager" @click="$emit('close')">×</button>
      </div>
    </header>

    <!-- Anything the host wants above the stack. On a phone the map's basemap
         and heatmap controls render here, because three more buttons did not
         fit on the bar beside them. -->
    <div v-if="$slots.top" class="lm-top"><slot name="top" /></div>

    <!-- The active stack, first and separately.
         An overlay list is read in two completely different ways: "what can I
         add" is a browse, "what is on and in what order" is a glance. The old
         single dropdown answered only the first, and the second had to be
         reconstructed by scanning forty checkboxes for ticks. -->
    <section v-if="activeList.length" class="lm-active" :class="{ collapsed: activePanelCollapsed }">
      <div class="lm-sec-head">
        <button class="lm-collapse-toggle" @click="activePanelCollapsed = !activePanelCollapsed"
                :aria-expanded="String(!activePanelCollapsed)" title="Collapse active layers">
          <span class="lm-caret" :class="{ open: !activePanelCollapsed }" aria-hidden="true">▸</span>
          <span>Drawn, top first</span>
        </button>
        <HelpLink option="map-layer-order" />
      </div>
      <ul class="lm-stack">
        <li v-for="(item, i) in activeList" :key="item.key" class="lm-on"
            :class="{ muted: solo && solo !== item.key }">
          <div class="lm-on-top">
            <button class="lm-swatch" :title="`Hide ${item.name}`" @click="$emit('toggle', item.key)">
              <span class="lm-tick" aria-hidden="true">✓</span>
            </button>
            <span class="lm-on-name" :title="item.name">{{ item.name }}</span>
            <span v-if="eeLoading.has(item.key)" class="lm-ee-spinner" aria-label="Rendering…" title="Rendering layer…"></span>
            <!-- Solo answers "what is this one contributing", which otherwise
                 costs you the stack you built and a minute rebuilding it. The
                 others stay ticked; they are only not drawn. -->
            <button class="lm-solo" :class="{ on: solo === item.key }"
                    :aria-pressed="String(solo === item.key)"
                    :title="solo === item.key ? 'Draw every layer again' : `Draw only ${item.name}`"
                    @click="$emit('solo', solo === item.key ? '' : item.key)">S</button>
            <!-- Stacking, because overlays hide each other: land ownership under
                 a hillshade is a different map from the same two the other way
                 up, and there is no other way to say which you meant. -->
            <span class="lm-order">
              <button :disabled="i === 0" title="Send to the top"
                      @click="$emit('move', item.key, 'top')">⤒</button>
              <button :disabled="i === 0" title="Move up" @click="$emit('move', item.key, -1)">▲</button>
              <button :disabled="i === activeList.length - 1" title="Move down"
                      @click="$emit('move', item.key, 1)">▼</button>
              <button :disabled="i === activeList.length - 1" title="Send to the bottom"
                      @click="$emit('move', item.key, 'bottom')">⤓</button>
            </span>
          </div>
          <label class="lm-op">
            <span class="lm-op-label">{{ Math.round(opacityOf(item.key) * 100) }}%</span>
            <input type="range" min="0.05" max="1" step="0.05" :value="opacityOf(item.key)"
                   :aria-label="`Opacity of ${item.name}`"
                   @input="$emit('opacity', item.key, Number($event.target.value))" />
          </label>
          <!-- Opacity and blending answer different questions, and opacity
               answers one of them badly: two layers at 50% is both washed out,
               where multiply keeps both at full strength and combines them by
               value. Shown per layer because one layer in a stack is usually
               the one that should combine. Blend controls now hidden under Advanced. -->
          <div class="lm-blend-wrapper">
            <button class="lm-advanced-toggle" @click="toggleBlend(item.key)"
                    :aria-expanded="String(expandedBlends.has(item.key))">
              <span class="lm-caret" :class="{ open: expandedBlends.has(item.key) }" aria-hidden="true">▸</span>
              Advanced
            </button>
            <div v-show="expandedBlends.has(item.key)" class="lm-blend-controls">
              <label class="lm-blend">
                <span class="lm-blend-label">Blend</span>
                <select :value="blendOf(item.key)" :aria-label="`Blend mode of ${item.name}`"
                        :title="blendNote(blendOf(item.key))"
                        @change="$emit('blend', item.key, $event.target.value)">
                  <option value="">{{ inheritLabel }}</option>
                  <option v-for="m in BLEND_MODES" :key="m.key" :value="m.key" :title="m.note">
                    {{ m.label }}
                  </option>
                </select>
              </label>
            </div>
          </div>
        </li>
      </ul>
    </section>

    <!-- Collapsible layer selection panel -->
    <section class="lm-selection" :class="{ collapsed: selectionPanelCollapsed }">
      <div class="lm-search">
        <input v-model="query" type="search" placeholder="Search layers"
               aria-label="Search layers" />
      </div>

    <!-- How the browse below is sectioned. Subject is the catalogue's own
         grouping; source and type re-cut the same layers for when you are
         after a provider or a kind of raster rather than a topic. -->
    <div class="lm-groupby" role="group" aria-label="Group layers by">
      <span class="lm-groupby-label">Group by</span>
      <div class="lm-seg">
        <button v-for="m in GROUP_MODES" :key="m.key" type="button"
                class="lm-seg-btn" :class="{ on: groupMode === m.key }"
                :aria-pressed="groupMode === m.key" @click="setGroupMode(m.key)">
          {{ m.label }}
        </button>
      </div>
    </div>

    <div class="lm-body">
      <p v-if="!filtered.length" class="lm-empty">
        Nothing matches “{{ query }}”.
      </p>

      <!-- Each group is an expansion panel. The catalogue has grown past what a
           single open list can carry without scrolling — nine reference layers,
           the Earth Engine layers, and whatever assets are registered — so a
           reader browses one section at a time rather than the whole inventory
           at once. A search overrides the panels and opens everything that
           matches, since when you are looking for a layer you do not want to
           first guess which section hid it. -->
      <section v-for="g in filtered" :key="g.label" class="lm-group">
        <button type="button" class="lm-group-head" :aria-expanded="isOpen(g.label)"
                @click="togglePanel(g.label)">
          <span class="lm-caret" :class="{ open: isOpen(g.label) }" aria-hidden="true">▸</span>
          <span class="lm-group-label">{{ g.label }}</span>
          <span class="lm-group-meta">
            <span v-if="activeCount(g)" class="lm-group-on">{{ activeCount(g) }} on</span>
            <span class="lm-group-total">{{ g.items.length }}</span>
          </span>
        </button>
        <div v-show="isOpen(g.label)" class="lm-group-items">
          <label v-for="o in g.items" :key="o.key" class="lm-row" :class="{ on: active.has(o.key) }">
            <input type="checkbox" :checked="active.has(o.key)" @change="$emit('toggle', o.key)" />
            <span class="lm-row-main">
              <span class="lm-row-name">{{ o.name }}</span>
              <span v-if="eeLoading.has(o.key)" class="lm-ee-spinner" aria-label="Rendering…" title="Rendering layer…"></span>
              <!-- Listed but marked, rather than hidden: knowing FRMS
                   computes it is part of what membership is for. -->
              <em v-if="o.tier && o.tier !== 'free'" class="lm-tier">{{ o.tier }}</em>
              <small v-if="o.note" class="lm-note">{{ o.note }}</small>
            </span>
          </label>
        </div>
      </section>
    </div>
  </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { filterLayerGroups } from '~/composables/mapLayers'
import { BLEND_MODES, blendLabel } from '~/composables/blendModes'

// A window for managing the overlays, rather than one dropdown holding all of
// them.
//
// The list had grown past what a dropdown can carry — the built-in catalogue,
// the Earth Engine layers, and now however many assets FRMS registers of
// its own. Beyond about a dozen entries a flat list of checkboxes stops being a
// control and becomes an inventory: you cannot see what is on without reading
// all of it, cannot say which draws over which, and cannot dim one without
// dimming every one.
//
// Not modal, and that is the point: switching a layer on is something you judge
// by looking at the map, so the map has to stay visible and usable underneath.

const props = defineProps({
  open: { type: Boolean, default: false },
  // [{ label, items: [{ key, name, group, tier, note }] }]
  groups: { type: Array, default: () => [] },
  active: { type: Set, default: () => new Set() },
  // Active keys, topmost first. The map owns the stacking; this only shows it.
  order: { type: Array, default: () => [] },
  opacity: { type: Object, default: () => ({}) },
  // Per-layer blend overrides, key → mode. An absent key inherits the stack
  // default rather than meaning "normal", so changing the default still moves
  // every layer nobody has set by hand.
  blend: { type: Object, default: () => ({}) },
  // The stack default, for naming the inherit option — a dropdown whose first
  // entry says "Default" and nothing else makes you go and look it up.
  stackBlend: { type: String, default: 'normal' },
  // The one layer drawn on its own, or '' for all of them.
  solo: { type: String, default: '' },
  // EE layers currently fetching their tile template: Map<key, name>
  eeLoading: { type: Map, default: () => new Map() },
  // Docked against the controls on a wide screen; a bottom sheet on a phone.
  docked: { type: Boolean, default: true },
})

defineEmits(['toggle', 'opacity', 'move', 'blend', 'solo', 'clear', 'close'])

const query = ref('')
const win = ref(null)

// Collapsible active layers panel - state persisted in localStorage
const activePanelCollapsed = ref(false)
// Collapsible layer selection panel - state persisted in localStorage
const selectionPanelCollapsed = ref(false)
onMounted(() => {
  if (import.meta.client) {
    try {
      const savedActive = localStorage.getItem('layer-manager-active-collapsed')
      if (savedActive) activePanelCollapsed.value = savedActive === 'true'
      const savedSelection = localStorage.getItem('layer-manager-selection-collapsed')
      if (savedSelection) selectionPanelCollapsed.value = savedSelection === 'true'
    } catch {
      // Preference reads silently fall back to defaults.
    }
  }
})
watch(activePanelCollapsed, (val) => {
  if (import.meta.client) {
    try {
      localStorage.setItem('layer-manager-active-collapsed', String(val))
    } catch (e) {
      // Preference writes silently fail if storage is blocked.
    }
  }
})
watch(selectionPanelCollapsed, (val) => {
  if (import.meta.client) {
    try {
      localStorage.setItem('layer-manager-selection-collapsed', String(val))
    } catch (e) {
      // Preference writes silently fail if storage is blocked.
    }
  }
})

// Per-layer blend controls expanded state
const expandedBlends = ref(new Set())
function toggleBlend(key) {
  const next = new Set(expandedBlends.value)
  if (next.has(key)) next.delete(key)
  else next.add(key)
  expandedBlends.value = next
}

// Track which layers are visible for global toggle
const allVisible = computed(() => props.order.length > 0 && props.order.every(key => props.active.has(key)))
function toggleAllVisibility() {
  if (allVisible.value) {
    // Hide all - emit clear
    emit('clear')
  } else {
    // Show all - emit toggle for each inactive layer
    props.groups.forEach(g => {
      g.items.forEach(item => {
        if (!props.active.has(item.key)) emit('toggle', item.key)
      })
    })
  }
}

const opacityOf = (key) => props.opacity[key] ?? 1
const blendOf = (key) => props.blend[key] || ''
const blendNote = (mode) => BLEND_MODES.find((m) => m.key === mode)?.note || ''

// The default is only the default when there is a stack, so the label says so
// rather than promising a mode a single layer will not draw with.
const inheritLabel = computed(() => (props.stackBlend === 'normal'
  ? 'Default (normal)'
  : `Default (${blendLabel(props.stackBlend).toLowerCase()} when stacked)`))

// How the browse list is carved into sections. Subject is the catalogue's own
// grouping (Fire, Soil, …); the other two re-cut the same layers by where the
// data comes from and what kind of raster it is, which is how you look when you
// are after "everything from Sentinel-2" or "every categorical layer" rather
// than a subject.
const GROUP_MODES = [
  { key: 'group', label: 'Subject' },
  { key: 'source', label: 'Source' },
  { key: 'type', label: 'Type' },
]
const groupMode = ref('group')

// Subject grouping arrives pre-built and in catalogue order, so it is used as
// given. The other two are rebuilt from the same items, keyed by the chosen
// field, each section in first-seen order and the sections sorted by name — but
// with the imagery/other catch-alls kept last so a real source is never buried
// under them.
const displayGroups = computed(() => {
  if (groupMode.value === 'group') return props.groups
  const field = groupMode.value
  const order = []
  const map = new Map()
  for (const g of props.groups) {
    for (const o of g.items) {
      const label = o[field] || 'Other'
      if (!map.has(label)) { map.set(label, []); order.push(label) }
      map.get(label).push(o)
    }
  }
  const trailing = (label) => /^(Basemap|Other)\b/.test(label)
  return order
    .sort((a, b) => (trailing(a) - trailing(b)) || a.localeCompare(b))
    .map((label) => ({ label, items: map.get(label) }))
})

function setGroupMode(mode) {
  if (mode === groupMode.value) return
  groupMode.value = mode
  seedPanels()
}

// Which expansion panels are open, keyed by group label. A search opens every
// matching panel regardless (see isOpen), so this only governs the browse.
const openPanels = ref(new Set())

const activeCount = (g) => g.items.reduce((n, o) => n + (props.active.has(o.key) ? 1 : 0), 0)

const isOpen = (label) => (query.value ? true : openPanels.value.has(label))

function togglePanel(label) {
  const next = new Set(openPanels.value)
  if (next.has(label)) next.delete(label)
  else next.add(label)
  openPanels.value = next
}

// On opening the manager, expand the sections that already have a layer on, so
// what you turned on is in front of you. If nothing is on there is nothing to
// prioritise, so open everything rather than present a wall of collapsed
// headers with no hint of what is inside.
function seedPanels() {
  const shown = displayGroups.value
  const withActive = shown
    .filter((g) => g.items.some((o) => props.active.has(o.key)))
    .map((g) => g.label)
  openPanels.value = new Set(withActive.length ? withActive : shown.map((g) => g.label))
}

/** Every layer by key, so the active stack can be named without a second list. */
const byKey = computed(() => {
  const map = new Map()
  for (const g of props.groups) for (const o of g.items) map.set(o.key, o)
  return map
})

const activeList = computed(() =>
  props.order.map((key) => byKey.value.get(key)).filter(Boolean))

// A group matches as a prefix, a layer anywhere. See filterLayerGroups: the
// obvious version of this returned the whole Terrain group for "rain".
const filtered = computed(() => filterLayerGroups(displayGroups.value, query.value))

// A search left over from last time hides most of the catalogue on reopening,
// which reads as layers having gone missing.
watch(() => props.open, (v) => {
  if (v) seedPanels()
  else query.value = ''
})

// The catalogue (Earth Engine layers especially) can arrive after the manager
// is already open. Seed the panels once the groups it should reflect exist.
watch(() => props.groups, () => {
  if (props.open && !openPanels.value.size) seedPanels()
})

onMounted(() => { if (props.open) seedPanels() })
</script>

<style scoped>
.lm {
  position: absolute; z-index: 1200;
  top: 8px; right: 8px; width: 310px;
  max-height: calc(100% - 16px);
  display: flex; flex-direction: column;
  background: var(--surface, #fff); color: var(--text, #222);
  border: 1px solid var(--border, #ddd); border-radius: 10px;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.22);
  font-size: 0.82rem;
  overflow: hidden;
}

.lm-head {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 9px 10px 9px 12px; border-bottom: 1px solid var(--border-soft, #eee);
  flex: 0 0 auto;
}
.lm-title { display: flex; align-items: baseline; gap: 8px; }
.lm-title strong { font-size: 0.88rem; }
.lm-count {
  background: var(--surface-2, #eee); border-radius: 999px; padding: 1px 7px;
  font-size: 0.68rem; color: var(--muted, #666);
}
.lm-head-acts { display: flex; align-items: center; gap: 4px; }
.lm-text-btn {
  border: 0; background: transparent; color: var(--muted, #666);
  font: inherit; font-size: 0.74rem; cursor: pointer; padding: 3px 6px; border-radius: 5px;
}
.lm-text-btn:hover { color: var(--text); background: var(--surface-2); }
.lm-eye-icon { font-size: 0.9rem; }
.lm-close {
  border: 0; background: transparent; color: var(--muted, #777);
  font-size: 1.25rem; line-height: 1; cursor: pointer; padding: 0 4px;
}
.lm-close:hover { color: var(--text); }

.lm-sec-head {
  display: flex; align-items: center; gap: 6px;
  font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted, #777); font-weight: 700; margin-bottom: 5px;
}

/* The drawn stack: its own scroll at the top, never the whole window.
 *
 * Two things had to be true for that and neither was. It was flex: 0 0 auto, so
 * it could not shrink and simply took whatever it needed. And its cap was a
 * percentage, which resolves against the parent's height — but .lm has only a
 * max-height, so its height is indefinite, the percentage was ignored, and the
 * cap did nothing at all. With four or five layers on, the stack filled the
 * panel and pushed the browse list out of it entirely: nothing left to scroll
 * to, and no way to reach another layer without switching one off.
 *
 * So: shrinkable, and capped in units that always resolve. */
.lm-top {
  flex: 0 0 auto; padding: 8px 12px;
  border-bottom: 1px solid var(--border-soft, #eee);
}
.lm-top :deep(.lm-extra) { font-size: 0.78rem; }
.lm-top :deep(.lm-extra + .lm-extra) { margin-top: 6px; }
.lm-top :deep(.lm-extra > summary) {
  cursor: pointer; font-weight: 600; color: var(--text);
  display: flex; align-items: baseline; gap: 6px; padding: 3px 0;
}
.lm-top :deep(.lm-extra > summary em) {
  font-style: normal; color: var(--muted); font-weight: 400;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.lm-top :deep(.lm-extra[open] > summary) { margin-bottom: 4px; }

.lm-active {
  flex: 0 1 auto;
  min-height: 0;
  max-height: 260px;
  overflow-y: auto; overscroll-behavior: contain;
  padding: 9px 12px; border-bottom: 1px solid var(--border-soft, #eee);
  background: var(--surface-2, #f7f7f7);
}
.lm-active.collapsed {
  max-height: none;
  overflow-y: visible;
  padding: 0 12px;
}
.lm-stack { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 7px; }
.lm-on-top { display: flex; align-items: center; gap: 7px; }
.lm-swatch {
  flex: 0 0 auto; width: 17px; height: 17px; padding: 0; cursor: pointer;
  border: 1px solid var(--accent, #2b7a3d); background: var(--accent, #2b7a3d);
  border-radius: 4px; color: #fff; display: inline-flex; align-items: center; justify-content: center;
}
.lm-tick { font-size: 0.66rem; line-height: 1; }
.lm-on-name {
  flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  font-weight: 600;
}
.lm-order { flex: 0 0 auto; display: flex; gap: 2px; }
.lm-order button {
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff); color: var(--muted, #666);
  border-radius: 4px; width: 20px; height: 20px; padding: 0; cursor: pointer;
  font-size: 0.6rem; line-height: 1;
}
.lm-order button:hover:not(:disabled) { color: var(--text); border-color: var(--muted); }
.lm-order button:disabled { opacity: 0.35; cursor: default; }

.lm-op { display: flex; align-items: center; gap: 7px; padding-left: 24px; }
.lm-op-label {
  flex: 0 0 auto; width: 4ch; color: var(--muted, #777); font-size: 0.7rem;
  font-variant-numeric: tabular-nums;
}
.lm-op input { flex: 1 1 auto; min-width: 0; accent-color: var(--accent, #2b7a3d); }

/* Collapsible section header */
.lm-collapse-toggle {
  display: flex; align-items: center; gap: 6px;
  border: 0; background: transparent; color: var(--text, #222);
  font: inherit; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted, #777); font-weight: 700; cursor: pointer;
  padding: 0; margin-right: 6px;
}
.lm-collapse-toggle:hover { color: var(--text); }
.lm-caret {
  flex: 0 0 auto; color: var(--muted, #888); font-size: 0.7rem;
  transition: transform 0.12s ease; transform: rotate(0deg);
}
.lm-caret.open { transform: rotate(90deg); }

/* Advanced blend controls wrapper */
.lm-blend-wrapper {
  padding-left: 24px; margin-top: 2px;
}
.lm-advanced-toggle {
  display: inline-flex; align-items: center; gap: 4px;
  border: 0; background: transparent; color: var(--muted, #777);
  font-size: 0.7rem; cursor: pointer; padding: 2px 4px;
  border-radius: 4px;
}
.lm-advanced-toggle:hover { background: var(--surface-2, #eee); color: var(--text); }
.blend-caret {
  font-size: 0.6rem; transition: transform 0.12s ease; transform: rotate(0deg);
}
.blend-caret.open { transform: rotate(180deg); }
.lm-blend-controls {
  margin-top: 4px; padding: 6px; background: var(--surface, #fff);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px;
}
.lm-blend { display: flex; align-items: center; gap: 7px; }
.lm-blend-label { flex: 0 0 auto; width: 4ch; color: var(--muted, #777); font-size: 0.7rem; }
.lm-blend select {
  flex: 1 1 auto; min-width: 0;
  background: var(--surface, #fff); color: var(--text, #222);
  border: 1px solid var(--border, #ddd); border-radius: 4px;
  padding: 2px 4px; font: inherit; font-size: 0.72rem;
}

/* Global visibility toggle button */
.lm-icon-btn {
  border: 0; background: transparent; color: var(--muted, #666);
  font-size: 1rem; line-height: 1; cursor: pointer; padding: 0 4px;
  border-radius: 4px;
}
.lm-icon-btn:hover { background: var(--surface-2, #eee); color: var(--text); }
.visibility-icon { font-size: 1rem; }

/* One letter, because it sits between the name and four order buttons and a
   word would push them off a phone. It is the standard mark for this in every
   mixer and every editor that has the idea at all. */
.lm-solo {
  flex: 0 0 auto; width: 20px; height: 20px; padding: 0; cursor: pointer;
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff);
  color: var(--muted, #666); border-radius: 4px;
  font-size: 0.66rem; font-weight: 700; line-height: 1;
}
.lm-solo:hover { color: var(--text); border-color: var(--muted); }
.lm-solo.on {
  background: #b3822f; border-color: #b3822f; color: #fff;
}

/* Still listed, still ticked, just not drawn. Dimmed rather than hidden, so
   the stack you built is still the stack you can see. */
.lm-on.muted { opacity: 0.45; }
.lm-text-btn.solo-off { color: #b3822f; }

.lm-search { flex: 0 0 auto; padding: 9px 12px 0; }
.lm-search input {
  width: 100%; box-sizing: border-box;
  background: var(--surface-2, #f4f4f4); color: var(--text, #222);
  border: 1px solid var(--border, #ddd); border-radius: 6px;
  padding: 5px 8px; font: inherit; font-size: 0.78rem;
}
.lm-search input:focus { border-color: var(--accent, #2b7a3d); outline: none; }

/* Collapsible selection panel */
.lm-selection {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  min-height: 0;
}
.lm-selection.collapsed {
  flex: 0 0 auto;
}

.lm-groupby {
  flex: 0 0 auto; display: flex; align-items: center; gap: 8px;
  padding: 8px 12px 0;
}
.lm-groupby-label {
  flex: 0 0 auto; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted, #777); font-weight: 700;
}
.lm-seg {
  flex: 1 1 auto; display: flex; border: 1px solid var(--border, #ddd);
  border-radius: 6px; overflow: hidden;
}
.lm-seg-btn {
  flex: 1 1 0; border: 0; border-left: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #666);
  font: inherit; font-size: 0.72rem; cursor: pointer; padding: 4px 6px;
}
.lm-seg-btn:first-child { border-left: 0; }
.lm-seg-btn:hover { background: var(--surface-2, #f4f4f4); color: var(--text); }
.lm-seg-btn.on { background: var(--accent, #2b7a3d); color: #fff; font-weight: 600; }

/* min-height: 0, because a flex item's default min-height is auto — it refuses
   to shrink below its content, so overflow-y: auto never gets anything to
   scroll and the item pushes the panel open instead. */
.lm-body {
  flex: 1 1 auto; min-height: 0;
  overflow-y: auto; overscroll-behavior: contain; padding: 10px 12px 12px;
}
.lm-selection.collapsed .lm-body {
  display: none;
}
.lm-empty { margin: 4px 0; color: var(--muted, #777); font-size: 0.78rem; }

.lm-group { border-bottom: 1px solid var(--border-soft, #eee); }
.lm-group:last-child { border-bottom: 0; }

.lm-group-head {
  display: flex; align-items: center; gap: 7px; width: 100%;
  border: 0; background: transparent; color: var(--text, #222);
  font: inherit; text-align: left; cursor: pointer;
  padding: 8px 6px; margin: 0 -6px;
}
.lm-group-head:hover { color: var(--text); }
.lm-caret {
  flex: 0 0 auto; color: var(--muted, #888); font-size: 0.7rem;
  transition: transform 0.12s ease; transform: rotate(0deg);
}
.lm-caret.open { transform: rotate(90deg); }
.lm-group-label {
  flex: 1 1 auto; min-width: 0;
  font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted, #777); font-weight: 700;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.lm-group-meta { flex: 0 0 auto; display: flex; align-items: center; gap: 6px; }
.lm-group-on {
  background: var(--accent, #2b7a3d); color: #fff;
  border-radius: 999px; padding: 1px 7px; font-size: 0.64rem; font-weight: 600;
}
.lm-group-total {
  color: var(--muted, #999); font-size: 0.68rem;
  font-variant-numeric: tabular-nums;
}
.lm-group-items { padding-bottom: 8px; }

.lm-row {
  display: flex; align-items: flex-start; gap: 8px; cursor: pointer;
  padding: 4px 6px; margin: 0 -6px; border-radius: 6px; line-height: 1.4;
}
.lm-row:hover { background: var(--surface-2, #f4f4f4); }
.lm-row input { margin: 3px 0 0; accent-color: var(--accent, #2b7a3d); flex: 0 0 auto; }
.lm-row-main { flex: 1 1 auto; min-width: 0; display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px; }
.lm-row.on .lm-row-name { font-weight: 600; }
.lm-tier {
  font-style: normal; font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.05em;
  background: var(--surface-3, #e6e6e6); color: var(--muted, #666);
  border-radius: 999px; padding: 1px 6px;
}
.lm-note {
  flex: 1 0 100%; color: var(--muted, #777); font-size: 0.72rem; line-height: 1.4;
  /* The caveats are long and worth having, but four lines each turns the list
     into an essay. Two lines is enough to know whether to read the rest in the
     key. */
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
}

@keyframes lm-spin { to { transform: rotate(360deg); } }
.lm-ee-spinner {
  display: inline-block; flex: 0 0 auto; width: 12px; height: 12px;
  border: 2px solid var(--border, #ccc);
  border-top-color: var(--accent, #2b7a3d);
  border-radius: 50%;
  animation: lm-spin 0.7s linear infinite;
}

/* On a phone a floating window over a map is most of the map. A sheet from the
   bottom leaves the top half visible, which is where you look to judge whether
   the layer you just ticked did anything. */
@media (max-width: 720px) {
  .lm {
    top: auto; right: 6px; left: 6px; bottom: 6px; width: auto;
    /* A little taller than it was, but not much: this window is meant to be
       judged against the map behind it, so taking the whole screen would cost
       the thing it is for. */
    max-height: 70%;
  }
  /* The split between the two lists, as a contract rather than as whatever
     flexbox happened to do. The stack takes what it needs up to a third of the
     screen; the browse list is never squeezed below a usable height, because
     reaching it is the reason the window is open. */
  .lm-active { flex: 1 1 auto; max-height: 32vh; }
  .lm-body { min-height: 116px; }

  /* A drawn layer was three stacked rows — name, opacity, blend — so four
     layers filled the sheet and you scrolled a list one item at a time.
     Opacity and blend share a row here; both are still full-width targets. */
  .lm-on {
    display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 4px 10px; align-items: center;
  }
  .lm-on-top { grid-column: 1 / -1; }
  .lm-op, .lm-blend { padding-left: 0; margin-top: 0; }
  .lm-stack { gap: 12px; }

  /* Touch targets, which the desktop sizes are slightly under. */
  .lm-order button, .lm-solo { width: 28px; height: 28px; font-size: 0.72rem; }
  .lm-blend select { padding: 4px 4px; }
}
</style>
