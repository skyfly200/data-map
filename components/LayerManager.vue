<template>
  <!-- Phone: a transparent catcher over the map so a tap outside dismisses the
       sheet. Absent while the sheet is peeked, so the map is fully usable then. -->
  <div v-if="open && compact && !peek" class="lm-backdrop" aria-hidden="true" @click="$emit('close')"></div>
  <div v-if="open" ref="win" class="lm" :class="{ docked, compact, peek, dragging: drag.active }"
       role="dialog" aria-label="Layer manager" :style="sheetStyle">
    <div v-if="compact" class="lm-grab" role="button" tabindex="0"
         :aria-label="peek ? 'Expand layers sheet' : 'Collapse layers sheet'"
         @pointerdown="dragStart" @pointermove="dragMove" @pointerup="dragEnd" @pointercancel="dragEnd"
         @keydown.enter.prevent="peek = !peek" @keydown.space.prevent="peek = !peek">
      <span class="lm-grab-bar" aria-hidden="true"></span>
    </div>
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
        <button v-if="hasChannelOverrides" class="lm-text-btn" title="Reset all channel adjustments"
                @click="$emit('reset-channels')">Reset channels</button>
        <button v-if="activeList.length" class="lm-text-btn" title="Switch every overlay off"
                @click="$emit('clear')">Clear</button>
        <button class="lm-close" aria-label="Close the layer manager" @click="$emit('close')">×</button>
      </div>
    </header>

    <!-- What is on, always visible, including while the sheet is peeked. -->
    <ul v-if="chips.length" class="lm-chips" aria-label="Active layers">
      <li v-for="c in chips" :key="c.key">
        <button type="button" class="lm-chip" :title="`Hide ${c.label}`" :aria-label="`Hide ${c.label}`"
                @click="c.extra ? $emit('extra-off', c.key) : $emit('toggle', c.key)">
          <span class="lm-chip-name">{{ c.label }}</span><span class="lm-chip-x" aria-hidden="true">×</span>
        </button>
      </li>
    </ul>

    <div v-show="!(compact && peek)" class="lm-scroll">
    <!-- On a phone the map's basemap and heatmap controls render here as
         collapsible groups, because they did not fit on the bar. -->
    <section v-if="$slots.basemap" class="lm-sec">
      <button type="button" class="lm-sec-btn" :aria-expanded="secOpen.basemap" @click="toggleSec('basemap')">
        <span class="lm-caret" :class="{ open: secOpen.basemap }" aria-hidden="true">▸</span>
        <span class="lm-sec-title">Basemap</span><em class="lm-sec-meta">{{ basemapName }}</em>
      </button>
      <div v-show="secOpen.basemap" class="lm-sec-body"><slot name="basemap" /></div>
    </section>

    <h3 v-if="compact" class="lm-sec-h">Overlays</h3>

    <!-- Tab bar: Active layers vs Browse catalogue -->
    <div class="lm-tabs" role="tablist">
      <button role="tab" class="lm-tab" :class="{ on: activeTab === 'active' }"
              :aria-selected="activeTab === 'active'" @click="activeTab = 'active'">
        Active
        <span v-if="activeList.length" class="lm-tab-badge">{{ activeList.length }}</span>
      </button>
      <button role="tab" class="lm-tab" :class="{ on: activeTab === 'browse' }"
              :aria-selected="activeTab === 'browse'" @click="activeTab = 'browse'">
        Browse
      </button>
    </div>

    <!-- Active layers tab -->
    <section v-show="activeTab === 'active'" class="lm-active lm-tab-panel" role="tabpanel">
      <div v-if="!activeList.length" class="lm-empty lm-empty-active">
        No layers drawn yet — switch to Browse to add some.
      </div>
      <div v-else class="lm-sec-head">
        <span>Drawn, top first</span>
        <HelpLink option="map-layer-order" />
      </div>
      <ul v-if="activeList.length" class="lm-stack">
        <li v-for="(item, i) in activeList" :key="item.key" class="lm-on"
            :class="{ muted: solo && solo !== item.key }">
          <div class="lm-on-top">
            <button class="lm-swatch" :title="`Hide ${item.name}`" @click="$emit('toggle', item.key)">
              <span class="lm-tick" aria-hidden="true">✓</span>
            </button>
            <span class="lm-on-name" :title="item.name">{{ item.name }}</span>
            <span v-if="eeLoading.has(item.key)" class="lm-ee-spinner" aria-label="Rendering…" title="Rendering layer…"></span>
            <button class="lm-solo" :class="{ on: solo === item.key }"
                    :aria-pressed="String(solo === item.key)"
                    :title="solo === item.key ? 'Draw every layer again' : `Draw only ${item.name}`"
                    @click="$emit('solo', solo === item.key ? '' : item.key)">S</button>
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
              <div class="lm-channels">
                <span class="lm-ch-head">Channels</span>
                <label v-for="ch in ['r','g','b','a']" :key="ch" class="lm-ch-row">
                  <span class="lm-ch-label" :class="`lm-ch-${ch}`">{{ ch.toUpperCase() }}</span>
                  <input type="range" min="0" max="2" step="0.05"
                         :value="channelOf(item.key, ch)"
                         :aria-label="`${ch.toUpperCase()} channel of ${item.name}`"
                         @input="$emit('channel', item.key, ch, Number($event.target.value))" />
                  <span class="lm-ch-val">{{ Math.round(channelOf(item.key, ch) * 100) }}%</span>
                  <button v-if="channelOf(item.key, ch) !== 1" class="lm-ch-reset"
                          title="Reset to 100%"
                          @click="$emit('channel', item.key, ch, 1)">↺</button>
                </label>
              </div>
            </div>
          </div>
        </li>
      </ul>
    </section>

    <!-- Browse / layer selector tab -->
    <section v-show="activeTab === 'browse'" class="lm-selection lm-tab-panel" role="tabpanel">
      <div class="lm-search">
        <input v-model="query" type="search" placeholder="Search layers"
               aria-label="Search layers" />
      </div>

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
        Nothing matches "{{ query }}".
      </p>

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
              <em v-if="o.tier && o.tier !== 'free'" class="lm-tier">{{ o.tier }}</em>
              <small v-if="o.note" class="lm-note">{{ o.note }}</small>
            </span>
          </label>
        </div>
      </section>
    </div>
  </section>

    <section v-if="$slots.heatmaps" class="lm-sec">
      <button type="button" class="lm-sec-btn" :aria-expanded="secOpen.heatmaps" @click="toggleSec('heatmaps')">
        <span class="lm-caret" :class="{ open: secOpen.heatmaps }" aria-hidden="true">▸</span>
        <span class="lm-sec-title">Heatmaps</span><em class="lm-sec-meta">{{ heatmapName }}</em>
      </button>
      <div v-show="secOpen.heatmaps" class="lm-sec-body"><slot name="heatmaps" /></div>
    </section>

    <section v-if="$slots.access" class="lm-sec">
      <button type="button" class="lm-sec-btn" :aria-expanded="secOpen.access" @click="toggleSec('access')">
        <span class="lm-caret" :class="{ open: secOpen.access }" aria-hidden="true">▸</span>
        <span class="lm-sec-title">Access</span><em class="lm-sec-meta">{{ accessName }}</em>
      </button>
      <div v-show="secOpen.access" class="lm-sec-body"><slot name="access" /></div>
    </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'
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
  // Per-layer blend overrides, key → mode.
  blend: { type: Object, default: () => ({}) },
  // Per-layer channel multipliers, key → {r,g,b,a}. 1 = unchanged.
  channel: { type: Object, default: () => ({}) },
  // The stack default, for naming the inherit option — a dropdown whose first
  // entry says "Default" and nothing else makes you go and look it up.
  stackBlend: { type: String, default: 'normal' },
  // The one layer drawn on its own, or '' for all of them.
  solo: { type: String, default: '' },
  // EE layers currently fetching their tile template: Map<key, name>
  eeLoading: { type: Map, default: () => new Map() },
  // Docked against the controls on a wide screen; a bottom sheet on a phone.
  docked: { type: Boolean, default: true },
  // Phone layout: a bottom sheet with a drag handle and collapsible groups.
  compact: { type: Boolean, default: false },
  // Non-catalogue things that are on (heatmap, access): [{ key, label }], shown
  // as chips next to the drawn layers. Removing one emits 'extra-off'.
  extraChips: { type: Array, default: () => [] },
  basemapName: { type: String, default: '' },
  heatmapName: { type: String, default: '' },
  accessName: { type: String, default: '' },
})

const emit = defineEmits(['toggle', 'opacity', 'move', 'blend', 'channel', 'solo', 'clear', 'close', 'reset-channels', 'extra-off'])

// ── Sheet behaviour (phone) ────────────────────────────────────────────────
// Peek keeps only the handle, header and chips, so the sheet never has to
// cover the map to stay useful. Swipe down once to peek, again to dismiss.
const peek = ref(false)
const drag = reactive({ active: false, y0: 0, dy: 0 })
const sheetStyle = computed(() => (drag.active && drag.dy > 0 ? { transform: `translateY(${drag.dy}px)` } : null))
function dragStart(e) {
  drag.active = true; drag.y0 = e.clientY; drag.dy = 0
  e.currentTarget.setPointerCapture?.(e.pointerId)
}
function dragMove(e) { if (drag.active) drag.dy = e.clientY - drag.y0 }
function dragEnd(e) {
  if (!drag.active) return
  const dy = e.clientY - drag.y0
  drag.active = false; drag.dy = 0
  if (Math.abs(dy) < 6) peek.value = !peek.value // a tap
  else if (dy > 120 || (dy > 50 && peek.value)) emit('close')
  else if (dy > 50) peek.value = true
  else if (dy < -50) peek.value = false
}
const secOpen = reactive({ basemap: false, heatmaps: false, access: false })
const toggleSec = (k) => { secOpen[k] = !secOpen[k] }
// A feature that is on is worth showing open the first time.
watch(() => props.accessName, (v, o) => { if (v && !o) secOpen.access = true })

function onKey(e) { if (e.key === 'Escape' && props.open) emit('close') }
onMounted(() => window.addEventListener('keydown', onKey))
onBeforeUnmount(() => window.removeEventListener('keydown', onKey))
watch(() => props.open, (v) => { if (v) peek.value = false })

const query = ref('')
const win = ref(null)

// Tab state: 'active' shows drawn stack, 'browse' shows layer catalogue
const activeTab = ref('browse')
onMounted(() => {
  if (import.meta.client) {
    try {
      const saved = localStorage.getItem('layer-manager-tab')
      if (saved === 'active' || saved === 'browse') activeTab.value = saved
    } catch {
      // Preference reads silently fall back to defaults.
    }
  }
})
watch(activeTab, (val) => {
  if (import.meta.client) {
    try { localStorage.setItem('layer-manager-tab', val) } catch { /* blocked */ }
  }
})

// Auto-switch to Active tab when first layer is added
watch(() => props.order.length, (len, prev) => {
  if (len > 0 && prev === 0) activeTab.value = 'active'
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
const channelOf = (key, ch) => props.channel[key]?.[ch] ?? 1

// The default is only the default when there is a stack, so the label says so
// rather than promising a mode a single layer will not draw with.
const inheritLabel = computed(() => (props.stackBlend === 'normal'
  ? 'Default (normal)'
  : `Default (${blendLabel(props.stackBlend).toLowerCase()} when stacked)`))

// Check if any layer has channel overrides (any channel != 1)
const hasChannelOverrides = computed(() => {
  for (const [, channels] of Object.entries(props.channel)) {
    if (channels && Object.values(channels).some((v) => v !== 1)) return true
  }
  return false
})

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

const chips = computed(() => [
  ...activeList.value.map((o) => ({ key: o.key, label: o.name, extra: false })),
  ...props.extraChips.map((c) => ({ ...c, extra: true })),
])

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
  top: 8px; left: 8px; width: 310px;
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
/* Tab bar */
.lm-tabs {
  display: flex; flex: 0 0 auto;
  border-bottom: 1px solid var(--border-soft, #eee);
}
.lm-tab {
  flex: 1 1 0; border: 0; background: transparent;
  color: var(--muted, #777); font: inherit; font-size: 0.78rem; font-weight: 600;
  cursor: pointer; padding: 7px 10px; display: flex; align-items: center; justify-content: center; gap: 5px;
  border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.lm-tab:hover { color: var(--text); background: var(--surface-2, #f4f4f4); }
.lm-tab.on { color: var(--accent, #2b7a3d); border-bottom-color: var(--accent, #2b7a3d); background: transparent; }
.lm-tab-badge {
  background: var(--accent, #2b7a3d); color: #fff;
  border-radius: 999px; padding: 1px 6px; font-size: 0.64rem; font-weight: 700;
}

/* Tab panels — each fills the remaining height and scrolls internally */
.lm-tab-panel {
  flex: 1 1 auto; min-height: 0;
  display: flex; flex-direction: column;
}

.lm-active {
  overflow-y: auto; overscroll-behavior: contain;
  padding: 10px 12px 12px;
  background: var(--surface-2, #f7f7f7);
}
.lm-empty-active {
  color: var(--muted, #777); font-size: 0.78rem; text-align: center;
  padding: 24px 12px;
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
.lm-channels { margin-top: 8px; }
.lm-ch-head {
  display: block; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted, #777); margin-bottom: 4px;
}
.lm-ch-row {
  display: flex; align-items: center; gap: 5px; margin: 3px 0; cursor: pointer;
}
.lm-ch-label {
  flex: 0 0 14px; font-size: 0.7rem; font-weight: 700; text-align: center;
}
.lm-ch-r { color: #e05; }
.lm-ch-g { color: #1a9; }
.lm-ch-b { color: #39f; }
.lm-ch-a { color: var(--muted, #777); }
.lm-ch-row input[type=range] { flex: 1 1 auto; min-width: 0; accent-color: var(--accent); }
.lm-ch-val { flex: 0 0 3.2ch; font-size: 0.68rem; text-align: right; color: var(--muted, #777); }
.lm-ch-reset {
  border: 0; background: transparent; cursor: pointer; font-size: 0.75rem;
  color: var(--muted, #777); padding: 0 2px; line-height: 1;
}
.lm-ch-reset:hover { color: var(--text); }

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

/* Browse tab panel */
.lm-selection {
  overflow: hidden;
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

/* Desktop: the scroll wrapper is transparent to layout, so the panel keeps
   its original flex column. */
.lm-scroll { display: contents; }
.lm-chips {
  list-style: none; margin: 0; padding: 6px 12px; display: flex; gap: 6px; flex: 0 0 auto;
  overflow-x: auto; overscroll-behavior-x: contain; scrollbar-width: none;
  border-bottom: 1px solid var(--border-soft, #eee);
}
.lm-chips::-webkit-scrollbar { display: none; }
.lm-chip {
  display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; cursor: pointer;
  border: 1px solid var(--accent, #2b7a3d); background: var(--surface-2, #f4f4f4); color: var(--text, #222);
  border-radius: 999px; padding: 3px 6px 3px 10px; font: inherit; font-size: 0.72rem;
}
.lm-chip-name { max-width: 16ch; overflow: hidden; text-overflow: ellipsis; }
.lm-chip-x { color: var(--muted, #777); font-size: 1rem; line-height: 1; }
.lm-chip:hover .lm-chip-x { color: var(--text); }

.lm-sec { border-bottom: 1px solid var(--border-soft, #eee); flex: 0 0 auto; }
.lm-sec-btn {
  display: flex; align-items: center; gap: 7px; width: 100%; text-align: left; cursor: pointer;
  border: 0; background: transparent; color: var(--text, #222); font: inherit; padding: 8px 12px;
}
.lm-sec-title {
  font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 700;
  color: var(--muted, #777);
}
.lm-sec-meta {
  font-style: normal; color: var(--muted, #777); font-size: 0.74rem; margin-left: auto;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0;
}
.lm-sec-body { padding: 2px 12px 12px; }
.lm-sec-h { display: none; }
.lm-grab { display: none; }

/* Phone: a bottom sheet. The map's top half stays visible for judging what a
   layer did; peeking collapses it to handle + header + chips; a tap outside,
   swipe down, Escape or the close button dismisses it. */
.lm-backdrop { position: absolute; inset: 0; z-index: 1190; background: transparent; }
.lm.compact {
  top: auto; left: 0; right: 0; bottom: 0; width: 100%; max-width: 100%; box-sizing: border-box;
  border-radius: 16px 16px 0 0; border-bottom: none;
  max-height: min(78%, calc(100% - 64px));
  padding-bottom: env(safe-area-inset-bottom, 0px);
  transition: transform 0.18s ease;
}
.lm.compact.dragging { transition: none; }
.lm.compact .lm-scroll {
  display: block; flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;
}
.lm.compact .lm-grab {
  display: flex; justify-content: center; align-items: center; flex: 0 0 auto;
  height: 28px; cursor: grab; touch-action: none;
}
.lm-grab:focus-visible { outline: 2px solid var(--accent, #2b7a3d); outline-offset: -2px; }
.lm-grab-bar { width: 44px; height: 5px; border-radius: 3px; background: var(--border, #ccc); }
.lm.compact .lm-head { padding: 0 6px 4px 14px; border-bottom: 0; }
.lm.compact .lm-head-acts { flex-wrap: wrap; justify-content: flex-end; }
.lm.compact .lm-text-btn, .lm.compact .lm-close {
  min-height: 44px; min-width: 44px; font-size: 0.82rem; padding: 0 10px;
}
.lm.compact .lm-close { font-size: 1.6rem; }
.lm.compact .lm-chips { padding: 4px 14px 8px; }
.lm.compact .lm-chip { min-height: 44px; padding: 0 8px 0 14px; font-size: 0.82rem; }
.lm.compact .lm-chip-x { font-size: 1.3rem; min-width: 24px; text-align: center; }
.lm.compact .lm-sec-btn { min-height: 48px; padding: 0 14px; }
.lm.compact .lm-sec-title { font-size: 0.74rem; }
.lm.compact .lm-sec-body { padding: 4px 14px 14px; }
.lm.compact .lm-sec-h {
  display: block; margin: 0; padding: 10px 14px 0; font-size: 0.74rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted, #777);
}
.lm.compact .lm-tab-panel { display: block; min-height: 0; }
.lm.compact .lm-active, .lm.compact .lm-body { overflow: visible; max-height: none; padding-left: 14px; padding-right: 14px; }
.lm.compact .lm-search { padding: 10px 14px 0; }
.lm.compact .lm-search input { min-height: 44px; font-size: 1rem; padding: 0 12px; }
.lm.compact .lm-groupby { padding: 8px 14px 0; }
.lm.compact .lm-seg-btn, .lm.compact .lm-tab { min-height: 44px; font-size: 0.82rem; }
.lm.compact .lm-group-head { min-height: 48px; margin: 0; padding: 0 4px; }
.lm.compact .lm-group-label { font-size: 0.74rem; }
.lm.compact .lm-row { min-height: 44px; align-items: center; margin: 0; padding: 4px; }
.lm.compact .lm-row input { width: 22px; height: 22px; margin: 0; }
.lm.compact .lm-row-name { font-size: 0.9rem; }
.lm.compact .lm-advanced-toggle { min-height: 44px; padding: 0 8px; font-size: 0.8rem; }

/* A drawn layer: name row, order row, then a full-width thumb-sized opacity slider. */
.lm.compact .lm-stack { gap: 14px; }
.lm.compact .lm-on { display: grid; gap: 4px; }
.lm.compact .lm-on-top { display: flex; flex-wrap: wrap; gap: 4px 6px; align-items: center; }
.lm.compact .lm-swatch {
  width: 44px; height: 44px; background: transparent; border: 0; border-radius: 8px;
}
.lm.compact .lm-tick {
  width: 26px; height: 26px; border-radius: 6px; background: var(--accent, #2b7a3d);
  display: inline-flex; align-items: center; justify-content: center; font-size: 0.85rem;
}
.lm.compact .lm-on-name { font-size: 0.92rem; }
.lm.compact .lm-solo { width: 44px; height: 44px; font-size: 0.85rem; border-radius: 8px; }
.lm.compact .lm-order { flex: 1 0 100%; gap: 6px; }
.lm.compact .lm-order button { flex: 1 1 0; width: auto; height: 44px; font-size: 0.9rem; border-radius: 8px; }
.lm.compact .lm-op { padding-left: 0; gap: 10px; }
.lm.compact .lm-op-label { width: 4.5ch; font-size: 0.85rem; }
.lm.compact .lm-blend-wrapper { padding-left: 0; }
.lm.compact .lm-blend select, .lm.compact .lm-ch-row input[type=range] { min-height: 44px; font-size: 0.9rem; }
.lm.compact .lm-ch-row { min-height: 44px; }
.lm.compact .lm-ch-reset { min-width: 44px; min-height: 44px; font-size: 1rem; }
/* Thumb-sized range controls. */
.lm.compact input[type=range] { -webkit-appearance: none; appearance: none; background: transparent; height: 44px; margin: 0; touch-action: pan-y; }
.lm.compact input[type=range]::-webkit-slider-runnable-track { height: 6px; border-radius: 3px; background: var(--border, #ccc); }
.lm.compact input[type=range]::-webkit-slider-thumb {
  -webkit-appearance: none; width: 28px; height: 28px; margin-top: -11px; border-radius: 50%;
  background: var(--accent, #2b7a3d); border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
}
.lm.compact input[type=range]::-moz-range-track { height: 6px; border-radius: 3px; background: var(--border, #ccc); }
.lm.compact input[type=range]::-moz-range-thumb {
  width: 24px; height: 24px; border-radius: 50%; background: var(--accent, #2b7a3d);
  border: 2px solid #fff; box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
}
</style>
