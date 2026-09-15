<template>
  <div v-if="open" ref="win" class="lm" :class="{ docked }" role="dialog" aria-label="Layer manager">
    <header class="lm-head">
      <div class="lm-title">
        <strong>Layers</strong>
        <span v-if="activeList.length" class="lm-count">{{ activeList.length }} on</span>
      </div>
      <div class="lm-head-acts">
        <button v-if="activeList.length" class="lm-text-btn" title="Switch every overlay off"
                @click="$emit('clear')">Clear</button>
        <button class="lm-close" aria-label="Close the layer manager" @click="$emit('close')">×</button>
      </div>
    </header>

    <!-- The active stack, first and separately.
         An overlay list is read in two completely different ways: "what can I
         add" is a browse, "what is on and in what order" is a glance. The old
         single dropdown answered only the first, and the second had to be
         reconstructed by scanning forty checkboxes for ticks. -->
    <section v-if="activeList.length" class="lm-active">
      <div class="lm-sec-head">
        <span>Drawn, top first</span>
        <HelpLink option="map-layer-order" />
      </div>
      <ul class="lm-stack">
        <li v-for="(item, i) in activeList" :key="item.key" class="lm-on">
          <div class="lm-on-top">
            <button class="lm-swatch" :title="`Hide ${item.name}`" @click="$emit('toggle', item.key)">
              <span class="lm-tick" aria-hidden="true">✓</span>
            </button>
            <span class="lm-on-name" :title="item.name">{{ item.name }}</span>
            <!-- Stacking, because overlays hide each other: land ownership under
                 a hillshade is a different map from the same two the other way
                 up, and there is no other way to say which you meant. -->
            <span class="lm-order">
              <button :disabled="i === 0" title="Move up" @click="$emit('move', item.key, -1)">▲</button>
              <button :disabled="i === activeList.length - 1" title="Move down"
                      @click="$emit('move', item.key, 1)">▼</button>
            </span>
          </div>
          <label class="lm-op">
            <span class="lm-op-label">{{ Math.round(opacityOf(item.key) * 100) }}%</span>
            <input type="range" min="0.05" max="1" step="0.05" :value="opacityOf(item.key)"
                   :aria-label="`Opacity of ${item.name}`"
                   @input="$emit('opacity', item.key, Number($event.target.value))" />
          </label>
        </li>
      </ul>
    </section>

    <div class="lm-search">
      <input v-model="query" type="search" placeholder="Search layers"
             aria-label="Search layers" />
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
              <!-- Listed but marked, rather than hidden: knowing FRMS
                   computes it is part of what membership is for. -->
              <em v-if="o.tier && o.tier !== 'free'" class="lm-tier">{{ o.tier }}</em>
              <small v-if="o.note" class="lm-note">{{ o.note }}</small>
            </span>
          </label>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { filterLayerGroups } from '~/composables/mapLayers'

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
  // Docked against the controls on a wide screen; a bottom sheet on a phone.
  docked: { type: Boolean, default: true },
})

defineEmits(['toggle', 'opacity', 'move', 'clear', 'close'])

const query = ref('')
const win = ref(null)

const opacityOf = (key) => props.opacity[key] ?? 1

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
  const withActive = props.groups
    .filter((g) => g.items.some((o) => props.active.has(o.key)))
    .map((g) => g.label)
  openPanels.value = new Set(withActive.length ? withActive : props.groups.map((g) => g.label))
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
const filtered = computed(() => filterLayerGroups(props.groups, query.value))

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

.lm-active {
  flex: 0 0 auto; padding: 9px 12px; border-bottom: 1px solid var(--border-soft, #eee);
  background: var(--surface-2, #f7f7f7);
  max-height: 40%; overflow-y: auto;
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

.lm-search { flex: 0 0 auto; padding: 9px 12px 0; }
.lm-search input {
  width: 100%; box-sizing: border-box;
  background: var(--surface-2, #f4f4f4); color: var(--text, #222);
  border: 1px solid var(--border, #ddd); border-radius: 6px;
  padding: 5px 8px; font: inherit; font-size: 0.78rem;
}
.lm-search input:focus { border-color: var(--accent, #2b7a3d); outline: none; }

.lm-body { flex: 1 1 auto; overflow-y: auto; overscroll-behavior: contain; padding: 10px 12px 12px; }
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

/* On a phone a floating window over a map is most of the map. A sheet from the
   bottom leaves the top half visible, which is where you look to judge whether
   the layer you just ticked did anything. */
@media (max-width: 720px) {
  .lm {
    top: auto; right: 6px; left: 6px; bottom: 6px; width: auto;
    max-height: 62%;
  }
  .lm-active { max-height: 35%; }
}
</style>
