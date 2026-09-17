<template>
  <div v-if="open" class="tour" role="dialog" aria-modal="true" :aria-label="`Map tour, step ${i + 1} of ${STEPS.length}`">
    <div class="tour-backdrop" @click="dismiss"></div>
    <div class="tour-card">
      <button class="tour-x" aria-label="Close the tour" @click="dismiss">×</button>

      <div class="tour-icon" aria-hidden="true">{{ step.icon }}</div>
      <div class="tour-step">Step {{ i + 1 }} of {{ STEPS.length }}</div>
      <h2 class="tour-title">{{ step.title }}</h2>
      <p class="tour-body">{{ step.body }}</p>

      <div class="tour-dots" aria-hidden="true">
        <span v-for="(s, n) in STEPS" :key="n" class="tour-dot" :class="{ on: n === i }"
              @click="i = n"></span>
      </div>

      <div class="tour-acts">
        <button v-if="i > 0" class="tour-btn ghost" @click="i -= 1">Back</button>
        <button class="tour-btn ghost skip" @click="dismiss">{{ i === STEPS.length - 1 ? 'Close' : 'Skip' }}</button>
        <button v-if="i < STEPS.length - 1" class="tour-btn primary" @click="i += 1">Next</button>
        <button v-else class="tour-btn primary" @click="dismiss">Explore the map</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

// A first-run tour of the map, shown once and then never again unless replayed.
//
// Deliberately not anchored to individual controls: those live behind popovers
// on a phone and move between breakpoints, so a tooltip pinned to one is fragile
// and often points at nothing. A centred card that names each feature and where
// to find it survives every layout, which is what a first-run tour has to do.

// Bump the version to re-show the tour to everyone after a big change.
const SEEN_KEY = 'nexstrata-map-tour-v1'

const STEPS = [
  {
    icon: '🗺️',
    title: 'Welcome to the map',
    body: 'Every dot is a real observation, enriched with the environment around it. '
      + 'Here is a two-minute tour of what you can do; skip any time.',
  },
  {
    icon: '🧭',
    title: 'Stack up layers',
    body: 'The Layers button opens overlays for fire history, forest and soil, terrain '
      + '(slope, aspect, wetness), weather, and satellite greenness and moisture. Group '
      + 'them by subject, source or type, and stack several at once.',
  },
  {
    icon: '🎨',
    title: 'Colour and size the points',
    body: 'Under Points, colour the observations by any field (species, season, elevation) '
      + 'and size them by a second. The legend explains whatever you pick.',
  },
  {
    icon: '🔥',
    title: 'Aggregate into heatmaps',
    body: 'Switch from points to a heatmap to bin the finds into hexes or squares, from a '
      + 'hundred metres to tens of kilometres, and read density or an averaged field per cell.',
  },
  {
    icon: '📍',
    title: 'Drop a pin anywhere',
    body: 'Click the map to drop a point and read its coordinates, plus code and ground '
      + 'elevation, the heatmap cell under it, and the nearest recorded find. Drag to fine-tune.',
  },
  {
    icon: '🔎',
    title: 'Filter and search',
    body: 'Narrow to a species, a season, a date range or a region, and every view, the map, '
      + 'the charts, the stats, follows the same filters.',
  },
  {
    icon: '💾',
    title: 'Save offline and share',
    body: 'Cache an area with its layers for the field where there is no signal, and share any '
      + 'view as a link or an embed that restores exactly what you are looking at.',
  },
]

const open = ref(false)
const i = ref(0)

function start() { i.value = 0; open.value = true }

function dismiss() {
  open.value = false
  try { localStorage.setItem(SEEN_KEY, '1') } catch { /* private mode; just close */ }
}

function seen() {
  try { return localStorage.getItem(SEEN_KEY) === '1' } catch { return false }
}

function onReplay() { start() }

const step = computed(() => STEPS[i.value])

onMounted(() => {
  if (!seen()) start()
  // Lets a menu item or shortcut replay the tour: dispatch 'map-tour-open'.
  window.addEventListener('map-tour-open', onReplay)
})
onBeforeUnmount(() => window.removeEventListener('map-tour-open', onReplay))

defineExpose({ start })
</script>

<style scoped>
.tour { position: absolute; inset: 0; z-index: 3000; display: grid; place-items: center; }
.tour-backdrop {
  position: absolute; inset: 0; background: rgba(4, 10, 16, 0.55);
  backdrop-filter: blur(2px);
}
.tour-card {
  position: relative; z-index: 1; width: min(420px, calc(100vw - 32px));
  background: var(--surface, #fff); color: var(--text, #1a1a1a);
  border: 1px solid var(--border, #ddd); border-radius: 16px;
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.4);
  padding: 22px 22px 18px; text-align: center;
}
.tour-x {
  position: absolute; top: 10px; right: 12px; border: 0; background: transparent;
  color: var(--muted, #888); font-size: 1.4rem; line-height: 1; cursor: pointer;
}
.tour-x:hover { color: var(--text, #222); }
.tour-icon {
  font-size: 2.4rem; line-height: 1; margin: 4px auto 10px;
  filter: drop-shadow(0 2px 8px rgba(56, 189, 248, 0.35));
}
.tour-step {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: #38bdf8; font-weight: 700; margin-bottom: 6px;
}
.tour-title { margin: 0 0 8px; font-size: 1.2rem; color: var(--text-strong, #111); }
.tour-body { margin: 0 0 16px; font-size: 0.92rem; line-height: 1.55; color: var(--muted, #555); }

.tour-dots { display: flex; justify-content: center; gap: 7px; margin-bottom: 16px; }
.tour-dot {
  width: 7px; height: 7px; border-radius: 50%; background: var(--border, #ccc);
  cursor: pointer; transition: background 0.12s ease, transform 0.12s ease;
}
.tour-dot.on { background: #38bdf8; transform: scale(1.35); }

.tour-acts { display: flex; gap: 8px; justify-content: center; }
.tour-btn {
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff); color: var(--text, #222);
  border-radius: 8px; padding: 8px 16px; font: inherit; font-size: 0.9rem; font-weight: 600; cursor: pointer;
}
.tour-btn.ghost:hover { background: var(--surface-2, #f2f2f2); }
.tour-btn.skip { color: var(--muted, #777); }
.tour-btn.primary {
  background: linear-gradient(180deg, #34d399, #2b7a3d); border-color: #2b7a3d; color: #04140b;
  box-shadow: 0 6px 18px rgba(52, 211, 153, 0.28);
}
.tour-btn.primary:hover { filter: brightness(1.05); }

@media (max-width: 480px) {
  .tour { place-items: end center; }
  .tour-card { margin-bottom: 12px; border-radius: 16px 16px 12px 12px; }
}
</style>
