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
    body: 'Each dot is a species record enriched with environmental data. '
      + 'A short tour of the controls—skip any time.',
  },
  {
    icon: '🧭',
    title: 'Environmental layers',
    body: 'Open Layers to overlay fire history, forest type, soil, terrain analysis, '
      + 'weather, and satellite vegetation indices. Stack several at once.',
  },
  {
    icon: '🎨',
    title: 'Colour and size the points',
    body: 'Under Points, colour observations by any field—species, elevation, season—'
      + 'and size them by a second. The legend updates to match.',
  },
  {
    icon: '🔥',
    title: 'Aggregate into heatmaps',
    body: 'Switch to Heatmap to bin records into hexes or squares, from ~100 m to ~28 km. '
      + 'Show density or the mean of any environmental field per cell.',
  },
  {
    icon: '📍',
    title: 'Drop a pin',
    body: 'Click anywhere to read coordinates, elevation, the heatmap cell value, and '
      + 'the nearest observation. Drag to move the pin.',
  },
  {
    icon: '🔎',
    title: 'Filters apply everywhere',
    body: 'Set a taxon, date range, season, or region under Filters and every view—'
      + 'map, charts, statistics—updates to match.',
  },
  {
    icon: '💾',
    title: 'Offline and sharing',
    body: 'Cache a map area for use in the field without signal. Share any view as a '
      + 'URL that restores your exact filters, layers, and position.',
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
