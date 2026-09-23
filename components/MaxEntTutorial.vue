<template>
  <div v-if="open" class="tut" role="dialog" aria-modal="true" :aria-label="`MaxEnt tutorial, step ${i + 1} of ${STEPS.length}`">
    <div class="tut-backdrop" @click="dismiss"></div>
    <div class="tut-card">
      <button class="tut-x" aria-label="Close tutorial" @click="dismiss">×</button>

      <div class="tut-progress" aria-hidden="true">
        <div class="tut-bar" :style="{ width: (100 * (i + 1) / STEPS.length) + '%' }"></div>
      </div>

      <div class="tut-icon" aria-hidden="true">{{ step.icon }}</div>
      <div class="tut-step">Step {{ i + 1 }} of {{ STEPS.length }}</div>
      <h2 class="tut-title">{{ step.title }}</h2>
      <p class="tut-body" v-html="step.body"></p>

      <div v-if="step.tip" class="tut-tip">
        <span class="tut-tip-label">Tip</span>
        {{ step.tip }}
      </div>

      <div class="tut-dots" aria-hidden="true">
        <span
          v-for="(s, n) in STEPS"
          :key="n"
          class="tut-dot"
          :class="{ on: n === i, done: n < i }"
          @click="i = n"
        ></span>
      </div>

      <div class="tut-acts">
        <button v-if="i > 0" class="tut-btn ghost" @click="i -= 1">Back</button>
        <button class="tut-btn ghost skip" @click="dismiss">{{ i === STEPS.length - 1 ? 'Close' : 'Skip' }}</button>
        <button v-if="i < STEPS.length - 1" class="tut-btn primary" @click="i += 1">Next →</button>
        <button v-else class="tut-btn primary" @click="dismiss">Start modeling</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'

// Bump the version string to re-show the tutorial to all users after major changes.
const SEEN_KEY = 'nexstrata-maxent-tutorial-v1'

const STEPS = [
  {
    icon: '🧬',
    title: 'What is MaxEnt?',
    body: 'MaxEnt is a <strong>species distribution model</strong>. It learns which environmental conditions '
        + '(elevation, temperature, rainfall, vegetation) tend to occur where your species was recorded, '
        + 'then predicts a <strong>habitat suitability score</strong> for every pixel on the map.',
    tip: 'Suitability is not probability of occurrence — it measures environmental similarity to observed locations.',
  },
  {
    icon: '📂',
    title: 'Step 1 — Pick a dataset',
    body: 'Choose a <strong>source dataset</strong> from the dropdown. Each dataset is a collection of '
        + 'georeferenced observations you have uploaded or created. The more observations (and the better '
        + 'they cover the species\' range) the more reliable your model will be.',
    tip: 'Aim for at least 30 records. Fewer than 10 produces unreliable results.',
  },
  {
    icon: '🌿',
    title: 'Step 2 — Choose predictors',
    body: 'Predictors are the <strong>environmental variables</strong> the model will use. Tick the ones '
        + 'with a plausible ecological link to your species: '
        + '<em>elevation</em> for altitude-restricted taxa, <em>temperature</em> for climate-sensitive ones, '
        + '<em>NDVI</em> for habitat-dependent species. More is not always better — correlated predictors '
        + 'can inflate performance without adding real information.',
    tip: 'Terrain + NDVI + one climate variable is a solid baseline for most fungi.',
  },
  {
    icon: '🎲',
    title: 'Step 3 — Set background points',
    body: 'MaxEnt needs <strong>background points</strong> — random locations drawn from across the landscape — '
        + 'to contrast against your presences. More points (up to ~10,000) produce a more stable background '
        + 'estimate, but training takes longer. <strong>1,000</strong> is a good starting point.',
    tip: 'If your observations cluster in a small area, reducing background points can help the model focus on that region.',
  },
  {
    icon: '▶️',
    title: 'Step 4 — Queue the model',
    body: 'Give your model a descriptive name, then click <strong>Queue Model</strong>. Training runs on '
        + 'Google Earth Engine and typically takes <strong>2–5 minutes</strong>. When it finishes, your '
        + 'model appears in the list below with an AUC score and a <em>View on Map</em> button.',
    tip: 'You can leave the page and come back — the job runs in the background.',
  },
  {
    icon: '📊',
    title: 'Step 5 — Read the results',
    body: 'Once the model finishes, check the <strong>AUC score</strong> (above 0.75 is generally useful), '
        + 'inspect the <strong>response curves</strong> to see which conditions the model favors, and look '
        + 'at the <strong>variable contributions</strong> to understand which predictors mattered most. '
        + 'Then click <em>View on Map</em> to overlay the suitability surface on the map.',
    tip: 'A suspiciously perfect AUC (> 0.98) often means the training data and background overlap heavily — check for spatial autocorrelation.',
  },
]

const open = ref(false)
const i = ref(0)
const step = computed(() => STEPS[i.value])

function start() { i.value = 0; open.value = true }

function dismiss() {
  open.value = false
  try { localStorage.setItem(SEEN_KEY, '1') } catch { /* private mode */ }
}

function seen() {
  try { return localStorage.getItem(SEEN_KEY) === '1' } catch { return false }
}

function onReplay() { start() }

onMounted(() => {
  if (!seen()) start()
  window.addEventListener('maxent-tutorial-open', onReplay)
})
onBeforeUnmount(() => window.removeEventListener('maxent-tutorial-open', onReplay))

defineExpose({ start })
</script>

<style scoped>
.tut { position: fixed; inset: 0; z-index: 3000; display: grid; place-items: center; }
.tut-backdrop {
  position: absolute; inset: 0;
  background: rgba(4, 10, 16, 0.60);
  backdrop-filter: blur(2px);
}
.tut-card {
  position: relative; z-index: 1;
  width: min(460px, calc(100vw - 32px));
  background: var(--surface, #fff); color: var(--text, #1a1a1a);
  border: 1px solid var(--border, #ddd); border-radius: 16px;
  box-shadow: 0 24px 70px rgba(0,0,0,0.45);
  padding: 0 22px 20px; text-align: center;
  overflow: hidden;
}
.tut-x {
  position: absolute; top: 10px; right: 14px; border: 0; background: transparent;
  color: var(--muted, #888); font-size: 1.4rem; line-height: 1; cursor: pointer; z-index: 2;
}
.tut-x:hover { color: var(--text, #222); }

.tut-progress {
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: var(--border, #e5e5e5);
}
.tut-bar {
  height: 100%; background: #34d399;
  transition: width 0.25s ease;
}

.tut-icon {
  font-size: 2.5rem; line-height: 1; margin: 22px auto 10px;
  filter: drop-shadow(0 2px 8px rgba(52,211,153,0.35));
}
.tut-step {
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.65rem; letter-spacing: 0.15em; text-transform: uppercase;
  color: #34d399; font-weight: 700; margin-bottom: 6px;
}
.tut-title { margin: 0 0 10px; font-size: 1.2rem; color: var(--text-strong, #111); }
.tut-body {
  margin: 0 0 14px; font-size: 0.9rem; line-height: 1.6;
  color: var(--muted, #555); text-align: left;
}

.tut-tip {
  text-align: left; font-size: 0.82rem; line-height: 1.5;
  background: color-mix(in srgb, #34d399 10%, transparent);
  border-left: 3px solid #34d399;
  border-radius: 0 6px 6px 0; padding: 8px 10px; margin-bottom: 14px;
  color: var(--text, #333);
}
.tut-tip-label {
  display: inline-block; font-size: 0.65rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; color: #2b7a3d; margin-right: 6px;
}

.tut-dots { display: flex; justify-content: center; gap: 7px; margin-bottom: 16px; }
.tut-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--border, #ccc); cursor: pointer;
  transition: background 0.12s ease, transform 0.12s ease;
}
.tut-dot.done { background: color-mix(in srgb, #34d399 50%, #ccc); }
.tut-dot.on { background: #34d399; transform: scale(1.35); }

.tut-acts { display: flex; gap: 8px; justify-content: center; }
.tut-btn {
  border: 1px solid var(--border, #ddd); background: var(--surface, #fff); color: var(--text, #222);
  border-radius: 8px; padding: 8px 18px; font: inherit; font-size: 0.9rem; font-weight: 600; cursor: pointer;
}
.tut-btn.ghost:hover { background: var(--surface-2, #f2f2f2); }
.tut-btn.skip { color: var(--muted, #777); }
.tut-btn.primary {
  background: linear-gradient(180deg, #34d399, #2b7a3d);
  border-color: #2b7a3d; color: #04140b;
  box-shadow: 0 6px 18px rgba(52,211,153,0.28);
}
.tut-btn.primary:hover { filter: brightness(1.05); }

@media (max-width: 480px) {
  .tut { place-items: end center; }
  .tut-card { margin-bottom: 12px; border-radius: 16px 16px 12px 12px; }
}
</style>
