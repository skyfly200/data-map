<template>
  <div class="options">
    <h1>Options</h1>
    <p class="lede">
      Everything the app remembers about how you like it, in one place. These are
      the same settings the panels on each page expose — changed here or there,
      it is the same setting.
    </p>

    <section class="opt-group">
      <h2>Units</h2>
      <div class="opt-row">
        <span class="opt-name">Elevation</span>
        <div class="seg" role="group" aria-label="Elevation units">
          <button :class="{ on: unit === 'ft' }" @click="unit = 'ft'">Feet</button>
          <button :class="{ on: unit === 'm' }" @click="unit = 'm'">Metres</button>
        </div>
      </div>
      <div class="opt-row">
        <span class="opt-name">Temperature</span>
        <div class="seg" role="group" aria-label="Temperature units">
          <button :class="{ on: tempUnit === 'F' }" @click="tempUnit = 'F'">°F</button>
          <button :class="{ on: tempUnit === 'C' }" @click="tempUnit = 'C'">°C</button>
        </div>
      </div>
    </section>

    <section class="opt-group">
      <h2>Appearance</h2>
      <div class="opt-row">
        <span class="opt-name">Theme</span>
        <div class="seg" role="group" aria-label="Theme">
          <button :class="{ on: theme === 'dark' }" @click="setTheme('dark')">Dark</button>
          <button :class="{ on: theme === 'light' }" @click="setTheme('light')">Light</button>
        </div>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Palette
          <small>Shared by the map and every chart.</small>
        </span>
        <select v-model="paletteKey" @change="appearance.persist()">
          <option v-for="p in PALETTES" :key="p.key" :value="p.key">{{ p.label }}</option>
        </select>
      </div>
      <div class="opt-row">
        <span class="opt-name">Palette preview</span>
        <span class="swatches">
          <span v-for="c in activeColors" :key="c" class="sw" :style="{ background: c }" :title="c"></span>
        </span>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Point outline
          <small>The ring around each map dot. Helps when sparse, crowds when dense.</small>
        </span>
        <label class="chk">
          <input v-model="pointOutline" type="checkbox" @change="appearance.persist()" />
          Outline dots
        </label>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Heatmap colours
          <small>The ramp the map's grid heatmaps are shaded with.</small>
        </span>
        <select v-model="heatmapRampKey" @change="heatmaps.persist()">
          <option v-for="r in RAMP_PRESETS" :key="r.key" :value="r.key">{{ r.label }}</option>
          <option value="custom">Custom (set on the map)</option>
        </select>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Heatmap cells
          <small>The shape the observations are binned into.</small>
        </span>
        <select v-model="cellShape" @change="heatmaps.persist()">
          <option v-for="s in CELL_SHAPES" :key="s.value" :value="s.value">{{ s.label }}</option>
        </select>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Heatmap opacity
          <small>How strongly the grid cells cover the basemap under them.</small>
        </span>
        <span class="opt-slider">
          <input v-model.number="heatmapOpacity" type="range" min="0.05" max="1" step="0.05"
                 aria-label="Heatmap opacity" @change="heatmaps.persist()" />
          <strong>{{ Math.round(heatmapOpacity * 100) }}%</strong>
        </span>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Map layer opacity
          <small>How strongly the reference tile layers — hillshade, rainfall, land cover — cover the basemap.</small>
        </span>
        <span class="opt-slider">
          <input v-model.number="tileOpacity" type="range" min="0.05" max="1" step="0.05"
                 aria-label="Map layer opacity" @change="heatmaps.persist()" />
          <strong>{{ Math.round(tileOpacity * 100) }}%</strong>
        </span>
      </div>
      <div class="opt-actions">
        <button class="btn" @click="appearance.shuffleColors()">🎨 Shuffle colours</button>
        <button class="btn" @click="appearance.reset()">Reset appearance</button>
      </div>
    </section>

    <section class="opt-group">
      <h2>Offline</h2>
      <p class="opt-lede">
        Keep the app and its data in this browser so it opens with no signal — which is
        where a map of where things grow is most often read. Map tiles for a particular
        area are saved from the map's settings, where the app knows what is on screen.
      </p>
      <ClientOnly>
        <OfflineControls />
        <template #fallback><p class="opt-note">Loading…</p></template>
      </ClientOnly>
    </section>

    <section class="opt-group">
      <h2>Data</h2>
      <div class="opt-row">
        <span class="opt-name">
          Precise coordinates only
          <small>
            Drop records whose published location was obscured or poorly measured.
            Their terrain was sampled somewhere the mushroom probably was not.
          </small>
        </span>
        <label class="chk">
          <input type="checkbox" :checked="filters.preciseOnly"
                 @change="setFilter('preciseOnly', $event.target.checked)" />
          Precise only
        </label>
      </div>
      <div class="opt-row">
        <span class="opt-name">
          Active filters
          <small>Filters apply to every view at once.</small>
        </span>
        <span class="opt-value">
          {{ activeCount ? `${activeCount} active` : 'None' }}
          <NuxtLink to="/data" class="btn small">Manage on Data</NuxtLink>
        </span>
      </div>
    </section>

    <ClientOnly>
      <section v-if="configured" class="opt-group">
        <h2>Account</h2>
        <div class="opt-row">
          <span class="opt-name">Signed in as</span>
          <span class="opt-value">{{ user?.email || 'Not signed in' }}</span>
        </div>
        <div class="opt-row">
          <span class="opt-name">
            Sync
            <small>Settings and saved charts follow your account to another device.</small>
          </span>
          <SyncStatus />
        </div>
        <div class="opt-actions">
          <NuxtLink v-if="!isAuthed" to="/login" class="btn">Sign in</NuxtLink>
          <button v-else class="btn danger" @click="signOut">Sign out</button>
        </div>
      </section>
    </ClientOnly>

    <section class="opt-group">
      <h2>Keyboard</h2>
      <p class="opt-note">
        Press <kbd>?</kbd> anywhere for the full list. Shortcuts are ignored while
        you are typing in a field.
      </p>
      <button class="btn" @click="shortcuts.helpOpen.value = true">Show shortcuts</button>
    </section>

    <p class="foot">
      Every option here is documented in the
      <NuxtLink to="/guide#reference">option reference</NuxtLink>.
    </p>
  </div>
</template>

<script setup>
import { useUnits } from '~/composables/useUnits'
import { useAppearance } from '~/composables/useAppearance'

useHead({ title: 'Options · Nexstrata' })

const { unit, tempUnit } = useUnits()
const appearance = useAppearance()
const { PALETTES, paletteKey, activeColors, pointOutline } = appearance
const heatmaps = useMapHeatmaps()
const { RAMP_PRESETS, heatmapRampKey, heatmapOpacity, tileOpacity, cellShape, CELL_SHAPES } = heatmaps
const { filters, setFilter, activeCount } = useFilters()
const { user, isAuthed, configured, signOut } = useAuth()
const shortcuts = useShortcuts()

// Theme lives in app.vue's shared state; this page writes the same key so the
// two controls cannot disagree about which theme is on.
const theme = useState('theme', () => 'dark')
function setTheme(t) {
  theme.value = t
  if (import.meta.client) {
    document.documentElement.setAttribute('data-theme', t)
    localStorage.setItem('theme', t)
  }
}

onMounted(() => {
  appearance.loadFromStorage()
  heatmaps.loadFromStorage()
})
</script>

<style scoped>
.options { max-width: 720px; margin: 0 auto; padding: 28px 20px 64px; color: var(--text); }
h1 { margin: 0 0 8px; font-size: 1.6rem; }
.lede { color: var(--muted); line-height: 1.6; margin: 0 0 26px; }

.opt-group {
  border: 1px solid var(--border); border-radius: 10px; background: var(--surface);
  padding: 14px 16px; margin-bottom: 18px;
}
.opt-group h2 {
  margin: 0 0 12px; font-size: 0.78rem; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--muted);
}
.opt-row {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 16px;
  padding: 9px 0; border-top: 1px solid var(--border-soft, var(--border));
}
.opt-group .opt-row:first-of-type { border-top: 0; }
.opt-name { display: flex; flex-direction: column; gap: 3px; font-weight: 600; min-width: 0; }
.opt-lede { margin: 0 0 14px; color: var(--muted); font-size: 0.84rem; line-height: 1.55; }
.opt-name small { font-weight: 400; color: var(--muted); font-size: 0.76rem; line-height: 1.4; }
.opt-slider { display: inline-flex; align-items: center; gap: 8px; flex: 0 0 auto; }
.opt-slider input { width: 130px; }
.opt-slider strong { font-variant-numeric: tabular-nums; min-width: 3ch; text-align: right; }
.opt-value { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 0.85rem; flex: 0 0 auto; }

.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; flex: 0 0 auto; }
.seg button {
  border: 0; background: transparent; color: var(--muted); cursor: pointer;
  padding: 6px 14px; font-size: 0.82rem; font-weight: 600;
}
.seg button.on { background: var(--accent); color: var(--accent-ink); }

select {
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px;
  background: var(--surface); color: var(--text); font-size: 0.85rem; flex: 0 0 auto;
}
.chk { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; flex: 0 0 auto; }
.chk input { accent-color: var(--accent); }
.swatches { display: inline-flex; gap: 3px; flex-wrap: wrap; }
.sw { width: 16px; height: 16px; border-radius: 3px; }

.opt-actions { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.btn {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 6px; padding: 7px 12px; font-size: 0.82rem; font-weight: 600;
  cursor: pointer; text-decoration: none; display: inline-block;
}
.btn:hover { background: var(--surface-3); }
.btn.small { padding: 4px 9px; font-size: 0.76rem; }
.btn.danger { color: var(--danger); }

.opt-note { color: var(--muted); font-size: 0.85rem; line-height: 1.6; margin: 0 0 10px; }
kbd {
  display: inline-block; font: 600 0.75rem/1 ui-monospace, monospace;
  background: var(--surface-2); border: 1px solid var(--border);
  border-bottom-width: 2px; border-radius: 4px; padding: 3px 5px;
}
.foot { color: var(--muted); font-size: 0.85rem; }
.foot a { color: var(--accent); }

@media (max-width: 560px) {
  .opt-row { flex-direction: column; align-items: stretch; gap: 8px; }
}
</style>
