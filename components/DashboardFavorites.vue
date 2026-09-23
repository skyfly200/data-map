<template>
  <div class="dash-widget favorites">
    <div class="widget-head">
      <h3 class="widget-title">⭐ Favorites</h3>
      <NuxtLink to="/map" class="widget-link">Map ›</NuxtLink>
    </div>

    <div v-if="!favorites.length" class="fav-empty">
      <p>No favorites yet.</p>
      <p class="fav-hint">Star a species on the map or table to track it here.</p>
    </div>

    <ul v-else class="fav-list">
      <li v-for="sp in favorites" :key="sp.name" class="fav-item">
        <span class="fav-name" :title="sp.name">{{ sp.name }}</span>
        <span class="fav-badges">
          <span v-if="sp.fruiting" class="fav-badge badge-fruiting" title="Currently in fruiting window">🍄</span>
          <span class="fav-count" :title="`${sp.count.toLocaleString()} observations`">
            {{ sp.count.toLocaleString() }}
          </span>
          <NuxtLink :to="`/map?species=${encodeURIComponent(sp.name)}`" class="fav-map-btn" title="Filter map to this species">↗</NuxtLink>
          <button class="fav-remove" title="Remove from favorites" @click="remove(sp.name)">×</button>
        </span>
      </li>
    </ul>

    <form class="fav-add-form" @submit.prevent="addFromInput">
      <input
        v-model="addInput"
        class="fav-input"
        placeholder="Add species…"
        list="fav-species-list"
        autocomplete="off"
      />
      <datalist id="fav-species-list">
        <option v-for="s in speciesOptions.slice(0, 50)" :key="s.species" :value="s.species" />
      </datalist>
      <button type="submit" class="fav-add-btn">+</button>
    </form>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, speciesOptions } = useObservations()

const STORAGE_KEY = 'dashboard-favorites'
const WINDOW_DAYS = 30

const favoriteNames = ref([])
const addInput = ref('')

function loadFavorites() {
  if (!import.meta.client) return
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    if (Array.isArray(saved)) favoriteNames.value = saved
  } catch { /* ignore */ }
}

function saveFavorites() {
  if (!import.meta.client) return
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(favoriteNames.value)) } catch { /* ignore */ }
}

function remove(name) {
  favoriteNames.value = favoriteNames.value.filter((n) => n !== name)
  saveFavorites()
}

function addFromInput() {
  const name = addInput.value.trim()
  if (!name || favoriteNames.value.includes(name)) { addInput.value = ''; return }
  favoriteNames.value = [...favoriteNames.value, name]
  saveFavorites()
  addInput.value = ''
}

function dayOfYear(date) {
  const start = new Date(date.getFullYear(), 0, 0)
  return Math.floor((date - start) / 86400000)
}
function doyDist(a, b) {
  const d = Math.abs(a - b)
  return d > 182 ? 365 - d : d
}
const todayDoy = dayOfYear(new Date())

function obsDoy(r) {
  if (r.date) { const d = new Date(r.date); if (!isNaN(d)) return dayOfYear(d) }
  if (r.month) return dayOfYear(new Date(2001, r.month - 1, 15))
  return null
}

const countsBySpecies = computed(() => {
  const m = new Map()
  for (const r of rows.value || []) {
    if (r.species) m.set(r.species, (m.get(r.species) || 0) + 1)
  }
  return m
})

const fruitingNow = computed(() => {
  const inWindow = new Set()
  for (const r of rows.value || []) {
    if (!r.species) continue
    const doy = obsDoy(r)
    if (doy != null && doyDist(doy, todayDoy) <= WINDOW_DAYS) inWindow.add(r.species)
  }
  return inWindow
})

const favorites = computed(() =>
  favoriteNames.value.map((name) => ({
    name,
    count: countsBySpecies.value.get(name) || 0,
    fruiting: fruitingNow.value.has(name),
  }))
)

onMounted(() => {
  loadFavorites()
  load()
})
</script>

<style scoped>
.favorites { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }

.fav-empty { flex: 1; display: flex; flex-direction: column; justify-content: center; align-items: center; gap: 0.2rem; }
.fav-empty p { margin: 0; font-size: 0.85rem; color: var(--muted, #888); }
.fav-hint { font-size: 0.75rem !important; }

.fav-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; flex: 1; overflow-y: auto; }
.fav-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.28rem 0.5rem; border-radius: 6px;
  background: var(--surface-2, #f5f5f5); border: 1px solid var(--border-soft, #eee);
}
.fav-name { font-size: 0.8rem; font-style: italic; color: var(--text, #222); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; flex: 1; }
.fav-badges { display: flex; align-items: center; gap: 0.35rem; flex-shrink: 0; }
.fav-badge { font-size: 0.8rem; }
.fav-count { font-size: 0.72rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }
.fav-map-btn { font-size: 0.75rem; color: var(--accent, #2a78d6); text-decoration: none; }
.fav-map-btn:hover { text-decoration: underline; }
.fav-remove { border: 0; background: none; color: var(--muted, #bbb); font-size: 1rem; line-height: 1; cursor: pointer; padding: 0; }
.fav-remove:hover { color: #b3492f; }

.fav-add-form { display: flex; gap: 0.4rem; margin-top: auto; }
.fav-input {
  flex: 1; font: inherit; font-size: 0.8rem; border: 1px solid var(--border, #ddd);
  border-radius: 6px; padding: 0.3rem 0.5rem; background: var(--surface, #fff); color: var(--text, #222);
  min-width: 0;
}
.fav-input:focus { outline: none; border-color: var(--accent, #2a78d6); }
.fav-add-btn {
  border: 1px solid var(--accent, #2a78d6); background: var(--accent, #2a78d6); color: #fff;
  border-radius: 6px; font-size: 1rem; line-height: 1; padding: 0.3rem 0.6rem; cursor: pointer;
}
.fav-add-btn:hover { filter: brightness(1.1); }
</style>
