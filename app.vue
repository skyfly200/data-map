<template>
  <div class="app">
    <NuxtRouteAnnouncer />
    <header class="app-header">
      <div class="brand">
        <h1>Mushroom Observations</h1>
        <p>iNaturalist finds enriched with terrain &amp; environmental exposure, clustered by similarity.</p>
      </div>
      <div class="app-controls">
        <div class="dataset-picker">
          <label for="dataset-select">Dataset</label>
          <select id="dataset-select" :value="selectedDataset" @change="handleDatasetChange">
            <option v-for="dataset in availableDatasets" :key="dataset.id" :value="dataset.path">
              {{ dataset.label }}
            </option>
          </select>
        </div>
        <div class="units" role="group" aria-label="Elevation units">
          <button :class="{ active: unit === 'ft' }" @click="unit = 'ft'">ft</button>
          <button :class="{ active: unit === 'm' }" @click="unit = 'm'">m</button>
        </div>
        <nav class="app-nav">
          <NuxtLink to="/" class="nav-link">Map</NuxtLink>
          <NuxtLink to="/table" class="nav-link">Table</NuxtLink>
          <NuxtLink to="/charts" class="nav-link">Charts</NuxtLink>
        </nav>
      </div>
    </header>
    <main class="app-main">
      <NuxtPage />
    </main>
  </div>
</template>

<script setup>
import { useRoute } from '#app'
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

useHead({
  title: 'data-map · Mushroom Observations',
  meta: [{ name: 'description', content: 'Mushroom observations enriched with terrain and environmental exposure.' }],
})

const route = useRoute()
const { selectedDataset, availableDatasets, setDataset } = useObservations()

function handleDatasetChange(event) {
  setDataset(event.target.value)
}

watch(() => route.path, () => {
  // Force a reload when the route changes so each page maintains its own dataset selection.
  const datasetState = useState(`observations-${(route.path || '/').replace(/^\/+|\/+$/g, '') || 'root'}-dataset`)
  if (!datasetState.value && import.meta.client) {
    const saved = localStorage.getItem(`observations-${(route.path || '/').replace(/^\/+|\/+$/g, '') || 'root'}-dataset`)
    if (saved) datasetState.value = saved
  }
}, { immediate: true })

// Elevation unit: default feet, remembered per viewer.
const { unit } = useUnits()
onMounted(() => {
  const saved = localStorage.getItem('elev-unit')
  if (saved === 'm' || saved === 'ft') unit.value = saved
})
watch(unit, (v) => {
  if (import.meta.client) localStorage.setItem('elev-unit', v)
})
</script>

<style>
html, body, #__nuxt { height: 100%; margin: 0; }
body { font-family: system-ui, -apple-system, sans-serif; color: #1f2933; }

.app { display: flex; flex-direction: column; height: 100vh; }

.app-header {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  padding: 10px 20px; background: #1f2933; color: #fff; flex: 0 0 auto;
}
.brand h1 { margin: 0; font-size: 1.15rem; }
.brand p { margin: 2px 0 0; font-size: 0.82rem; opacity: 0.8; }

.app-controls { display: flex; align-items: center; gap: 14px; }

.dataset-picker {
  display: inline-flex; align-items: center; gap: 8px; font-size: 0.8rem; color: #dfe4ea;
}
.dataset-picker label { font-weight: 600; }
.dataset-picker select {
  background: #fff; color: #1f2933; border: 1px solid #d0d7de; border-radius: 6px;
  padding: 4px 8px; font-size: 0.8rem;
}

.units { display: inline-flex; border: 1px solid #52606d; border-radius: 6px; overflow: hidden; }
.units button {
  border: 0; background: transparent; color: #cbd2d9; cursor: pointer;
  padding: 5px 11px; font-size: 0.85rem; font-weight: 600;
}
.units button:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }
.units button.active { background: #3e4c59; color: #fff; }

.app-nav { display: flex; gap: 6px; }
.nav-link {
  color: #cbd2d9; text-decoration: none; font-size: 0.9rem; font-weight: 500;
  padding: 6px 12px; border-radius: 6px;
}
.nav-link:hover { background: rgba(255, 255, 255, 0.1); color: #fff; }
.nav-link.router-link-exact-active { background: #3e4c59; color: #fff; }

.app-main { flex: 1 1 auto; min-height: 0; overflow: auto; }
</style>
