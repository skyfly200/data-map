<template>
  <div class="app">
    <NuxtRouteAnnouncer />
    <header class="app-header">
      <NuxtLink to="/" class="brand">
        <h1>Mushroom Observations</h1>
      </NuxtLink>
      <div class="app-controls">
        <div class="dataset-picker">
          <label for="dataset-select">Dataset</label>
          <select id="dataset-select" :value="selectedDataset" @change="handleDatasetChange">
            <option v-for="dataset in availableDatasets" :key="dataset.id" :value="dataset.path">
              {{ dataset.label }}
            </option>
          </select>
        </div>
        <NuxtLink v-if="filterCount" to="/data" class="filter-flag" title="Active filters — manage on the Data tab">
          Filters: {{ filterCount }}
        </NuxtLink>
        <div class="units" role="group" aria-label="Elevation units">
          <button :class="{ active: unit === 'ft' }" @click="unit = 'ft'">ft</button>
          <button :class="{ active: unit === 'm' }" @click="unit = 'm'">m</button>
        </div>
        <div class="units" role="group" aria-label="Temperature units">
          <button :class="{ active: tempUnit === 'F' }" @click="tempUnit = 'F'">°F</button>
          <button :class="{ active: tempUnit === 'C' }" @click="tempUnit = 'C'">°C</button>
        </div>
        <nav class="app-nav">
          <NuxtLink to="/" class="nav-link">Home</NuxtLink>
          <NuxtLink to="/map" class="nav-link">Map</NuxtLink>
          <NuxtLink to="/table" class="nav-link">Table</NuxtLink>
          <NuxtLink to="/charts" class="nav-link">Charts</NuxtLink>
          <NuxtLink to="/explore" class="nav-link">Explore</NuxtLink>
          <NuxtLink to="/data" class="nav-link">Data</NuxtLink>
        </nav>
        <ClientOnly>
          <div class="auth-box" v-if="configured">
            <template v-if="isAuthed">
              <span class="avatar" :title="user?.email || ''">{{ initial }}</span>
              <span class="who" :title="user?.email || ''">{{ shortEmail }}</span>
              <button class="auth-btn" @click="signOut">Sign out</button>
            </template>
            <NuxtLink v-else to="/login" class="auth-btn as-link">Sign in</NuxtLink>
          </div>
        </ClientOnly>
      </div>
    </header>
    <main class="app-main">
      <NuxtPage />
    </main>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

useHead({
  title: 'data-map · Mushroom Observations',
  meta: [{ name: 'description', content: 'Mushroom observations enriched with terrain and environmental exposure.' }],
})

const { selectedDataset, availableDatasets, setDataset, loadDatasets } = useObservations()
onMounted(loadDatasets)

const { activeCount: filterCount } = useFilters()

const { user, isAuthed, configured, signOut } = useAuth()
const shortEmail = computed(() => {
  const e = user.value?.email
  return e ? e.split('@')[0] : 'Account'
})
const initial = computed(() => {
  const e = user.value?.email
  return (e ? e[0] : '?').toUpperCase()
})

function handleDatasetChange(event) {
  setDataset(event.target.value)
}

// Units: default feet + Fahrenheit, remembered per viewer.
const { unit, tempUnit } = useUnits()
onMounted(() => {
  const e = localStorage.getItem('elev-unit')
  if (e === 'm' || e === 'ft') unit.value = e
  const t = localStorage.getItem('temp-unit')
  if (t === 'F' || t === 'C') tempUnit.value = t
  // Restore the dataset choice after a hard reload (SSR can't read localStorage).
  const ds = localStorage.getItem('observations-dataset')
  if (ds && ds !== selectedDataset.value) setDataset(ds)
})
watch(unit, (v) => {
  if (import.meta.client) localStorage.setItem('elev-unit', v)
})
watch(tempUnit, (v) => {
  if (import.meta.client) localStorage.setItem('temp-unit', v)
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
.brand { text-decoration: none; color: inherit; }
.brand h1 { margin: 0; font-size: 1.15rem; }
.brand:hover h1 { opacity: 0.85; }

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

.filter-flag { background: #2b7a3d; color: #fff; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; text-decoration: none; white-space: nowrap; }
.filter-flag:hover { background: #256a34; }

.auth-box { display: inline-flex; align-items: center; gap: 8px; }
.auth-box .avatar {
  display: inline-flex; align-items: center; justify-content: center;
  width: 24px; height: 24px; border-radius: 50%; background: #3e4c59; color: #fff;
  font-size: 0.72rem; font-weight: 700; flex: 0 0 auto;
}
.auth-box .who { font-size: 0.8rem; color: #cbd2d9; max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.auth-btn { border: 1px solid #52606d; background: transparent; color: #cbd2d9; border-radius: 6px; padding: 5px 11px; font-size: 0.82rem; font-weight: 600; cursor: pointer; text-decoration: none; }
.auth-btn:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }
.auth-btn.as-link { display: inline-block; }

.app-main { flex: 1 1 auto; min-height: 0; overflow: auto; }
</style>
