<template>
  <div class="dashboard-species-list">
    <h3 class="widget-title">🍄 Species List</h3>
    <div v-if="species.length === 0" class="no-species">
      <p>No species tracked yet.</p>
      <NuxtLink to="/explore" class="explore-link">Explore species</NuxtLink>
    <div class="widget-head">
      <h3 class="widget-title">🍄 Tracked Taxa</h3>
      <NuxtLink to="/data" class="widget-link">Data table ›</NuxtLink>
    </div>

    <div v-if="loading && !speciesList.length" class="no-species">
      <p>Loading observations…</p>
    </div>

    <div v-else-if="speciesList.length === 0" class="no-species">
      <p>No species records loaded.</p>
      <NuxtLink to="/map" class="explore-link">Explore the map</NuxtLink>
    </div>

    <ul v-else class="species-list">
      <li v-for="sp in species" :key="sp.id" class="species-item">
        <span class="species-name">{{ sp.name }}</span>
        <span v-if="sp.count" class="species-count">{{ sp.count }} records</span>
      <li v-for="sp in speciesList" :key="sp.species" class="species-item">
        <span class="species-name" :title="sp.species">{{ sp.species }}</span>
        <span class="species-count">{{ sp.count.toLocaleString() }} obs</span>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { computed, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'

const species = ref([])
const { speciesOptions, load, pending: loading } = useObservations()

onMounted(() => {
  if (import.meta.client) {
    const saved = localStorage.getItem('inat-taxa')
    if (saved) {
      try {
        const taxa = JSON.parse(saved)
        species.value = Array.isArray(taxa) ? taxa.slice(0, 8) : []
      } catch {
        species.value = []
      }
    }
  }
const speciesList = computed(() => (speciesOptions.value || []).slice(0, 8))

onMounted(async () => {
  await load()
})
</script>

<style scoped>
.dashboard-species-list {
  height: 100%;
  display: flex;
  flex-direction: column;
}

.widget-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 0.75rem;
}

.widget-title {
  margin: 0 0 1rem 0;
  margin: 0;
  font-size: 1rem;
  color: var(--text, #222);
}

.widget-link {
  font-size: 0.8rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
}

.widget-link:hover {
  text-decoration: underline;
}

.no-species {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted, #999);
}

.explore-link {
  display: inline-block;
  margin-top: 0.75rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
  font-weight: 500;
}

.explore-link:hover {
  text-decoration: underline;
}

.species-list {
  list-style: none;
  padding: 0;
  margin: 0;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  gap: 0.4rem;
  overflow-y: auto;
  max-height: 320px;
}

.species-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  padding: 0.45rem 0.65rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 6px;
  font-size: 0.9rem;
  font-size: 0.85rem;
}

.species-name {
  font-weight: 500;
  color: var(--text, #222);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 75%;
}

.species-count {
  font-size: 0.75rem;
  color: var(--muted, #666);
  font-variant-numeric: tabular-nums;
  flex-shrink: 0;
}
</style>
