<template>
  <div class="dashboard-species-list">
    <h3 class="widget-title">🍄 Species List</h3>
    <div v-if="species.length === 0" class="no-species">
      <p>No species tracked yet.</p>
      <NuxtLink to="/explore" class="explore-link">Explore species</NuxtLink>
    </div>
    <ul v-else class="species-list">
      <li v-for="sp in species" :key="sp.id" class="species-item">
        <span class="species-name">{{ sp.name }}</span>
        <span v-if="sp.count" class="species-count">{{ sp.count }} records</span>
      </li>
    </ul>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'

const species = ref([])

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
})
</script>

<style scoped>
.dashboard-species-list {
  height: 100%;
}

.widget-title {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text, #222);
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
}

.species-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.5rem 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 6px;
  font-size: 0.9rem;
}

.species-name {
  font-weight: 500;
  color: var(--text, #222);
}

.species-count {
  font-size: 0.75rem;
  color: var(--muted, #666);
}
</style>
