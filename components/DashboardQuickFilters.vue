<template>
  <div class="dashboard-quick-filters">
    <h3 class="widget-title">🔍 Quick Filters</h3>
    <div v-if="filters.length === 0" class="no-filters">
      <p>No saved filters yet.</p>
      <NuxtLink to="/explore" class="save-filter-link">Save your first filter</NuxtLink>
    </div>
    <div v-else class="filters-list">
      <button 
        v-for="filter in filters" 
        :key="filter.id"
        class="filter-chip"
        @click="applyFilter(filter)"
      >
        {{ filter.name }}
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useSavedFilters } from '~/composables/useSavedFilters'

const { savedFilters } = useSavedFilters()
const filters = ref([])

function applyFilter(filter) {
  // Emit event or navigate to apply filter
  console.log('Applying filter:', filter)
  // Could emit an event or use router to apply
}

onMounted(() => {
  filters.value = savedFilters.value?.slice(0, 8) || []
})
</script>

<style scoped>
.dashboard-quick-filters {
  height: 100%;
}

.widget-title {
  margin: 0 0 1rem 0;
  font-size: 1rem;
  color: var(--text, #222);
}

.no-filters {
  text-align: center;
  padding: 1.5rem;
  color: var(--muted, #999);
}

.save-filter-link {
  display: inline-block;
  margin-top: 0.75rem;
  color: var(--primary, #2a78d6);
  text-decoration: none;
  font-weight: 500;
}

.save-filter-link:hover {
  text-decoration: underline;
}

.filters-list {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.filter-chip {
  padding: 0.5rem 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border, #ddd);
  border-radius: 999px;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 500;
  color: var(--text, #222);
  transition: all 0.2s;
}

.filter-chip:hover {
  background: var(--primary, #2a78d6);
  color: white;
  border-color: var(--primary, #2a78d6);
}
</style>
