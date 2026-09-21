<template>
  <div class="dashboard-quick-filters">
    <div class="widget-head">
      <h3 class="widget-title">🔍 Quick Filters</h3>
      <NuxtLink to="/data" class="widget-link">Data filters ›</NuxtLink>
    </div>

    <div v-if="filterList.length === 0" class="no-filters">
      <p>No saved filters yet.</p>
      <NuxtLink to="/data" class="save-filter-link">Save a filter set on Data page</NuxtLink>
    </div>

    <div v-else class="filters-list">
      <button 
        v-for="filter in filterList" 
        :key="filter.id"
        class="filter-chip"
        :class="{ active: filter.id === activeId }"
        @click="applyFilter(filter.id)"
        :title="describeFilters(filter.snapshot)"
      >
        <span class="chip-name">{{ filter.name }}</span>
        <span class="chip-desc">{{ describeFilters(filter.snapshot) }}</span>
      </button>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSavedFilters, describeFilters } from '~/composables/useSavedFilters'

const router = useRouter()
const { subsets, activeId, loadFromStorage, apply } = useSavedFilters()

const filterList = computed(() => (subsets.value || []).slice(0, 8))

function applyFilter(id) {
  apply(id)
  router.push('/map')
}

onMounted(() => {
  loadFromStorage()
})
</script>

<style scoped>
.dashboard-quick-filters {
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
  flex-direction: column;
  gap: 0.5rem;
  overflow-y: auto;
  max-height: 320px;
}

.filter-chip {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 0.5rem 0.75rem;
  background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border, #ddd);
  border-radius: 6px;
  cursor: pointer;
  text-align: left;
  transition: all 0.2s;
  width: 100%;
}

.filter-chip:hover {
  background: var(--surface, #fff);
  border-color: var(--primary, #2a78d6);
}

.filter-chip.active {
  background: rgba(42, 120, 214, 0.08);
  border-color: var(--primary, #2a78d6);
}

.chip-name {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text, #222);
  margin-bottom: 0.2rem;
}

.chip-desc {
  font-size: 0.75rem;
  color: var(--muted, #777);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}
</style>
