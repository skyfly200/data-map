<template>
  <div class="dash-widget species-list">
    <div class="widget-head">
      <h3 class="widget-title">🍄 Top species</h3>
      <NuxtLink to="/charts" class="widget-link">Charts ›</NuxtLink>
    </div>

    <p v-if="loading && !top.length" class="widget-note">Loading…</p>
    <p v-else-if="!top.length" class="widget-note">No observations loaded.</p>

    <ol v-else class="species-rows">
      <li v-for="s in top" :key="s.name" class="species-row">
        <span class="species-name">{{ s.name }}</span>
        <span class="species-bar"><span class="fill" :style="{ width: `${(s.count / top[0].count) * 100}%` }"></span></span>
        <span class="species-count">{{ s.count.toLocaleString() }}</span>
      </li>
    </ol>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()

const top = computed(() => {
  const counts = new Map()
  for (const r of rows.value || []) {
    const name = r.species
    if (!name) continue
    counts.set(name, (counts.get(name) || 0) + 1)
  }
  return [...counts.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 6)
})

onMounted(() => { load() })
</script>

<style scoped>
.species-list { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }
.species-rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.45rem; }
.species-row { display: grid; grid-template-columns: 1fr 3rem auto; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
.species-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-style: italic; color: var(--text, #222); }
.species-bar { height: 8px; background: var(--surface-2, #eee); border-radius: 4px; overflow: hidden; }
.species-bar .fill { display: block; height: 100%; background: var(--accent, #2a78d6); }
.species-count { font-variant-numeric: tabular-nums; color: var(--muted, #666); font-size: 0.78rem; }
</style>
