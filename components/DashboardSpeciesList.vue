<template>
  <div class="dash-widget species-list">
    <div class="widget-head">
      <h3 class="widget-title"><span aria-hidden="true">🍄 </span>Top species</h3>
      <div class="widget-actions">
        <select v-model="sortMode" class="sort-select" title="Sort by">
          <option value="count">By count</option>
          <option value="recent">Recently observed</option>
          <option value="genus">By genus</option>
        </select>
        <NuxtLink to="/charts" class="widget-link">Charts ›</NuxtLink>
      </div>
    </div>

    <p v-if="loading && !top.length" class="widget-note">Loading…</p>
    <p v-else-if="!top.length" class="widget-note">No observations loaded.</p>

    <ol v-else class="species-rows">
      <li v-for="s in top" :key="s.name" class="species-row">
        <span class="species-name" :title="s.name">
          <span v-if="sortMode === 'genus' && s.genus" class="genus-prefix">{{ s.genus }}</span>
          <span v-if="sortMode === 'genus' && s.genus"> · </span>{{ sortMode === 'genus' && s.genus ? s.name.slice(s.genus.length).trim() || s.name : s.name }}
        </span>
        <span class="species-bar"><span class="fill" :style="{ width: `${(s.count / maxCount) * 100}%` }"></span></span>
        <span class="species-meta">
          <span class="species-count">{{ s.count.toLocaleString() }}</span>
          <span v-if="sortMode === 'recent' && s.lastDate" class="species-date">{{ formatDate(s.lastDate) }}</span>
        </span>
      </li>
    </ol>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()
const sortMode = ref('count')

const MONTH = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
function formatDate(d) {
  const dt = new Date(d)
  if (isNaN(dt)) return ''
  return `${MONTH[dt.getMonth()]} ${dt.getDate()}`
}

const speciesMap = computed(() => {
  const m = new Map()
  for (const r of rows.value || []) {
    const name = r.species
    if (!name) continue
    if (!m.has(name)) m.set(name, { name, count: 0, lastDate: null })
    const entry = m.get(name)
    entry.count++
    if (r.date) {
      if (!entry.lastDate || r.date > entry.lastDate) entry.lastDate = r.date
    }
  }
  return m
})

const top = computed(() => {
  const list = [...speciesMap.value.values()]
  if (sortMode.value === 'recent') {
    return list
      .filter((s) => s.lastDate)
      .sort((a, b) => (b.lastDate > a.lastDate ? 1 : -1))
      .slice(0, 8)
  }
  if (sortMode.value === 'genus') {
    // One representative species per genus (highest count), sorted by genus name.
    const byGenus = new Map()
    for (const s of list) {
      const genus = s.name.split(' ')[0]
      if (!byGenus.has(genus) || s.count > byGenus.get(genus).count) {
        byGenus.set(genus, { ...s, genus })
      }
    }
    return [...byGenus.values()]
      .sort((a, b) => a.genus.localeCompare(b.genus))
      .slice(0, 8)
  }
  // default: by count
  return list.sort((a, b) => b.count - a.count).slice(0, 8)
})

const maxCount = computed(() => Math.max(1, ...top.value.map((s) => s.count)))

onMounted(() => { load() })
</script>

<style scoped>
.species-list { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; gap: 0.5rem; flex-wrap: wrap; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; align-items: center; gap: 0.5rem; }
.sort-select {
  font: inherit; font-size: 0.72rem; border: 1px solid var(--border, #ddd);
  background: var(--surface, #fff); color: var(--muted, #666);
  border-radius: 6px; padding: 0.15rem 0.4rem; cursor: pointer;
}
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; white-space: nowrap; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }
.species-rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.45rem; flex: 1; overflow-y: auto; }
.species-row { display: grid; grid-template-columns: 1fr 3rem auto; align-items: center; gap: 0.5rem; font-size: 0.85rem; }
.species-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-style: italic; color: var(--text, #222); display: flex; align-items: center; gap: 0.2rem; flex-wrap: nowrap; min-width: 0; }
.genus-prefix { font-weight: 700; flex-shrink: 0; }
.species-bar { height: 8px; background: var(--surface-2, #eee); border-radius: 4px; overflow: hidden; }
.species-bar .fill { display: block; height: 100%; background: var(--accent, #2a78d6); transition: width 0.3s; }
.species-meta { display: flex; flex-direction: column; align-items: flex-end; min-width: 4rem; }
.species-count { font-variant-numeric: tabular-nums; color: var(--muted, #666); font-size: 0.78rem; }
.species-date { font-size: 0.68rem; color: var(--muted, #aaa); }
</style>
