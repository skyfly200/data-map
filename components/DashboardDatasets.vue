<template>
  <div class="dash-widget datasets-widget">
    <div class="widget-head">
      <h3 class="widget-title">🗄️ Datasets</h3>
      <NuxtLink to="/data" class="widget-link">Manage ›</NuxtLink>
    </div>

    <p v-if="loading" class="widget-note">Loading…</p>
    <p v-else-if="error" class="widget-note error">{{ error }}</p>
    <p v-else-if="!datasets.length" class="widget-note">No datasets saved yet.</p>

    <ul v-else class="ds-list">
      <li v-for="ds in datasets" :key="ds.id" class="ds-item">
        <div class="ds-info">
          <span class="ds-title">{{ ds.title || ds.slug }}</span>
          <span class="ds-meta">
            <span class="ds-vis" :class="`vis-${ds.visibility}`">{{ VISIBILITY_LABELS[ds.visibility] || ds.visibility }}</span>
            <span class="ds-date">{{ formatDate(ds.created_at) }}</span>
          </span>
        </div>
        <NuxtLink :to="`/data?dataset=${ds.slug}`" class="ds-link" title="Open">›</NuxtLink>
      </li>
    </ul>

    <div class="ds-footer">
      <span class="ds-count">{{ datasets.length }} dataset{{ datasets.length !== 1 ? 's' : '' }}</span>
    </div>
  </div>
</template>

<script setup>
import { onMounted } from 'vue'
import { useDatasets } from '~/composables/useDatasets'

const { datasets, loading, error, refresh, VISIBILITY_LABELS } = useDatasets()

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
function formatDate(d) {
  if (!d) return ''
  const dt = new Date(d)
  if (isNaN(dt)) return ''
  return `${MONTHS[dt.getMonth()]} ${dt.getDate()}, ${dt.getFullYear()}`
}

onMounted(() => { refresh() })
</script>

<style scoped>
.datasets-widget { height: 100%; display: flex; flex-direction: column; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 0.75rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-note { text-align: center; padding: 1.25rem; color: var(--muted, #999); font-size: 0.85rem; }
.widget-note.error { color: #c0392b; }

.ds-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.4rem; flex: 1; overflow-y: auto; }
.ds-item {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.45rem 0.6rem; background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px; gap: 0.5rem;
}
.ds-info { display: flex; flex-direction: column; gap: 0.15rem; min-width: 0; }
.ds-title { font-size: 0.85rem; font-weight: 600; color: var(--text, #222); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ds-meta { display: flex; gap: 0.5rem; align-items: center; }
.ds-vis {
  font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 1px 5px; border-radius: 3px; font-weight: 600;
}
.vis-private { background: #fee; color: #c0392b; }
.vis-members { background: #fef9e7; color: #b7770d; }
.vis-public { background: #eafaf1; color: #1e8449; }
.ds-date { font-size: 0.7rem; color: var(--muted, #aaa); }
.ds-link { font-size: 1rem; color: var(--accent, #2a78d6); text-decoration: none; flex-shrink: 0; }
.ds-link:hover { color: var(--text, #222); }

.ds-footer { margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid var(--border-soft, #eee); }
.ds-count { font-size: 0.72rem; color: var(--muted, #aaa); }
</style>
