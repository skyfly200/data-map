<template>
  <div class="dash-widget model-lb">
    <div class="widget-head">
      <h3 class="widget-title">🏆 Model leaderboard</h3>
      <NuxtLink to="/modeling/maxent" class="widget-link">Models ›</NuxtLink>
    </div>

    <p v-if="loading" class="lb-note">Loading…</p>
    <p v-else-if="!models.length" class="lb-note">
      No models yet. <NuxtLink to="/modeling/maxent" class="lb-run-link">Run one ›</NuxtLink>
    </p>

    <ol v-else class="lb-list">
      <li v-for="(m, i) in ranked" :key="m.id" class="lb-row">
        <span class="lb-rank">{{ i + 1 }}</span>
        <span class="lb-name" :title="m.title">{{ m.title }}</span>
        <span class="lb-grade" :class="`grade-${(m.grade || 'n').toLowerCase()}`">
          {{ m.grade || '—' }}
        </span>
        <span class="lb-auc" :title="`AUC ${m.auc?.toFixed(3) ?? '—'}`">
          {{ m.auc != null ? m.auc.toFixed(3) : '—' }}
        </span>
        <NuxtLink :to="`/map?layer=maxent:${m.id}`" class="lb-map-link" title="View on map">↗</NuxtLink>
      </li>
    </ol>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useMaxEnt } from '~/composables/useMaxEnt'

const { models, fetchModels, pending: loading } = useMaxEnt()

const ranked = computed(() =>
  [...(models.value || [])]
    .filter((m) => m.auc != null)
    .sort((a, b) => (b.auc ?? 0) - (a.auc ?? 0))
    .concat((models.value || []).filter((m) => m.auc == null))
    .slice(0, 10)
)

onMounted(() => { fetchModels() })
</script>

<style scoped>
.model-lb { height: 100%; display: flex; flex-direction: column; gap: 0.5rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }

.lb-note { font-size: 0.85rem; color: var(--muted, #888); }
.lb-run-link { color: var(--accent, #2a78d6); text-decoration: none; }
.lb-run-link:hover { text-decoration: underline; }

.lb-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.25rem; }
.lb-row {
  display: grid; grid-template-columns: 1.2rem 1fr auto auto 1.4rem;
  align-items: center; gap: 0.4rem;
  padding: 0.28rem 0.5rem; border-radius: 6px;
  background: var(--surface-2, #f5f5f5); border: 1px solid var(--border-soft, #eee);
  font-size: 0.8rem;
}
.lb-rank { color: var(--muted, #aaa); font-size: 0.7rem; font-variant-numeric: tabular-nums; text-align: right; }
.lb-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text, #222); }
.lb-grade {
  font-size: 0.68rem; font-weight: 700; border-radius: 4px;
  padding: 0 0.35rem; line-height: 1.6;
}
.grade-a { background: #d4f0d4; color: #286028; }
.grade-b { background: #ddf0c8; color: #3a5c1a; }
.grade-c { background: #fef3c7; color: #7c5e10; }
.grade-d { background: #ffe0d0; color: #7c2e10; }
.grade-f { background: #ffd6d6; color: #7c1010; }
.grade-n { background: var(--surface-2, #eee); color: var(--muted, #888); }

.lb-auc { font-variant-numeric: tabular-nums; color: var(--muted, #666); font-size: 0.75rem; }
.lb-map-link { color: var(--accent, #2a78d6); text-decoration: none; text-align: center; }
.lb-map-link:hover { text-decoration: underline; }
</style>
