<template>
  <div class="dash-widget data-quality">
    <div class="widget-head">
      <h3 class="widget-title">🔬 Data quality</h3>
      <NuxtLink to="/data" class="widget-link">Data ›</NuxtLink>
    </div>

    <p v-if="loading && !rows.length" class="dq-note">Loading…</p>
    <p v-else-if="!rows.length" class="dq-note">No observations loaded.</p>

    <div v-else class="dq-body">
      <div class="dq-section">
        <div class="dq-label">Enrichment</div>
        <div class="dq-bar-wrap" :title="enrichTooltip">
          <div class="dq-seg seg-full"  :style="{ width: pct(enrich.full, total) }"></div>
          <div class="dq-seg seg-partial" :style="{ width: pct(enrich.partial, total) }"></div>
          <div class="dq-seg seg-none"  :style="{ width: pct(enrich.none, total) }"></div>
        </div>
        <div class="dq-legend">
          <span class="dq-dot dot-full"></span><span class="dq-leg-label">Full {{ pct(enrich.full, total) }}</span>
          <span class="dq-dot dot-partial"></span><span class="dq-leg-label">Partial {{ pct(enrich.partial, total) }}</span>
          <span class="dq-dot dot-none"></span><span class="dq-leg-label">None {{ pct(enrich.none, total) }}</span>
        </div>
      </div>

      <div class="dq-section">
        <div class="dq-label">Location precision</div>
        <div class="dq-bar-wrap" :title="precTooltip">
          <div class="dq-seg seg-precise"  :style="{ width: pct(prec.precise, total) }"></div>
          <div class="dq-seg seg-coarse"   :style="{ width: pct(prec.coarse, total) }"></div>
          <div class="dq-seg seg-obscured" :style="{ width: pct(prec.obscured, total) }"></div>
          <div class="dq-seg seg-unknown"  :style="{ width: pct(prec.unknown, total) }"></div>
        </div>
        <div class="dq-legend">
          <span class="dq-dot dot-precise"></span><span class="dq-leg-label">Precise {{ pct(prec.precise, total) }}</span>
          <span class="dq-dot dot-coarse"></span><span class="dq-leg-label">Coarse {{ pct(prec.coarse, total) }}</span>
          <span class="dq-dot dot-obscured"></span><span class="dq-leg-label">Obscured {{ pct(prec.obscured, total) }}</span>
        </div>
      </div>

      <div class="dq-stats">
        <div class="dq-stat">
          <span class="dq-stat-val">{{ total.toLocaleString() }}</span>
          <span class="dq-stat-key">Total obs</span>
        </div>
        <div class="dq-stat">
          <span class="dq-stat-val">{{ speciesCount.toLocaleString() }}</span>
          <span class="dq-stat-key">Species</span>
        </div>
        <div class="dq-stat">
          <span class="dq-stat-val">{{ withDate.toLocaleString() }}</span>
          <span class="dq-stat-key">With date</span>
        </div>
        <div class="dq-stat">
          <span class="dq-stat-val">{{ withElev.toLocaleString() }}</span>
          <span class="dq-stat-key">With elev</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted } from 'vue'
import { useObservations } from '~/composables/useObservations'

const { rows, load, pending: loading } = useObservations()

const total = computed(() => rows.value?.length || 0)

const enrich = computed(() => {
  const r = { full: 0, partial: 0, none: 0 }
  for (const row of rows.value || []) {
    const e = row.enrichment_level || 'none'
    r[e] = (r[e] || 0) + 1
  }
  return r
})

const prec = computed(() => {
  const r = { precise: 0, coarse: 0, obscured: 0, unknown: 0 }
  for (const row of rows.value || []) {
    const p = row.location_precision || 'unknown'
    r[p] = (r[p] || 0) + 1
  }
  return r
})

const speciesCount = computed(() => new Set((rows.value || []).map((r) => r.species).filter(Boolean)).size)
const withDate = computed(() => (rows.value || []).filter((r) => r.date).length)
const withElev = computed(() => (rows.value || []).filter((r) => r.elevation != null && r.elevation !== '').length)

const enrichTooltip = computed(() =>
  `Full: ${enrich.value.full.toLocaleString()} · Partial: ${enrich.value.partial.toLocaleString()} · None: ${enrich.value.none.toLocaleString()}`
)
const precTooltip = computed(() =>
  `Precise: ${prec.value.precise.toLocaleString()} · Coarse: ${prec.value.coarse.toLocaleString()} · Obscured: ${prec.value.obscured.toLocaleString()}`
)

function pct(n, d) {
  if (!d) return '0%'
  return `${Math.round((n / d) * 100)}%`
}

onMounted(() => { load() })
</script>

<style scoped>
.data-quality { height: 100%; display: flex; flex-direction: column; gap: 0.6rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.dq-note { font-size: 0.85rem; color: var(--muted, #888); }

.dq-body { display: flex; flex-direction: column; gap: 0.7rem; }

.dq-section { display: flex; flex-direction: column; gap: 0.3rem; }
.dq-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted, #888); }

.dq-bar-wrap {
  height: 10px; border-radius: 5px; overflow: hidden;
  display: flex; background: var(--surface-2, #eee);
}
.dq-seg { height: 100%; transition: width 0.4s; }
.seg-full     { background: #3d9e5f; }
.seg-partial  { background: #a8cf8a; }
.seg-none     { background: #d4d4d4; }
.seg-precise  { background: #2a78d6; }
.seg-coarse   { background: #7ab4f0; }
.seg-obscured { background: #f0b870; }
.seg-unknown  { background: #d4d4d4; }

.dq-legend { display: flex; flex-wrap: wrap; gap: 0.25rem 0.6rem; }
.dq-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%; vertical-align: middle; margin-right: 2px;
}
.dot-full     { background: #3d9e5f; }
.dot-partial  { background: #a8cf8a; }
.dot-none     { background: #d4d4d4; }
.dot-precise  { background: #2a78d6; }
.dot-coarse   { background: #7ab4f0; }
.dot-obscured { background: #f0b870; }
.dq-leg-label { font-size: 0.65rem; color: var(--muted, #666); }

.dq-stats {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.4rem;
  margin-top: 0.2rem;
}
.dq-stat {
  display: flex; flex-direction: column; align-items: center;
  background: var(--surface-2, #f5f5f5); border: 1px solid var(--border-soft, #eee);
  border-radius: 6px; padding: 0.3rem 0.2rem;
}
.dq-stat-val { font-size: 0.82rem; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--text, #222); }
.dq-stat-key { font-size: 0.6rem; color: var(--muted, #888); text-align: center; }
</style>
