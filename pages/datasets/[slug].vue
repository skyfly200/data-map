<template>
  <div class="ds-page">
    <NuxtLink to="/jobs" class="back">‹ Your datasets</NuxtLink>

    <p v-if="loading" class="msg">Loading…</p>
    <p v-else-if="error" class="msg error">{{ error }}</p>

    <template v-else-if="detail">
      <!-- ── Header ─────────────────────────────────────────────────── -->
      <header class="head">
        <h2>{{ ds.title }}</h2>
        <div class="head-meta">
          <span class="vis-badge" :class="ds.visibility">{{ VISIBILITY_LABELS[ds.visibility] || ds.visibility }}</span>
          <span v-if="!ds.mine" class="tag">Shared with you</span>
          <code>{{ ds.slug }}</code>
        </div>
        <p v-if="ds.description" class="desc">{{ ds.description }}</p>
        <div class="actions">
          <NuxtLink :to="`/map?dataset=${encodeURIComponent(ds.slug)}`" class="btn small primary">Open on map</NuxtLink>
          <NuxtLink :to="`/jobs?dataset=${encodeURIComponent(ds.slug)}`" class="btn small">Run a job on it</NuxtLink>
          <ExportMenu :source="() => datasetsApi.fetchGeojson(ds.slug)" :total="ds.feature_count || 0" :name="ds.title" />
        </div>
      </header>

      <dl class="facts">
        <div><dt>Made by</dt><dd>{{ detail.origin_label }}</dd></div>
        <div><dt>Points</dt><dd>{{ fmtNum(ds.feature_count ?? columns?.total) }}</dd></div>
        <div v-if="ds.bytes"><dt>Size</dt><dd>{{ fmtBytes(ds.bytes) }}</dd></div>
        <div><dt>Saved</dt><dd>{{ fmtWhen(ds.created_at) }}</dd></div>
        <div v-if="ds.updated_at && ds.updated_at !== ds.created_at"><dt>Updated</dt><dd>{{ fmtWhen(ds.updated_at) }}</dd></div>
      </dl>

      <!-- ── How it was made ────────────────────────────────────────── -->
      <section class="panel">
        <h3>How it was made</h3>

        <!-- From a job: the spec it ran and what came back. -->
        <template v-if="job">
          <dl class="settings">
            <div><dt>Job</dt><dd>{{ KIND_LABELS[job.kind] || job.kind }} · <span :class="`st-${job.status}`">{{ job.status }}</span></dd></div>
            <div v-if="detail.from"><dt>Source</dt>
              <dd>
                <span v-if="detail.from.hidden" class="muted">A dataset you can't see</span>
                <NuxtLink v-else :to="`/datasets/${encodeURIComponent(detail.from.slug)}`">{{ detail.from.title }}</NuxtLink>
              </dd>
            </div>
            <template v-else-if="settings.source?.type === 'bbox'">
              <div><dt>Source</dt><dd>Observations in an area</dd></div>
              <div v-if="settings.source.taxon"><dt>Taxon</dt><dd><em>{{ settings.source.taxon }}</em></dd></div>
              <div v-if="settings.source.bounds"><dt>Area</dt><dd>{{ fmtBounds(settings.source.bounds) }}</dd></div>
              <div><dt>Dates</dt><dd>{{ fmtRange(settings.source.dateFrom, settings.source.dateTo) }}</dd></div>
            </template>
            <div v-if="settings.points != null"><dt>Points sampled</dt><dd>{{ fmtNum(settings.points) }}</dd></div>
            <div v-if="settings.dates != null"><dt>Distinct dates</dt><dd>{{ fmtNum(settings.dates) }}</dd></div>
            <div v-if="job.cost_units != null"><dt>Cost</dt><dd>{{ job.cost_units }} units</dd></div>
            <div v-if="job.started_at"><dt>Ran</dt><dd>{{ fmtWhen(job.started_at) }}<template v-if="runTime"> · {{ runTime }}</template></dd></div>
          </dl>

          <template v-if="jobStages.length">
            <h4>Stages requested</h4>
            <ul class="stage-list">
              <li v-for="s in jobStages" :key="s.key" class="stage">
                <div class="stage-top">
                  <strong>{{ s.label }}</strong>
                  <span v-if="skippedFor(s.key)" class="warn">{{ fmtNum(skippedFor(s.key)) }} skipped</span>
                </div>
                <p class="stage-desc">{{ s.description }}</p>
                <p class="stage-src"><code>{{ s.asset }}</code> · {{ s.scale >= 1000 ? `${(s.scale / 1000).toFixed(1)} km` : `${s.scale} m` }} · {{ s.perDate ? 'per observation date' : 'static' }}</p>
              </li>
            </ul>
          </template>
        </template>

        <!-- Fetched, uploaded or imported: what was recorded about it. -->
        <template v-else>
          <dl v-if="prov" class="settings">
            <div v-if="prov.taxon"><dt>Taxon</dt><dd><em>{{ prov.taxon }}</em></dd></div>
            <div v-if="isFetch"><dt>Dates</dt><dd>{{ fmtRange(prov.date_from, prov.date_to) }}</dd></div>
            <div v-if="prov.lat != null && prov.lng != null"><dt>Area</dt>
              <dd>{{ prov.radius_km }} km around {{ prov.lat.toFixed(3) }}, {{ prov.lng.toFixed(3) }}</dd>
            </div>
            <div v-if="prov.quality"><dt>Records</dt><dd>{{ prov.quality }} only</dd></div>
            <div v-if="prov.file_name"><dt>File</dt><dd><code>{{ prov.file_name }}</code><template v-if="prov.format"> ({{ prov.format.toUpperCase() }})</template></dd></div>
            <div v-if="prov.asset_path"><dt>Asset</dt><dd><code>{{ prov.asset_path }}</code></dd></div>
            <div v-if="prov.records != null"><dt>Kept</dt><dd>{{ fmtNum(prov.records) }}</dd></div>
            <div v-if="prov.skipped"><dt>Skipped</dt><dd>{{ fmtNum(prov.skipped) }} without usable coordinates</dd></div>
            <div v-if="prov.media"><dt>Media</dt><dd>{{ fmtNum(prov.media) }}</dd></div>
            <div v-if="prov.dna"><dt>DNA records</dt><dd>{{ fmtNum(prov.dna) }}</dd></div>
            <div v-if="prov.fetched_at"><dt>Fetched</dt><dd>{{ fmtWhen(prov.fetched_at) }}</dd></div>
          </dl>
          <p v-if="detail.inferred" class="msg">
            Saved before fetch and import settings were recorded, so only where it came from is known.
            Fetching or importing it again records the rest.
          </p>
        </template>

        <details v-if="job || prov" class="raw">
          <summary>Raw settings</summary>
          <pre>{{ JSON.stringify(job ? settings : prov, null, 2) }}</pre>
        </details>
      </section>

      <!-- ── What it carries ────────────────────────────────────────── -->
      <section class="panel">
        <div class="panel-head">
          <h3>Enriched with</h3>
          <span v-if="columns" class="muted small">{{ enrichedGroups.length }} layer{{ enrichedGroups.length === 1 ? '' : 's' }} · {{ columnCount }} columns</span>
        </div>
        <p class="sub">Read from the points themselves, with how many carry each value.</p>

        <p v-if="columnsLoading" class="msg">Reading the points…</p>
        <p v-else-if="columnsError" class="msg error">{{ columnsError }}</p>
        <template v-else-if="columns">
          <p v-if="!enrichedGroups.length" class="msg">
            No environmental layers yet.
            <NuxtLink :to="`/jobs?dataset=${encodeURIComponent(ds.slug)}`">Run an enrichment job</NuxtLink> to add them.
          </p>
          <div v-for="g in enrichedGroups" :key="g.key" class="col-group">
            <div class="group-head">
              <strong>{{ g.label }}</strong>
              <span class="muted small">{{ groupFill(g) }}</span>
            </div>
            <ul class="cols">
              <li v-for="c in g.columns" :key="c.name" class="col">
                <code class="col-name">{{ c.name }}</code>
                <span class="bar"><span class="fill" :class="fillClass(c.fraction)" :style="{ width: `${Math.round(c.fraction * 100)}%` }" /></span>
                <span class="pct">{{ fmtPct(c.fraction) }}</span>
              </li>
            </ul>
          </div>

          <details v-if="observationGroup" class="col-group obs">
            <summary><strong>Observation fields</strong> <span class="muted small">{{ observationGroup.columns.length }}</span></summary>
            <ul class="cols">
              <li v-for="c in observationGroup.columns" :key="c.name" class="col">
                <code class="col-name">{{ c.name }}</code>
                <span class="bar"><span class="fill" :class="fillClass(c.fraction)" :style="{ width: `${Math.round(c.fraction * 100)}%` }" /></span>
                <span class="pct">{{ fmtPct(c.fraction) }}</span>
              </li>
            </ul>
          </details>
        </template>
      </section>

      <!-- ── Jobs run over it ───────────────────────────────────────── -->
      <section v-if="detail.enrichments.length" class="panel">
        <h3>Jobs run on this dataset</h3>
        <ul class="runs">
          <li v-for="r in detail.enrichments" :key="r.id" class="run">
            <div class="run-top">
              <strong>{{ r.title || KIND_LABELS[r.kind] || r.kind }}</strong>
              <span :class="`st-${r.status}`">{{ r.status }}</span>
              <span class="muted small">{{ fmtWhen(r.created_at) }}</span>
            </div>
            <p v-if="(r.settings.stages || []).length" class="run-stages">
              <span v-for="k in r.settings.stages" :key="k" class="chip">{{ stageLabel(k) }}</span>
            </p>
            <p v-if="r.saved_as" class="muted small">
              Saved as <NuxtLink :to="`/datasets/${encodeURIComponent(r.saved_as.slug)}`">{{ r.saved_as.title }}</NuxtLink>
            </p>
          </li>
        </ul>
      </section>
    </template>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { VISIBILITY_LABELS, useDatasets } from '~/composables/useDatasets'
import { columnSummary } from '~/netlify/lib/dataset-columns.mjs'

const route = useRoute()
const datasetsApi = useDatasets()

const KIND_LABELS = { enrich: 'Enrichment', model: 'Habitat model', export: 'Export' }

const detail = ref(null)
const loading = ref(true)
const error = ref('')
const columns = ref(null)
const columnsLoading = ref(false)
const columnsError = ref('')

const ds = computed(() => detail.value?.dataset || {})
const job = computed(() => detail.value?.job || null)
const settings = computed(() => job.value?.settings || {})
const prov = computed(() => detail.value?.provenance || null)
const isFetch = computed(() => ['inat', 'gbif'].includes(detail.value?.origin))
const stageByKey = computed(() => new Map((detail.value?.stages || []).map((s) => [s.key, s])))
const jobStages = computed(() => (settings.value.stages || []).map((k) => stageByKey.value.get(k) || { key: k, label: k }))
const stageLabel = (k) => stageByKey.value.get(k)?.label || k
const skippedFor = (k) => job.value?.result?.skipped?.[k] || 0

const enrichedGroups = computed(() => (columns.value?.groups || []).filter((g) => g.key !== 'observation'))
const observationGroup = computed(() => (columns.value?.groups || []).find((g) => g.key === 'observation') || null)
const columnCount = computed(() => enrichedGroups.value.reduce((n, g) => n + g.columns.length, 0))

const runTime = computed(() => {
  const a = Date.parse(job.value?.started_at || '')
  const b = Date.parse(job.value?.finished_at || '')
  if (!a || !b || b < a) return ''
  const s = Math.round((b - a) / 1000)
  return s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`
})

async function load(slug) {
  loading.value = true
  error.value = ''
  detail.value = null
  columns.value = null
  try {
    detail.value = await datasetsApi.fetchDetail(slug)
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
  if (detail.value) loadColumns(slug)
}

async function loadColumns(slug) {
  columnsLoading.value = true
  columnsError.value = ''
  try {
    const geojson = await datasetsApi.fetchGeojson(slug)
    columns.value = columnSummary(geojson?.features || [], detail.value?.stages || [])
  } catch (e) {
    columnsError.value = `Could not read the points: ${e.message}`
  } finally {
    columnsLoading.value = false
  }
}

watch(() => route.params.slug, (slug) => { if (slug) load(String(slug)) }, { immediate: true })

useHead(() => ({ title: ds.value.title ? `${ds.value.title} · Dataset` : 'Dataset' }))

const fmtNum = (n) => (n == null ? '—' : Number(n).toLocaleString())
const fmtPct = (f) => (f >= 0.995 ? '100%' : f > 0 && f < 0.01 ? '<1%' : `${Math.round(f * 100)}%`)
const fillClass = (f) => (f >= 0.9 ? 'good' : f >= 0.5 ? 'fair' : 'thin')
function groupFill(g) {
  const lo = Math.min(...g.columns.map((c) => c.fraction))
  return lo >= 0.995 ? 'complete' : `lowest ${fmtPct(lo)}`
}
function fmtBytes(b) {
  if (b < 1024) return `${b} B`
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(0)} KB`
  return `${(b / 1024 ** 2).toFixed(1)} MB`
}
function fmtWhen(d) {
  const t = new Date(d)
  if (Number.isNaN(t.getTime())) return '—'
  return t.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })
}
function fmtRange(from, to) {
  if (!from && !to) return 'Any date'
  if (from && to) return `${from} to ${to}`
  return from ? `From ${from}` : `Up to ${to}`
}
function fmtBounds(b) {
  const f = (v) => Number(v).toFixed(2)
  return `${f(b.south)}° to ${f(b.north)}° N, ${f(b.west)}° to ${f(b.east)}° E`
}
</script>

<style scoped>
.ds-page { padding: 16px 18px; max-width: 860px; margin: 0 auto; }
.back { display: inline-block; margin-bottom: 12px; font-size: 0.82rem; color: var(--muted); text-decoration: none; }
.back:hover { color: var(--text); }

.head h2 { margin: 0 0 6px; font-size: 1.25rem; overflow-wrap: anywhere; }
.head-meta { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 0.8rem; }
.head-meta code { color: var(--muted); overflow-wrap: anywhere; }
.desc { margin: 8px 0 0; font-size: 0.86rem; color: var(--muted); }
.actions { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 12px; }

.facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px;
  margin: 16px 0 18px; }
.facts div { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; background: var(--surface); }
.facts dt { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); }
.facts dd { margin: 2px 0 0; font-size: 0.92rem; font-weight: 600; }

.panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  background: var(--surface); margin-bottom: 18px; }
.panel h3 { margin: 0 0 12px; font-size: 0.95rem; }
.panel h4 { margin: 14px 0 8px; font-size: 0.85rem; }
.panel-head { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
.panel-head h3 { margin-bottom: 4px; }
.sub { margin: 0 0 10px; color: var(--muted); font-size: 0.8rem; }

.settings { display: grid; grid-template-columns: max-content 1fr; gap: 6px 14px; margin: 0; font-size: 0.85rem; }
.settings div { display: contents; }
.settings dt { color: var(--muted); }
.settings dd { margin: 0; overflow-wrap: anywhere; }

.stage-list { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.stage { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.stage-top { display: flex; justify-content: space-between; gap: 8px; font-size: 0.85rem; }
.stage-desc { margin: 3px 0 0; font-size: 0.8rem; color: var(--muted); }
.stage-src { margin: 3px 0 0; font-size: 0.74rem; color: var(--muted); overflow-wrap: anywhere; }

.raw { margin-top: 12px; font-size: 0.8rem; }
.raw summary { cursor: pointer; color: var(--muted); }
.raw pre { margin: 8px 0 0; padding: 10px; background: var(--bg); border: 1px solid var(--border);
  border-radius: 6px; overflow-x: auto; font-size: 0.75rem; max-height: 320px; }

.col-group { margin-top: 12px; }
.group-head { display: flex; justify-content: space-between; gap: 8px; font-size: 0.85rem; margin-bottom: 4px; }
.obs summary { cursor: pointer; font-size: 0.85rem; }
.cols { list-style: none; margin: 0; padding: 0; display: grid; gap: 3px; }
.col { display: grid; grid-template-columns: minmax(0, 11rem) 1fr 3rem; align-items: center; gap: 8px; font-size: 0.78rem; }
.col-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.fill { display: block; height: 100%; border-radius: 3px; }
.fill.good { background: #3d8b5f; }
.fill.fair { background: #c49a3d; }
.fill.thin { background: #b3492f; }
.pct { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }

.runs { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.run { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; }
.run-top { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; font-size: 0.85rem; }
.run-stages { margin: 6px 0 0; display: flex; gap: 4px; flex-wrap: wrap; }
.run p { margin: 4px 0 0; }
.chip { font-size: 0.72rem; border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px; }

.st-succeeded { color: #3d8b5f; }
.st-failed { color: #b3492f; }
.st-cancelled { color: var(--muted); }
.st-running, .st-queued, .st-pending { color: #8b6b3d; }
.warn { color: #8b6b3d; font-size: 0.78rem; }
.muted { color: var(--muted); }
.small { font-size: 0.76rem; }
.tag { font-size: 0.7rem; border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px; color: var(--muted); }
.vis-badge { font-size: 0.7rem; border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px; color: var(--muted); }
.vis-badge.members { border-color: #3d8b5f; color: #3d8b5f; }
.vis-badge.public { border-color: #8b5f3d; color: #8b5f3d; }

.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 9px; font: inherit; font-size: 0.78rem; cursor: pointer; text-decoration: none; display: inline-block; }
.btn:hover { border-color: var(--muted); }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
.msg { margin: 8px 0 0; font-size: 0.82rem; color: var(--muted); }
.msg.error { color: #b3492f; }

@media (max-width: 520px) {
  .ds-page { padding: 12px; }
  .col { grid-template-columns: minmax(0, 8.5rem) 1fr 2.6rem; }
}
</style>
