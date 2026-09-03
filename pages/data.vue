<template>
  <div class="data-page">
    <nav class="tabs">
      <button :class="{ on: tab === 'species' }" @click="tab = 'species'">Species</button>
      <button :class="{ on: tab === 'table' }" @click="tab = 'table'">Table</button>
      <button :class="{ on: tab === 'fetch' }" @click="tab = 'fetch'">Fetch new</button>
    </nav>

    <div class="dataset-bar">
      <label for="dataset-select">Dataset</label>
      <select id="dataset-select" :value="selectedDataset" @change="setDataset($event.target.value)">
        <option v-for="d in availableDatasets" :key="d.id" :value="d.path">{{ d.label }}</option>
      </select>
    </div>

    <div class="layout">
      <aside class="side">
        <FilterPanel />
      </aside>
      <section class="main">
        <!-- ── Fetch a new species ────────────────────────────────────── -->
        <FetchSpecies v-if="tab === 'fetch'" />

        <!-- ── Full observation table ─────────────────────────────────── -->
        <ObservationsTable v-else-if="tab === 'table'" />

        <!-- ── Taxon selection, at any rank ──────────────────────────── -->
        <template v-else>
          <div class="head">
            <div>
              <h2>{{ levelLabel }}</h2>
              <p class="sub">Choose which {{ levelLabel.toLowerCase() }} to show across the map, table, and charts.</p>
            </div>
            <div class="actions">
              <label class="level">
                <span>Rank</span>
                <select :value="level" @change="setLevel($event.target.value)">
                  <option v-for="r in ranks" :key="r.key" :value="r.key">{{ r.label }}</option>
                </select>
              </label>
              <span class="count">{{ selectedGroups }} / {{ groupedOptions.length }} selected</span>
              <button @click="selectAll">All</button>
              <button @click="clearAll">None</button>
            </div>
          </div>

          <div v-if="speciesOptions.length" class="species-search">
            <input v-model="speciesQuery" type="search" :placeholder="`Search ${levelLabel.toLowerCase()}…`" :aria-label="`Search ${levelLabel.toLowerCase()}`" />
            <span v-if="speciesQuery" class="found">{{ visibleGroups.length }} match{{ visibleGroups.length === 1 ? '' : 'es' }}</span>
          </div>

          <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
          <p v-else-if="pending && !speciesOptions.length" class="msg">Loading…</p>
          <p v-else-if="!speciesOptions.length" class="msg">No {{ levelLabel.toLowerCase() }} in the current dataset.</p>
          <p v-else-if="!visibleGroups.length" class="msg">No {{ levelLabel.toLowerCase() }} match “{{ speciesQuery }}”.</p>

          <div v-else class="table-wrap">
            <table>
              <thead>
                <tr>
                  <th class="c-check"><input type="checkbox" :checked="allChecked" :indeterminate.prop="someChecked" @change="toggleAll" /></th>
                  <th>{{ levelLabel }}</th>
                  <th v-if="level !== 'species'" class="c-num" title="Distinct species recorded under this group">Species</th>
                  <th class="c-num">Observations</th>
                  <th class="c-bar"></th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="opt in visibleGroups" :key="opt.key" :class="{ off: !selected.has(opt.key) }" @click="toggleGroup(opt)">
                  <td class="c-check"><input type="checkbox" :checked="selected.has(opt.key)" @click.stop="toggleGroup(opt)" /></td>
                  <td class="sp"><em>{{ opt.key }}</em></td>
                  <td v-if="level !== 'species'" class="c-num">{{ opt.species }}</td>
                  <td class="c-num">{{ opt.count }}</td>
                  <td class="c-bar"><span class="bar" :style="{ width: barWidth(opt.count) }"></span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </template>
      </section>
    </div>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'

const { data, speciesOptions, speciesFilter, setSpeciesFilter, error, pending, load,
  selectedDataset, availableDatasets, setDataset,
  taxonRank, setTaxonRank, availableRanks } = useObservations()
onMounted(load)

// Tabs (in the URL so /data?tab=table deep-links, and old /table redirects here).
const route = useRoute()
const router = useRouter()
const TABS = ['species', 'table', 'fetch']
const tab = computed({
  get: () => (TABS.includes(route.query.tab) ? route.query.tab : 'species'),
  set: (v) => router.replace({ query: { ...route.query, tab: v } }),
})

// Which rank to filter at. This used to derive genus and subspecies by
// splitting the species binomial on its spaces, which works only when a record
// happens to be identified to species and cannot reach family or above at all.
// The pipeline now resolves the real ancestry, so the rank is a column and the
// filter is applied at whichever one is chosen — the same picker narrows to a
// kingdom or to one species.
//
// Only the ranks the loaded dataset actually populates are offered; a dataset
// exported before the taxonomy work carries species and genus and nothing else.
const LEVEL_KEY = 'data-taxon-level'
const ranks = computed(() => (availableRanks.value.length
  ? availableRanks.value
  : [{ key: 'species', label: 'Species' }]))
const level = computed(() => taxonRank.value)
const levelLabel = computed(() =>
  ranks.value.find((r) => r.key === level.value)?.label || 'Species')

function setLevel(rank) {
  setTaxonRank(rank)
  if (import.meta.client) localStorage.setItem(LEVEL_KEY, rank)
}
onMounted(() => {
  const saved = import.meta.client ? localStorage.getItem(LEVEL_KEY) : null
  if (saved && saved !== taxonRank.value) setTaxonRank(saved)
})
// A dataset that does not carry the remembered rank would otherwise leave the
// picker on a column of blanks.
watch(ranks, (list) => {
  if (list.length && !list.some((r) => r.key === level.value)) setTaxonRank(list.at(-1).key)
})

// How many distinct species sit under each value at the chosen rank. At species
// level that column is the row itself, so it is not shown.
const speciesUnder = computed(() => {
  const rank = level.value
  if (rank === 'species') return new Map()
  const out = new Map()
  for (const f of data.value?.features || []) {
    const key = f.properties?.[rank]
    const sp = f.properties?.species
    if (!key || !sp) continue
    if (!out.has(key)) out.set(key, new Set())
    out.get(key).add(sp)
  }
  return out
})

const groupedOptions = computed(() => speciesOptions.value.map((o) => ({
  key: o.species, count: o.count, species: speciesUnder.value.get(o.species)?.size ?? 1,
})))

// Local selection mirrors the global filter. Empty filter == all species shown.
const selected = ref(new Set())

function syncFromFilter() {
  const all = speciesOptions.value.map((o) => o.species)
  selected.value = new Set(speciesFilter.value.length ? speciesFilter.value : all)
}
watch(speciesOptions, syncFromFilter, { immediate: true })

function commit() {
  const all = speciesOptions.value.map((o) => o.species)
  setSpeciesFilter(selected.value.size === all.length ? [] : [...selected.value])
}

// One row is now one value at the chosen rank, so a row is simply on or off —
// the tri-state this had was only there because a genus row stood for a set of
// species names that could be partly selected.
function toggleGroup(group) {
  const s = new Set(selected.value)
  if (s.has(group.key)) s.delete(group.key); else s.add(group.key)
  selected.value = s
  commit()
}
function selectAll() { selected.value = new Set(speciesOptions.value.map((o) => o.species)); commit() }
function clearAll() { selected.value = new Set(); commit() }
function toggleAll() { allChecked.value ? clearAll() : selectAll() }

const speciesQuery = ref('')
const visibleGroups = computed(() => {
  const q = speciesQuery.value.trim().toLowerCase()
  if (!q) return groupedOptions.value
  return groupedOptions.value.filter((g) => g.key.toLowerCase().includes(q))
})

const selectedGroups = computed(() => groupedOptions.value.filter((g) => selected.value.has(g.key)).length)
const allChecked = computed(() => selected.value.size === speciesOptions.value.length && speciesOptions.value.length > 0)
const someChecked = computed(() => selected.value.size > 0 && !allChecked.value)

const maxCount = computed(() => Math.max(1, ...groupedOptions.value.map((o) => o.count)))
function barWidth(n) { return `${(n / maxCount.value) * 100}%` }
</script>

<style scoped>
.data-page { padding: 16px 18px; max-width: 1200px; margin: 0 auto; }
.tabs { display: flex; gap: 4px; margin: -4px 0 14px; border-bottom: 1px solid var(--border); }
.tabs button {
  border: 0; background: transparent; color: var(--muted); cursor: pointer;
  padding: 8px 16px; font-size: 0.92rem; font-weight: 600; border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tabs button:hover { color: var(--text); }
.tabs button.on { color: var(--text); border-bottom-color: var(--accent); }

.dataset-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 14px; font-size: 0.82rem; color: var(--muted); }
.dataset-bar label { font-weight: 600; }
.dataset-bar select {
  flex: 0 1 320px; min-width: 0; border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 9px; font-size: 0.84rem; background: var(--input-bg); color: var(--text);
}

.layout { display: grid; grid-template-columns: minmax(300px, 360px) 1fr; gap: 22px; align-items: start; }
.side { position: sticky; top: 16px; }
.main { min-width: 0; }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .side { position: static; }
}
.head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 0; color: var(--muted); font-size: 0.82rem; }
.actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.level { display: inline-flex; align-items: center; gap: 6px; font-size: 0.82rem; color: var(--muted); }
.level select { border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; font-size: 0.82rem; background: var(--input-bg); color: var(--text); }
.count { color: var(--muted); font-size: 0.82rem; }
.actions button { border: 1px solid var(--border); background: var(--surface); border-radius: 6px; padding: 4px 10px; font-size: 0.82rem; cursor: pointer; }
.actions button:hover { background: var(--surface-2); }

.table-wrap { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
thead th { background: var(--surface-2); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text); }
tbody td { padding: 7px 10px; border-bottom: 1px solid var(--border-soft); }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--surface-2); }
tbody tr.off { color: var(--muted); }
tbody tr.off .sp em { color: var(--muted); }
.c-check { width: 34px; text-align: center; }
.c-num { text-align: right; width: 90px; font-variant-numeric: tabular-nums; }
.c-bar { width: 140px; }
.bar { display: block; height: 8px; background: #2a78d6; border-radius: 4px; }
tr.off .bar { background: #cbd5e1; }

.msg { padding: 16px; color: var(--muted); }
.msg.error { color: var(--danger); }

.species-search { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.species-search input {
  flex: 1 1 auto; border: 1px solid var(--border); border-radius: 8px; padding: 7px 11px;
  font-size: 0.88rem; background: var(--input-bg); color: var(--text);
}
.species-search .found { font-size: 0.8rem; color: var(--muted); white-space: nowrap; }
</style>
