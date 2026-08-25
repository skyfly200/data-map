<template>
  <div class="data-page">
    <div class="head">
      <div>
        <h2>Species</h2>
        <p class="sub">Choose which species to show across the map, table, charts, and explorer.</p>
      </div>
      <div class="actions">
        <span class="count">{{ selected.size }} / {{ speciesOptions.length }} selected</span>
        <button @click="selectAll">All</button>
        <button @click="clearAll">None</button>
      </div>
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !speciesOptions.length" class="msg">Loading…</p>
    <p v-else-if="!speciesOptions.length" class="msg">No species in the current dataset.</p>

    <div v-else class="table-wrap">
      <table>
        <thead>
          <tr>
            <th class="c-check"><input type="checkbox" :checked="allChecked" :indeterminate.prop="someChecked" @change="toggleAll" /></th>
            <th>Species</th>
            <th class="c-num">Observations</th>
            <th class="c-bar"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="opt in speciesOptions" :key="opt.species" :class="{ off: !selected.has(opt.species) }" @click="toggle(opt.species)">
            <td class="c-check"><input type="checkbox" :checked="selected.has(opt.species)" @click.stop="toggle(opt.species)" /></td>
            <td class="sp"><em>{{ opt.species }}</em></td>
            <td class="c-num">{{ opt.count }}</td>
            <td class="c-bar"><span class="bar" :style="{ width: barWidth(opt.count) }"></span></td>
          </tr>
        </tbody>
      </table>
    </div>

    <p class="hint">
      Tip: select <strong>All species</strong> in the Dataset menu to pick from every fetched species.
      To search for a <em>new</em> species, add it to the pipeline (<code>INAT_TAXON_NAME</code>) and re-run.
    </p>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'

const { speciesOptions, speciesFilter, setSpeciesFilter, error, pending, load } = useObservations()
onMounted(load)

// Local selection mirrors the global filter. Empty filter == all species shown.
const selected = ref(new Set())

function syncFromFilter() {
  const all = speciesOptions.value.map((o) => o.species)
  selected.value = new Set(speciesFilter.value.length ? speciesFilter.value : all)
}
watch(speciesOptions, syncFromFilter, { immediate: true })

function commit() {
  const all = speciesOptions.value.map((o) => o.species)
  // All selected → clear the filter (== show everything); otherwise store the subset.
  setSpeciesFilter(selected.value.size === all.length ? [] : [...selected.value])
}
function toggle(species) {
  const s = new Set(selected.value)
  s.has(species) ? s.delete(species) : s.add(species)
  selected.value = s
  commit()
}
function selectAll() { selected.value = new Set(speciesOptions.value.map((o) => o.species)); commit() }
function clearAll() { selected.value = new Set(); commit() }
function toggleAll() { allChecked.value ? clearAll() : selectAll() }

const allChecked = computed(() => selected.value.size === speciesOptions.value.length && speciesOptions.value.length > 0)
const someChecked = computed(() => selected.value.size > 0 && !allChecked.value)

const maxCount = computed(() => Math.max(1, ...speciesOptions.value.map((o) => o.count)))
function barWidth(n) { return `${(n / maxCount.value) * 100}%` }
</script>

<style scoped>
.data-page { padding: 16px 18px; max-width: 720px; }
.head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 0; color: #6b7280; font-size: 0.82rem; }
.actions { display: flex; align-items: center; gap: 8px; }
.count { color: #6b7280; font-size: 0.82rem; }
.actions button { border: 1px solid #cbd2d9; background: #fff; border-radius: 6px; padding: 4px 10px; font-size: 0.82rem; cursor: pointer; }
.actions button:hover { background: #f3f4f6; }

.table-wrap { border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
thead th { background: #f3f4f6; text-align: left; padding: 8px 10px; border-bottom: 1px solid #e5e7eb; color: #374151; }
tbody td { padding: 7px 10px; border-bottom: 1px solid #f1f2f4; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: #fafbfc; }
tbody tr.off { color: #b0b6be; }
tbody tr.off .sp em { color: #b0b6be; }
.c-check { width: 34px; text-align: center; }
.c-num { text-align: right; width: 90px; font-variant-numeric: tabular-nums; }
.c-bar { width: 140px; }
.bar { display: block; height: 8px; background: #2a78d6; border-radius: 4px; }
tr.off .bar { background: #cbd5e1; }

.hint { margin-top: 12px; font-size: 0.8rem; color: #6b7280; }
.hint code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }
.msg { padding: 16px; color: #555; }
.msg.error { color: #b00020; }
</style>
