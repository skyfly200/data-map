<template>
  <PopoverMenu icon="⌕" label="Filter" title="Narrow what the map draws"
               :active="!!activeCount" :badge="activeCount ? String(activeCount) : ''">
    <!-- Search first, because it is the one you reach for before you know what
         you are looking for. -->
    <div class="pop-field">
      <label :for="`${uid}-q`">Search</label>
      <input :id="`${uid}-q`" v-model="search" type="search"
             placeholder="A name, or a place" />
    </div>

    <div class="pop-field">
      <label :for="`${uid}-taxon`">Species or genus</label>
      <select :id="`${uid}-taxon`" v-model="taxon">
        <option value="">Any</option>
        <optgroup v-for="rank in ranks" :key="rank.key" :label="rank.label">
          <option v-for="t in rank.taxa.slice(0, TAXON_CAP)" :key="`${rank.key}:${t.name}`"
                  :value="t.name">
            {{ t.name }} ({{ t.count.toLocaleString() }})
          </option>
        </optgroup>
      </select>
    </div>

    <!-- Two numbers rather than a two-handled slider: the useful question is
         usually "above the treeline" or "below 8,000 ft", which is one end. -->
    <div v-if="elevRange" class="pop-field">
      <label :for="`${uid}-elev-min`">
        Elevation
        <span class="mf-unit">{{ unit === 'm' ? 'metres' : 'feet' }}</span>
      </label>
      <div class="mf-range">
        <input :id="`${uid}-elev-min`" v-model="elevMinShown" type="number"
               :placeholder="String(elevRange.min)" :step="unit === 'm' ? 50 : 100"
               aria-label="Lowest elevation" />
        <span class="mf-dash">to</span>
        <input v-model="elevMaxShown" type="number"
               :placeholder="String(elevRange.max)" :step="unit === 'm' ? 50 : 100"
               aria-label="Highest elevation" />
      </div>
      <p class="mf-note">
        The loaded data runs {{ elevRange.min.toLocaleString() }}–{{ elevRange.max.toLocaleString() }}.
        Records with no elevation are left out when either end is set.
      </p>
    </div>

    <p class="mf-count">
      <strong>{{ shownCount.toLocaleString() }}</strong> of
      {{ totalCount.toLocaleString() }} shown
      <button v-if="activeCount" type="button" class="linkish" @click="clearMine">Clear</button>
    </p>
    <p v-if="activeCount && !shownCount" class="mf-empty">
      Nothing matches. The points are still loaded — widen a filter to see them.
    </p>

    <p class="mf-more">
      <NuxtLink to="/data">Dates, places and more on Data →</NuxtLink>
    </p>
  </PopoverMenu>
</template>

<script setup>
// Narrowing the map without leaving it.
//
// The filters have always been on the Data tab, which is the right home for the
// full set — dates, places, saved subsets. But the three you reach for WHILE
// looking at the map are a search, a taxon and an elevation band, and going to
// another page to set them means losing the view you were looking at.
//
// So this is the same filter state, not a second one: it writes to useFilters,
// which every view already reads. Setting a taxon here and opening the table
// shows the same records.

import { computed, useId } from 'vue'

import { useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'
import { taxaInFeatures } from '~/netlify/lib/dataset-taxa.mjs'

/** Per rank. A select with four hundred options is not a picker. */
const TAXON_CAP = 150

const uid = useId()
const { filters, setFilter } = useFilters()
const { data, filteredData } = useObservations()
const { unit } = useUnits()

const totalCount = computed(() => data.value?.features?.length || 0)
const shownCount = computed(() => filteredData.value?.features?.length || 0)

// Offered from what is loaded, so every name in the list selects something.
// Counted off the unfiltered set: a list that shrank as you used it would stop
// you widening a filter you had just narrowed too far.
const ranks = computed(() => taxaInFeatures(data.value?.features || [], { min: 1 }))

const bound = (key) => computed({
  get: () => filters.value[key],
  set: (v) => setFilter(key, v),
})
const search = bound('search')
const taxon = bound('taxon')

// ─── Elevation ───────────────────────────────────────────────────────────────
// Stored in metres and shown in the viewer's unit. The conversion is here
// rather than in the filter, so switching to feet re-labels the same band
// instead of moving it.

const M_PER_FT = 0.3048
const toShown = (m) => (m == null ? null : Math.round(unit.value === 'm' ? m : m / M_PER_FT))
const toMetres = (v) => {
  const n = Number(v)
  if (!Number.isFinite(n)) return null
  return Math.round(unit.value === 'm' ? n : n * M_PER_FT)
}

/** The range the loaded data actually covers, in the viewer's unit. */
const elevRange = computed(() => {
  let lo = Infinity
  let hi = -Infinity
  for (const f of data.value?.features || []) {
    const e = Number(f.properties?.elevation)
    if (!Number.isFinite(e)) continue
    if (e < lo) lo = e
    if (e > hi) hi = e
  }
  if (!Number.isFinite(lo)) return null
  return { min: toShown(lo), max: toShown(hi) }
})

const elevMinShown = computed({
  get: () => toShown(filters.value.elevMin),
  set: (v) => setFilter('elevMin', v === '' || v == null ? null : toMetres(v)),
})
const elevMaxShown = computed({
  get: () => toShown(filters.value.elevMax),
  set: (v) => setFilter('elevMax', v === '' || v == null ? null : toMetres(v)),
})

/** How many of THESE filters are set. The bar's badge counts every filter. */
const activeCount = computed(() => {
  const f = filters.value
  let n = 0
  if (f.search) n += 1
  if (f.taxon) n += 1
  if (f.elevMin != null || f.elevMax != null) n += 1
  return n
})

// Clears what this panel sets and nothing else: the dates and places set on the
// Data tab are not this control's to throw away.
function clearMine() {
  setFilter('search', '')
  setFilter('taxon', '')
  setFilter('elevMin', null)
  setFilter('elevMax', null)
}
</script>

<style scoped>
.pop-field { display: grid; gap: 4px; }
.pop-field label {
  display: flex; align-items: baseline; gap: 6px;
  font-size: 0.74rem; color: var(--muted); font-weight: 600;
}
.pop-field input, .pop-field select {
  width: 100%; box-sizing: border-box;
  background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 6px; font: inherit; font-size: 0.8rem;
}
.mf-unit { font-weight: 400; font-size: 0.7rem; }

.mf-range { display: flex; align-items: center; gap: 6px; }
.mf-range input { flex: 1 1 0; min-width: 0; }
.mf-dash { flex: 0 0 auto; font-size: 0.72rem; color: var(--muted); }
.mf-note { margin: 2px 0 0; font-size: 0.68rem; color: var(--muted); line-height: 1.45; }

.mf-count {
  margin: 2px 0 0; font-size: 0.76rem; color: var(--muted);
  display: flex; align-items: center; gap: 8px;
  border-top: 1px solid var(--border-soft, var(--border)); padding-top: 7px;
}
.mf-count strong { color: var(--text-strong); }
.linkish {
  background: none; border: 0; padding: 0; cursor: pointer;
  font: inherit; font-size: 0.76rem; color: var(--accent); text-decoration: underline;
}
.mf-empty { margin: 0; font-size: 0.72rem; color: #e0714f; line-height: 1.45; }
.mf-more { margin: 0; font-size: 0.72rem; }
.mf-more a { color: var(--accent); text-decoration: none; }
.mf-more a:hover { text-decoration: underline; }
</style>
