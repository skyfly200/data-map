<template>
  <div class="table-page">
    <div class="toolbar">
      <input v-model="query" type="search" class="search" placeholder="Filter by species or location…" aria-label="Filter observations by species or location" />
      <span class="count">{{ filtered.length }} / {{ rows.length }} observations</span>
      <!-- Exports what the table is showing, filters and search included —
           which is the set somebody has just finished deciding they wanted. -->
      <ExportMenu :source="filteredFeatures" :name="exportName" :shown="exportColumns"
                  :subtitle="filtered.length < rows.length ? 'matching the current filters' : ''" />
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <div v-else ref="scroller" class="table-wrap" @scroll.passive="onScroll">
      <table aria-label="Observations">
        <thead>
          <tr>
            <th v-for="col in columns" :key="col.key"
                :class="[{ sortable: col.sortable }, colClass(col.key)]"
                @click="col.sortable && sortBy(col.key)">
              {{ col.label }}
              <span v-if="sortKey === col.key" class="arrow">{{ sortDir === 1 ? '▲' : '▼' }}</span>
            </th>
            <th>iNat</th>
          </tr>
        </thead>
        <tbody>
          <!-- Only the rows in view exist in the DOM; these spacers stand in for
               the ones above and below so the scrollbar still spans the full set. -->
          <tr v-if="padTop" class="spacer" :style="{ height: `${padTop}px` }"><td :colspan="columns.length + 1"></td></tr>
          <tr v-for="(row, i) in visibleRows" :key="row.uuid || start + i" ref="rowEls">
            <td v-for="col in columns" :key="col.key" :class="[col.numeric ? 'num' : '', colClass(col.key)]">
              <template v-if="col.key === 'cluster'">
                <span v-if="hasValue(row.cluster)" class="chip" :style="{ background: colorFor(row.cluster) }">{{ row.cluster }}</span>
                <span v-else class="muted">, </span>
              </template>
              <template v-else-if="col.key === 'species'">
                <em>{{ row.species || ', ' }}</em>
              </template>
              <template v-else-if="col.key === 'elevation'">
                {{ hasValue(row.elevation) ? (v => Number.isFinite(v) ? v.toLocaleString() : '—')(Math.round(elevValue(row.elevation))) : '—' }}
              </template>
              <template v-else>
                {{ display(col, row[col.key]) }}
              </template>
            </td>
            <td>
              <a v-if="inatUrl(row)" :href="inatUrl(row)" target="_blank" rel="noopener" class="ext">↗</a>
              <span v-else class="muted">: </span>
            </td>
          </tr>
          <tr v-if="padBottom" class="spacer" :style="{ height: `${padBottom}px` }"><td :colspan="columns.length + 1"></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { hasValue, inatUrl, useObservations } from '~/composables/useObservations'
import { colorFor } from '~/composables/useAppearance'
import { useUnits } from '~/composables/useUnits'

const { rows, filteredData, error, pending, load, selectedDataset, availableDatasets }
  = useObservations()
const { unit, elevValue } = useUnits()
onMounted(load)

const columns = computed(() => [
  { key: 'species', label: 'Species', sortable: true },
  { key: 'date', label: 'Observed', sortable: true },
  { key: 'day_of_year', label: 'Day of yr', sortable: true, numeric: true },
  { key: 'location', label: 'Location', sortable: true },
  { key: 'elevation', label: `Elev (${unit.value})`, sortable: true, numeric: true },
  { key: 'land_cover_label', label: 'Land cover', sortable: true },
  { key: 'cluster', label: 'Cluster', sortable: true, numeric: true },
  { key: 'ndvi', label: 'NDVI', sortable: true, numeric: true, round: 3 },
  { key: 'soil_moisture', label: 'Soil moist.', sortable: true, numeric: true, round: 3 },
  { key: 'solar_exposure', label: 'Solar', sortable: true, numeric: true, round: 2 },
  { key: 'wind_exposure', label: 'Wind', sortable: true, numeric: true, round: 2 },
  { key: 'water_retention', label: 'Water ret.', sortable: true, numeric: true, round: 2 },
])

const query = ref('')
const sortKey = ref('date')
const sortDir = ref(-1) // -1 desc, 1 asc

// What the export offers as its default column set: the table's own columns.
// `species` and `location` are properties; `elevation` and the rest are too,
// so the keys map straight across apart from the two the table synthesises.
const exportColumns = computed(() => columns.value
  .map((c) => c.key)
  .filter((k) => k !== 'cluster'))

const exportName = computed(() => {
  const label = availableDatasets.value.find((d) => d.path === selectedDataset.value)?.label
  // Dataset labels carry their record count — "All genuss (48233)" — which is
  // useful in a picker and noise in a filename, especially once the export has
  // been filtered down to some other number.
  return (label || 'observations').replace(/\s*\(\d[\d,]*\)\s*$/, '') || 'observations'
})

function sortBy(key) {
  if (sortKey.value === key) sortDir.value *= -1
  else { sortKey.value = key; sortDir.value = 1 }
}

const COL_CSS = {
  solar_exposure: 'col-solar', wind_exposure: 'col-wind',
  water_retention: 'col-water', cluster: 'col-cluster',
  land_cover_label: 'col-land-cover', ndvi: 'col-ndvi',
  soil_moisture: 'col-soil', day_of_year: 'col-day',
}
function colClass(key) { return COL_CSS[key] ?? '' }

function display(col, v) {
  if (!hasValue(v)) return ', '
  if (col.numeric && typeof col.round === 'number') { const n = Number(v); return Number.isFinite(n) ? n.toFixed(col.round) : '—' }
  return v
}

/**
 * The search and sort, over features rather than rows.
 *
 * Features first and rows derived from them, so that exporting what the table
 * is showing is the same set the table is showing rather than a second
 * pipeline that has to be kept in step. A row is a flattened feature; going the
 * other way would mean rebuilding a geometry from two columns.
 */
const filteredFeatures = computed(() => {
  const q = query.value.trim().toLowerCase()
  let list = filteredData.value?.features || []
  if (q) {
    list = list.filter((f) => {
      const p = f.properties || {}
      return (p.species || '').toLowerCase().includes(q)
        || (p.location || '').toLowerCase().includes(q)
    })
  }
  const key = sortKey.value
  const dir = sortDir.value
  return [...list].sort((fa, fb) => {
    const av = fa.properties?.[key], bv = fb.properties?.[key]
    const aNull = !hasValue(av), bNull = !hasValue(bv)
    if (aNull && bNull) return 0
    if (aNull) return 1              // nulls always sort last
    if (bNull) return -1
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir
    return String(av).localeCompare(String(bv)) * dir
  })
})

const filtered = computed(() => filteredFeatures.value.map((f) => ({
  ...f.properties,
  lon: f.geometry?.coordinates?.[0],
  lat: f.geometry?.coordinates?.[1],
})))

// ─── Windowed rendering ───────────────────────────────────────────────────────
// The full set is ~48k rows. Putting all of them in the DOM took ~30s to become
// interactive and made scrolling unusable, so only the rows on screen are
// rendered and two spacer rows carry the rest of the scroll height.

const ROW_HEIGHT = 29   // measured from a rendered row; a starting estimate
const OVERSCAN = 8      // rows kept beyond each edge, so a fast scroll stays filled

const scroller = ref(null)
const rowEls = ref([])
const scrollTop = ref(0)
const viewportHeight = ref(600)
const rowHeight = ref(ROW_HEIGHT)

const total = computed(() => filtered.value.length)
const start = computed(() =>
  Math.max(0, Math.floor(scrollTop.value / rowHeight.value) - OVERSCAN))
const count = computed(() =>
  Math.ceil(viewportHeight.value / rowHeight.value) + OVERSCAN * 2)
const end = computed(() => Math.min(total.value, start.value + count.value))

const visibleRows = computed(() => filtered.value.slice(start.value, end.value))
const padTop = computed(() => start.value * rowHeight.value)
const padBottom = computed(() => Math.max(0, (total.value - end.value) * rowHeight.value))

function onScroll(e) {
  scrollTop.value = e.target.scrollTop
}

function measure() {
  const el = scroller.value
  if (el) viewportHeight.value = el.clientHeight || viewportHeight.value
  // Trust a real rendered row over the estimate, so the spacers match the
  // content exactly and scrolling does not drift.
  const row = rowEls.value?.[0]
  const h = row?.offsetHeight
  if (h && Math.abs(h - rowHeight.value) > 0.5) rowHeight.value = h
}

// Filtering or re-sorting changes what row 0 is; jump back to the top so the
// window and the scroll position agree.
watch([query, sortKey, sortDir], () => {
  scrollTop.value = 0
  if (scroller.value) scroller.value.scrollTop = 0
})

watch(visibleRows, () => nextTick(measure))

let ro = null

onMounted(() => {
  nextTick(() => {
    measure()
    if (import.meta.client && scroller.value && typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(measure)
      ro.observe(scroller.value)
    }
  })
})
onUnmounted(() => {
  ro?.disconnect()
  ro = null
})
</script>

<style scoped>
.table-page { padding: 14px 18px; }

.toolbar { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; }
.search {
  flex: 0 1 320px; padding: 7px 11px; border: 1px solid var(--border); border-radius: 7px;
  font-size: 0.9rem;
}
.count { color: var(--muted); font-size: 0.85rem; }

.msg { color: var(--muted); }
.msg.error { color: var(--danger); }

/* Bounded height so the windowed body has a viewport to scroll inside. */
.table-wrap {
  overflow: auto; border: 1px solid var(--border); border-radius: 8px;
  max-height: calc(100vh - 240px); min-height: 320px;
}
/* Spacer rows stand in for the off-screen rows; they must not pick up row
   borders or hover styling. */
tbody tr.spacer { background: none; }
tbody tr.spacer:hover { background: none; }
tbody tr.spacer td { padding: 0; border: 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
thead th {
  position: sticky; top: 0; background: var(--surface-2); text-align: left;
  padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; color: var(--text);
}
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { background: var(--surface-3); }
.arrow { font-size: 0.7rem; color: var(--muted); }
tbody td { padding: 6px 10px; border-bottom: 1px solid var(--border-soft); white-space: nowrap; }
tbody tr:hover { background: var(--surface-2); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }

.chip {
  display: inline-block; min-width: 20px; padding: 1px 7px; border-radius: 10px;
  color: #fff; font-weight: 600; text-align: center; font-size: 0.78rem;
}
.muted { color: var(--muted); }
.ext { text-decoration: none; color: var(--accent); font-weight: 700; }

/* ─── Responsive: tablet ──────────────────────────────────────────────────
   Hide low-priority columns at tablet width to avoid horizontal scroll. */
@media (max-width: 900px) {
  .col-solar, .col-wind, .col-water, .col-cluster { display: none; }
}

/* ─── Responsive: phone ───────────────────────────────────────────────────
   Pin the species column and hide most enrichment columns. The key facts
   (species, date, location, elevation) stay visible; everything else is gone. */
@media (max-width: 600px) {
  .toolbar { flex-wrap: wrap; }
  .search { flex: 1 1 100%; }

  .col-land-cover, .col-ndvi, .col-soil, .col-day { display: none; }

  /* Pinned first column (species): sticky left so it stays visible while
     scrolling the narrower set of remaining columns horizontally. */
  thead th:first-child,
  tbody td:first-child { position: sticky; left: 0; z-index: 2; background: var(--surface-2); }
  tbody tr:hover td:first-child { background: var(--surface-3, var(--surface-2)); }
  tbody tr.spacer td:first-child { background: none; z-index: auto; }
}
</style>
