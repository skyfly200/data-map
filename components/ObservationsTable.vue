<template>
  <div class="table-page">
    <div class="toolbar">
      <input v-model="query" type="search" class="search" placeholder="Filter by species or location…" />
      <span class="count">{{ filtered.length }} / {{ rows.length }} observations</span>
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <div v-else ref="scroller" class="table-wrap" @scroll.passive="onScroll">
      <table>
        <thead>
          <tr>
            <th v-for="col in columns" :key="col.key" :class="{ sortable: col.sortable }" @click="col.sortable && sortBy(col.key)">
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
            <td v-for="col in columns" :key="col.key" :class="col.numeric ? 'num' : ''">
              <template v-if="col.key === 'cluster'">
                <span v-if="hasValue(row.cluster)" class="chip" :style="{ background: colorFor(row.cluster) }">{{ row.cluster }}</span>
                <span v-else class="muted">—</span>
              </template>
              <template v-else-if="col.key === 'species'">
                <em>{{ row.species || '—' }}</em>
              </template>
              <template v-else-if="col.key === 'elevation'">
                {{ hasValue(row.elevation) ? Math.round(elevValue(row.elevation)).toLocaleString() : '—' }}
              </template>
              <template v-else>
                {{ display(col, row[col.key]) }}
              </template>
            </td>
            <td>
              <a v-if="inatUrl(row)" :href="inatUrl(row)" target="_blank" rel="noopener" class="ext">↗</a>
              <span v-else class="muted">—</span>
            </td>
          </tr>
          <tr v-if="padBottom" class="spacer" :style="{ height: `${padBottom}px` }"><td :colspan="columns.length + 1"></td></tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<script setup>
import { colorFor, hasValue, inatUrl, useObservations } from '~/composables/useObservations'
import { useUnits } from '~/composables/useUnits'

const { rows, error, pending, load } = useObservations()
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

function sortBy(key) {
  if (sortKey.value === key) sortDir.value *= -1
  else { sortKey.value = key; sortDir.value = 1 }
}

function display(col, v) {
  if (!hasValue(v)) return '—'
  if (col.numeric && typeof col.round === 'number') return Number(v).toFixed(col.round)
  return v
}

const filtered = computed(() => {
  const q = query.value.trim().toLowerCase()
  let list = rows.value
  if (q) {
    list = list.filter((r) =>
      (r.species || '').toLowerCase().includes(q) ||
      (r.location || '').toLowerCase().includes(q))
  }
  const key = sortKey.value
  const dir = sortDir.value
  return [...list].sort((a, b) => {
    const av = a[key], bv = b[key]
    const aNull = !hasValue(av), bNull = !hasValue(bv)
    if (aNull && bNull) return 0
    if (aNull) return 1              // nulls always sort last
    if (bNull) return -1
    if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir
    return String(av).localeCompare(String(bv)) * dir
  })
})

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

onMounted(() => {
  nextTick(measure)
  if (import.meta.client) {
    window.addEventListener('resize', measure, { passive: true })
  }
})
onUnmounted(() => {
  if (import.meta.client) window.removeEventListener('resize', measure)
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
</style>
