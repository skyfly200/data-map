<template>
  <div class="table-page">
    <div class="toolbar">
      <input v-model="query" type="search" class="search" placeholder="Filter by species or location…" />
      <span class="count">{{ filtered.length }} / {{ rows.length }} observations</span>
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !rows.length" class="msg">Loading…</p>

    <div v-else class="table-wrap">
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
          <tr v-for="(row, i) in filtered" :key="row.uuid || i">
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
</script>

<style scoped>
.table-page { padding: 14px 18px; }

.toolbar { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; }
.search {
  flex: 0 1 320px; padding: 7px 11px; border: 1px solid #cbd2d9; border-radius: 7px;
  font-size: 0.9rem;
}
.count { color: #6b7280; font-size: 0.85rem; }

.msg { color: #555; }
.msg.error { color: #b00020; }

.table-wrap { overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 0.85rem; }
thead th {
  position: sticky; top: 0; background: #f3f4f6; text-align: left;
  padding: 8px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; color: #374151;
}
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { background: #e9ebee; }
.arrow { font-size: 0.7rem; color: #6b7280; }
tbody td { padding: 6px 10px; border-bottom: 1px solid #f1f2f4; white-space: nowrap; }
tbody tr:hover { background: #fafbfc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }

.chip {
  display: inline-block; min-width: 20px; padding: 1px 7px; border-radius: 10px;
  color: #fff; font-weight: 600; text-align: center; font-size: 0.78rem;
}
.muted { color: #b0b6be; }
.ext { text-decoration: none; color: #2b7a3d; font-weight: 700; }
</style>
