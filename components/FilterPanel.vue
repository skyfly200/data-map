<template>
  <div class="filter-panel">
    <div class="fp-head">
      <h3>Filters <span v-if="activeCount" class="badge">{{ activeCount }}</span></h3>
      <div class="fp-actions">
        <span class="showing">Showing {{ shownCount }} of {{ totalCount }}</span>
        <button class="reset" :disabled="!activeCount" @click="reset">Reset</button>
      </div>
    </div>

    <div class="groups">
      <!-- Location: admin -->
      <fieldset>
        <legend>Location</legend>
        <label>Country
          <select :value="filters.country" @change="setFilter('country', $event.target.value)">
            <option value="">Any</option>
            <option v-for="c in filterOptions.countries" :key="c" :value="c">{{ c }}</option>
          </select>
        </label>
        <label>State / Province
          <select :value="filters.state" @change="setFilter('state', $event.target.value)">
            <option value="">Any</option>
            <option v-for="s in filterOptions.states" :key="s" :value="s">{{ s }}</option>
          </select>
        </label>
        <label>County
          <select :value="filters.county" @change="setFilter('county', $event.target.value)">
            <option value="">Any</option>
            <option v-for="c in filterOptions.counties" :key="c" :value="c">{{ c }}</option>
          </select>
        </label>
      </fieldset>

      <!-- Location: radius -->
      <fieldset>
        <legend>Within radius</legend>
        <div class="radius-row">
          <label>Lat
            <input type="number" step="0.01" v-model.number="lat" placeholder="40.0" />
          </label>
          <label>Lng
            <input type="number" step="0.01" v-model.number="lng" placeholder="-105.0" />
          </label>
          <label>Radius (km)
            <input type="number" step="1" min="0" v-model.number="radiusKm" placeholder="50" />
          </label>
        </div>
        <div class="radius-actions">
          <button class="mini" @click="applyRadius">Apply radius</button>
          <button class="mini ghost" :disabled="!filters.center" @click="clearRadius">Clear</button>
          <span v-if="filters.center" class="rnote">
            ● {{ filters.center.lat.toFixed(2) }}, {{ filters.center.lng.toFixed(2) }} · {{ filters.radiusKm }} km
          </span>
        </div>
      </fieldset>

      <!-- Time -->
      <fieldset>
        <legend>Time</legend>
        <div class="time-row">
          <label>Year
            <select :value="filters.year" @change="setFilter('year', $event.target.value)">
              <option value="">Any</option>
              <option v-for="y in filterOptions.years" :key="y" :value="y">{{ y }}</option>
            </select>
          </label>
          <label>Month
            <select :value="filters.month" @change="setFilter('month', $event.target.value)">
              <option value="">Any</option>
              <option v-for="(m, i) in MONTHS" :key="i" :value="String(i + 1)">{{ m }}</option>
            </select>
          </label>
          <label>Week
            <input type="number" min="1" max="53" :value="filters.week" placeholder="1–53"
                   @input="setFilter('week', $event.target.value)" />
          </label>
        </div>
        <div class="time-row">
          <label>From
            <input type="date" :value="filters.dateFrom" @change="setFilter('dateFrom', $event.target.value)" />
          </label>
          <label>To
            <input type="date" :value="filters.dateTo" @change="setFilter('dateTo', $event.target.value)" />
          </label>
        </div>
      </fieldset>
    </div>
  </div>
</template>

<script setup>
const { filterOptions, filteredData, data } = useObservations()
const { filters, setFilter, setCenter, reset, activeCount } = useFilters()

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const shownCount = computed(() => filteredData.value?.features?.length || 0)
const totalCount = computed(() => data.value?.features?.length || 0)

// Radius inputs are local until "Apply" so partial typing doesn't refilter.
const lat = ref(filters.value.center?.lat ?? null)
const lng = ref(filters.value.center?.lng ?? null)
const radiusKm = ref(filters.value.radiusKm ?? null)

function applyRadius() {
  if (lat.value == null || lng.value == null || !radiusKm.value) return
  setCenter({ lat: Number(lat.value), lng: Number(lng.value) }, Number(radiusKm.value))
}
function clearRadius() {
  setCenter(null, null)
  lat.value = null; lng.value = null; radiusKm.value = null
}
</script>

<style scoped>
.filter-panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; background: var(--surface); }
.fp-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.fp-head h3 { margin: 0; font-size: 0.95rem; display: flex; align-items: center; gap: 8px; }
.badge { background: #2b7a3d; color: #fff; border-radius: 999px; font-size: 0.7rem; padding: 1px 7px; font-weight: 700; }
.fp-actions { display: flex; align-items: center; gap: 12px; }
.showing { font-size: 0.8rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.reset { border: 1px solid var(--border); background: var(--surface); border-radius: 6px; padding: 4px 12px; font-size: 0.8rem; cursor: pointer; }
.reset:disabled { opacity: 0.5; cursor: default; }
.reset:not(:disabled):hover { background: var(--surface-2); }

.groups { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; }
fieldset { border: 1px solid var(--border-soft); border-radius: 8px; padding: 10px 12px 12px; margin: 0; min-width: 0; }
legend { font-size: 0.78rem; font-weight: 700; color: var(--text); padding: 0 6px; }
label { display: flex; flex-direction: column; gap: 3px; font-size: 0.75rem; font-weight: 600; color: var(--muted); margin-top: 8px; min-width: 0; }
select, input { width: 100%; box-sizing: border-box; border: 1px solid var(--border); border-radius: 6px; padding: 5px 8px; font-size: 0.82rem; font-weight: 400; color: var(--text); background: var(--input-bg); }

.radius-row, .time-row {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px;
}
.radius-row label, .time-row label { margin-top: 0; }
.radius-actions { display: flex; align-items: center; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.mini { border: 1px solid #2b7a3d; background: #2b7a3d; color: #fff; border-radius: 6px; padding: 4px 10px; font-size: 0.78rem; font-weight: 600; cursor: pointer; }
.mini.ghost { background: var(--surface); color: var(--text); border-color: var(--border); }
.mini:disabled { opacity: 0.5; cursor: default; }
.rnote { font-size: 0.76rem; color: #2b7a3d; font-variant-numeric: tabular-nums; }
</style>
