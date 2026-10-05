<template>
  <div class="dash-widget now-fruiting">
    <div class="widget-head">
      <h3 class="widget-title"><span aria-hidden="true">🍄 </span>Now Fruiting</h3>
      <div class="widget-actions">
        <button v-if="selectedSpecies" class="clear-btn" @click="selectedSpecies = null">✕ {{ selectedSpecies }}</button>
        <NuxtLink to="/foray" class="widget-link foray-link" title="Rank places to look for what is in season">Plan a foray ›</NuxtLink>
        <button class="widget-link-btn" @click="goToMap">Map ›</button>
        <button class="widget-link-btn" @click="goToData">Data ›</button>
        <NuxtLink to="/modeling/maxent" class="widget-link">Model ›</NuxtLink>
      </div>
    </div>

    <p class="season-label">{{ seasonLabel }}</p>

    <div v-if="loading" class="nf-loading">Loading…</div>

    <p v-else-if="!topSpecies.length" class="nf-empty">
      No observations found for this time of year.
    </p>

    <ul v-else class="nf-list">
      <li v-for="s in topSpecies" :key="s.name" class="nf-item"
          :class="{ selected: selectedSpecies === s.name, 'has-model': s.hasModel }"
          @click="toggleSpecies(s.name)"
          :title="selectedSpecies === s.name ? 'Click Map › to filter, or click again to deselect' : 'Click to select, then use Map › to filter'">
        <span class="nf-name">{{ s.name }}</span>
        <span class="nf-meta">
          <span v-if="s.hasModel" class="nf-model-badge" title="MaxEnt model available">ML</span>
          <span class="nf-peak" :title="`Peak ${s.peakLabel} from today · ${s.iqr}d IQR`">{{ s.peakLabel }}</span>
          <span class="nf-iqr" :title="`Fruiting window width (IQR): ${s.iqr} days`">{{ s.iqr }}d</span>
          <span v-if="s.elevBand" class="nf-elev"
                :title="`Typical elevation band (IQR): ${elevLabel(s.elevBand.loM)}–${elevLabel(s.elevBand.hiM)}`">
            {{ elevLabel(s.elevBand.loM) }}–{{ elevLabel(s.elevBand.hiM) }}
          </span>
          <span class="nf-count">{{ s.count.toLocaleString() }}</span>
        </span>
      </li>
    </ul>

    <p v-if="latestModel" class="nf-model">
      <span class="nf-model-label">Latest model:</span>
      <NuxtLink :to="mapLink" class="nf-model-link">{{ latestModel.title }}</NuxtLink>
    </p>
    <p v-else-if="topSpecies.length" class="nf-model">
      <NuxtLink to="/modeling/maxent" class="nf-model-link">Train a habitat model ›</NuxtLink>
    </p>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useMaxEnt } from '~/composables/useMaxEnt'
import { useUnits } from '~/composables/useUnits'
import { useFilters } from '~/composables/useFilters'
import { inSeasonDayOfYear, useInSeason } from '~/composables/useInSeason'

const { rows, load } = useObservations()
const { models, fetchModels } = useMaxEnt()
const { elevLabel } = useUnits()
const { setFilter } = useFilters()
const router = useRouter()

const loading = ref(true)
const selectedSpecies = ref(null)

const today = new Date()
const todayDoy = inSeasonDayOfYear(today)

const { species: topSpecies } = useInSeason(rows, models, { day: todayDoy })

function toggleSpecies(name) {
  selectedSpecies.value = selectedSpecies.value === name ? null : name
}

function applyFilter() {
  if (selectedSpecies.value) setFilter('taxon', selectedSpecies.value)
  else setFilter('taxon', '')
}

function goToMap() {
  applyFilter()
  const m = latestModel.value
  router.push(m ? `/map?layer=maxent:${m.id}` : '/map')
}

function goToData() {
  applyFilter()
  router.push('/data')
}

const latestModel = computed(() => {
  const base = (models.value || [])
  const filtered = selectedSpecies.value
    ? base.filter((m) => m.species === selectedSpecies.value || m.target_species === selectedSpecies.value)
    : base
  return [...filtered]
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())[0] || null
})


const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const seasonLabel = computed(() => {
  const m = today.getMonth()
  return `${MONTH_NAMES[(m + 11) % 12]} – ${MONTH_NAMES[m]} – ${MONTH_NAMES[(m + 1) % 12]}`
})

onMounted(async () => {
  await Promise.all([load(), fetchModels()])
  loading.value = false
})
</script>

<style scoped>
.now-fruiting { height: 100%; display: flex; flex-direction: column; gap: 0.6rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.4rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.6rem; align-items: center; flex-wrap: wrap; }
.widget-link { font-size: 0.8rem; color: var(--accent, #2a78d6); text-decoration: none; }
.widget-link:hover { text-decoration: underline; }
.widget-link-btn {
  font: inherit; font-size: 0.8rem; color: var(--accent, #2a78d6); background: none;
  border: none; padding: 0; cursor: pointer;
}
.widget-link-btn:hover { text-decoration: underline; }
.clear-btn {
  font: inherit; font-size: 0.7rem; border: 1px solid var(--accent, #2a78d6);
  background: color-mix(in srgb, var(--accent, #2a78d6) 12%, transparent);
  color: var(--accent, #2a78d6); border-radius: 999px;
  padding: 0.1rem 0.5rem; cursor: pointer; max-width: 160px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.season-label { margin: 0; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted, #888); }

.nf-loading, .nf-empty { font-size: 0.85rem; color: var(--muted, #888); }

.nf-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; flex: 1; overflow-y: auto; }
.nf-item {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.3rem 0.5rem; background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px;
  cursor: pointer; transition: border-color 0.15s, background 0.15s;
}
.nf-item:hover { border-color: var(--accent, #2a78d6); }
.nf-item.selected { border-color: var(--accent, #2a78d6); background: color-mix(in srgb, var(--accent, #2a78d6) 8%, var(--surface-2, #f5f5f5)); }
.nf-item.has-model .nf-name { color: var(--accent, #2a78d6); }

.nf-name { font-size: 0.82rem; color: var(--text, #222); font-style: italic; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nf-meta { display: flex; gap: 0.5rem; align-items: center; flex-shrink: 0; }
.nf-model-badge {
  font-size: 0.6rem; font-weight: 700; color: #fff;
  background: var(--accent, #2a78d6); border-radius: 3px; padding: 0 3px;
  font-style: normal;
}
.nf-peak { font-size: 0.7rem; color: var(--accent, #2a78d6); font-variant-numeric: tabular-nums; white-space: nowrap; }
.nf-iqr { font-size: 0.7rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; white-space: nowrap; }
.nf-elev { font-size: 0.68rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; white-space: nowrap; opacity: 0.8; }
.nf-count { font-size: 0.75rem; color: var(--muted, #888); font-variant-numeric: tabular-nums; }

.nf-model {
  margin: 0.2rem 0 0; font-size: 0.76rem; color: var(--muted, #888);
  border-top: 1px solid var(--border-soft, #eee); padding-top: 0.5rem;
}
.nf-model-label { margin-right: 0.3rem; }
.nf-model-link { color: var(--accent, #2a78d6); text-decoration: none; }
.nf-model-link:hover { text-decoration: underline; }
</style>
