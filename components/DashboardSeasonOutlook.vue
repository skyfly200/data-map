<template>
  <div class="dash-widget season-outlook">
    <div class="widget-head">
      <h3 class="widget-title"><span aria-hidden="true">📆 </span>Season outlook</h3>
      <div class="widget-actions">
        <select v-model.number="radiusMi" class="so-radius" aria-label="Distance" :disabled="loading">
          <option v-for="r in RADII" :key="r" :value="r">{{ r }} mi</option>
        </select>
        <button class="widget-link-btn" :disabled="loading" @click="locate">{{ place ? 'Refresh' : 'Use my location' }}</button>
      </div>
    </div>

    <label v-if="place" class="so-pick">
      <span class="so-pick-label">Species</span>
      <select v-model="species" class="so-select" :disabled="loading">
        <option value="">Top species near here</option>
        <option v-for="n in choices" :key="n" :value="n">{{ n }}</option>
      </select>
    </label>

    <p v-if="!place && !locError" class="so-note">
      When the rest of each species' season should start and end near you, from past seasons and the WeatherNext forecast.
    </p>
    <p v-if="locError" class="so-note error">{{ locError }}</p>
    <p v-if="loading" class="so-note">{{ progress || 'Loading…' }}</p>
    <p v-else-if="message" class="so-note">{{ message }}</p>

    <ul v-if="!loading && outlooks.length" class="so-list">
      <li v-for="s in outlooks" :key="s.name" class="so-item" :class="{ open: isOpen(s.name) }"
          role="button" tabindex="0" :aria-expanded="isOpen(s.name)"
          @click="toggle(s.name)" @keydown.enter.prevent="toggle(s.name)" @keydown.space.prevent="toggle(s.name)">
        <div class="so-row">
          <span class="so-name">{{ s.name }}</span>
          <span class="so-chip" :class="s.outlook.status">{{ STATUS[s.outlook.status] }}</span>
        </div>
        <div class="so-row so-detail">
          <span>{{ summary(s.outlook) }}</span>
          <span v-if="s.outlook.status !== 'over'" class="so-chance" title="Chance it is fruiting in the next 7 days">
            {{ pct(s.outlook.pNext7) }} this week
          </span>
        </div>
        <div v-if="isOpen(s.name)" class="so-more" @click.stop>
          <p class="so-basis">{{ s.basis }}</p>
          <p v-if="s.outlook.status !== 'over'" class="so-basis">
            {{ pct(s.outlook.pNext14) }} chance of fruiting in the next 2 weeks.
          </p>
          <p class="so-past-title">Past seasons nearby</p>
          <ul class="so-past">
            <li v-for="y in [...s.past].reverse()" :key="y.year">
              <span class="so-past-year">{{ y.year }}</span>
              <span>{{ pastLabel(y) }}</span>
              <span class="so-past-n">{{ y.n }} finds</span>
            </li>
          </ul>
        </div>
      </li>
    </ul>

    <p v-if="!loading && meta" class="so-foot">
      <template v-if="meta.init">WeatherNext run {{ runLabel(meta.init) }}, then {{ meta.analogYears.length }} past years.</template>
      <template v-else>{{ meta.note }}</template>
      Experimental; ranges are the middle 80% of scenarios.
    </p>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { outlookSpecies, useSeasonOutlook } from '~/composables/useSeasonOutlook'
import { doyLabel, localSeasons } from '~/composables/seasonForecast'

const RADII = [25, 50, 100]
const STATUS = { in: 'In season', upcoming: 'Coming', over: 'Over', unlikely: 'Unlikely' }
const KM_PER_MI = 1.609344

const { rows, load, pending } = useObservations()
const { loading, message, outlooks, meta, progress, run } = useSeasonOutlook()

const radiusMi = ref(50)
const place = ref(null)
const locError = ref('')
const species = ref('')
const opened = ref(new Set())
const year = new Date().getUTCFullYear()

const d = (doy) => doyLabel(year, doy)
// " (Oct 18 – 29)" after a date, or nothing when every scenario agrees.
const range = (q) => (q.p10 === q.p90 ? '' : ` (${d(q.p10)} – ${d(q.p90)})`)
const pct = (p) => `${Math.round(p * 100)}%`

function summary(o) {
  if (o.status === 'unlikely') return `${pct(o.pNoSeason)} of scenarios never start`
  if (o.status === 'over') return `Ended ~${d(o.end.p50)}`
  if (o.status === 'in') return `Ends ~${d(o.end.p50)}${range(o.end)}`
  return `Starts ~${d(o.onset.p50)}${range(o.onset)}, ends ~${d(o.end.p50)}`
}

// Species with enough past seasons near the place to forecast, for the picker.
const choices = computed(() => (place.value
  ? outlookSpecies(localSeasons(rows.value || [], { ...place.value, radiusKm: radiusMi.value * KM_PER_MI, year }))
  : []))

// One species picked: open its detail straight away.
const isOpen = (name) => opened.value.has(name) || (species.value && outlooks.value.length === 1)
function toggle(name) {
  const next = new Set(opened.value)
  if (next.has(name)) next.delete(name)
  else next.add(name)
  opened.value = next
}

function pastLabel(y) {
  const at = (doy) => doyLabel(y.year, doy)
  return `${at(y.onset)} – ${at(y.end)}`
}

function runLabel(iso) {
  const t = new Date(iso)
  return `${t.toLocaleDateString('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })} ${String(t.getUTCHours()).padStart(2, '0')}z`
}

async function forecast() {
  if (!place.value) return
  await load()
  // Another widget may already be loading the observations; wait for it.
  if (pending.value) await new Promise((resolve) => { const stop = watch(pending, (p) => { if (!p) { stop(); resolve() } }) })
  opened.value = new Set()
  await run(rows.value || [], {
    ...place.value, radiusKm: radiusMi.value * KM_PER_MI, species: species.value, distance: `${radiusMi.value} mi`,
  })
}

function locate() {
  locError.value = ''
  if (!navigator.geolocation) { locError.value = 'This browser cannot share a location.'; return }
  navigator.geolocation.getCurrentPosition(
    (pos) => {
      place.value = { lat: +pos.coords.latitude.toFixed(3), lon: +pos.coords.longitude.toFixed(3) }
      forecast()
    },
    () => { locError.value = 'Location access denied.' },
  )
}

watch([radiusMi, species], forecast)
</script>

<style scoped>
.season-outlook { height: 100%; display: flex; flex-direction: column; gap: 0.6rem; }
.widget-head { display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 0.4rem; }
.widget-title { margin: 0; font-size: 1rem; color: var(--text, #222); }
.widget-actions { display: flex; gap: 0.6rem; align-items: center; }
.widget-link-btn {
  font: inherit; font-size: 0.8rem; color: var(--accent, #2a78d6); background: none;
  border: none; padding: 0; cursor: pointer;
}
.widget-link-btn:hover { text-decoration: underline; }
.widget-link-btn:disabled { opacity: 0.5; cursor: default; }
.so-radius {
  font: inherit; font-size: 0.75rem; padding: 0.1rem 0.25rem;
  border: 1px solid var(--border-soft, #ddd); border-radius: 4px;
  background: var(--surface, #fff); color: var(--text, #222);
}
.so-pick { display: flex; align-items: center; gap: 0.5rem; min-width: 0; }
.so-pick-label { font-size: 0.75rem; color: var(--muted, #888); flex-shrink: 0; }
.so-select {
  flex: 1; min-width: 0; font: inherit; font-size: 0.8rem; padding: 0.25rem 0.35rem;
  border: 1px solid var(--border-soft, #ddd); border-radius: 4px;
  background: var(--surface, #fff); color: var(--text, #222);
}
.so-note { margin: 0; font-size: 0.82rem; color: var(--muted, #888); }
.so-note.error { color: var(--danger, #c0392b); }

.so-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.3rem; flex: 1; overflow-y: auto; }
.so-item {
  padding: 0.35rem 0.5rem; background: var(--surface-2, #f5f5f5);
  border: 1px solid var(--border-soft, #eee); border-radius: 6px;
}
.so-item { cursor: pointer; }
.so-item:hover, .so-item.open { border-color: var(--accent, #2a78d6); }
.so-more { margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px solid var(--border-soft, #eee); cursor: default; }
.so-basis { margin: 0 0 0.3rem; font-size: 0.74rem; color: var(--text, #222); }
.so-past-title { margin: 0.2rem 0; font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted, #888); }
.so-past { list-style: none; margin: 0; padding: 0; display: grid; gap: 0.1rem; }
.so-past li { display: flex; gap: 0.6rem; font-size: 0.72rem; color: var(--muted, #777); font-variant-numeric: tabular-nums; }
.so-past-year { color: var(--text, #222); min-width: 2.6rem; }
.so-past-n { margin-left: auto; }
.so-row { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; min-width: 0; }
.so-name { font-size: 0.82rem; font-style: italic; color: var(--text, #222); min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.so-detail { margin-top: 0.15rem; font-size: 0.74rem; color: var(--muted, #777); flex-wrap: wrap; }
.so-chance { color: var(--accent, #2a78d6); font-variant-numeric: tabular-nums; white-space: nowrap; }
.so-chip {
  flex-shrink: 0; font-size: 0.66rem; padding: 0 0.4rem; border-radius: 999px;
  border: 1px solid currentColor; white-space: nowrap;
}
.so-chip.in { color: #2e8b57; }
.so-chip.upcoming { color: var(--accent, #2a78d6); }
.so-chip.over, .so-chip.unlikely { color: var(--muted, #888); }
.so-foot {
  margin: 0; font-size: 0.7rem; color: var(--muted, #888);
  border-top: 1px solid var(--border-soft, #eee); padding-top: 0.4rem;
}
</style>
