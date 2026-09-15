<template>
  <div class="coverage">
    <div class="head">
      <div>
        <h2>Raster coverage</h2>
        <p class="sub">Which environmental layers are cached, for what dates, and over what area.</p>
      </div>
      <span v-if="cov" class="generated">updated {{ cov.generated.replace('T', ' ').replace('Z', ' UTC') }}</span>
    </div>

    <p v-if="error" class="msg">No coverage summary found</p>
    <p v-else-if="!cov" class="msg">Loading…</p>

    <template v-else>
      <div class="stats">
        <div class="stat"><span class="n">{{ cov.layers.length }}</span><span class="l">layers</span></div>
        <div class="stat"><span class="n">{{ datedCount }}</span><span class="l">dated snapshots</span></div>
        <div class="stat"><span class="n">{{ totalFiles }}</span><span class="l">files</span></div>
        <div class="stat"><span class="n">{{ mb(cov.total_bytes) }}</span><span class="l">on disk</span></div>
      </div>

      <div class="cards">
        <div v-for="l in cov.layers" :key="l.key" class="card">
          <div class="card-top">
            <span class="dot" :style="{ background: colorOf(l.key) }"></span>
            <h3>{{ l.label }}</h3>
          </div>
          <dl>
            <div><dt>Files</dt><dd>{{ l.count }}</dd></div>
            <div><dt>Dates</dt><dd>{{ l.date_range ? `${l.date_range[0]} → ${l.date_range[1]}` : 'static' }}</dd></div>
            <div><dt>Size</dt><dd>{{ mb(l.total_bytes) }}</dd></div>
            <div><dt>Extent</dt><dd class="extent">{{ bboxLabel(l.bbox) }}</dd></div>
          </dl>
        </div>
      </div>

      <template v-if="matrixDates.length">
        <div class="cal-head">
          <h3 class="section">Coverage by date</h3>
          <div class="legend-scale">
            <span>fewer layers</span>
            <span v-for="n in maxIntensity + 1" :key="n" class="swatch" :style="{ background: cellColor(n - 1) }"></span>
            <span>more</span>
          </div>
        </div>
        <p class="sub">Each square is a day, shaded by how many layers have data for it.</p>

        <!-- One readout for the whole calendar, in a fixed place.
             A tooltip that follows the cursor across an 11px grid is unreadable
             — it covers the squares either side of the one you are pointing at,
             and the answer moves while you are reading it. This holds still and
             keeps the last day you touched, which is also what makes it usable
             with a keyboard. -->
        <!-- Both rows are always present, even when empty. The readout sits
             directly above the grid, so letting it grow when a day is picked
             pushed the calendar down under the cursor — which moved the square
             out from under the pointer that had just selected it. -->
        <div class="cal-readout" :class="{ empty: !hovered }" aria-live="polite">
          <div class="ro-line">
            <template v-if="hovered">
              <strong class="ro-date">{{ longDate(hovered.date) }}</strong>
              <span v-if="hovered.intensity" class="ro-count">
                {{ hovered.intensity }} of {{ cov.layers.length }} layers
              </span>
              <span v-else class="ro-none">no layer has data for this day</span>
            </template>
            <span v-else class="ro-hint">Point at a day to see which layers cover it.</span>
          </div>
          <div class="ro-layers">
            <span v-for="key in (hovered?.layers || [])" :key="key" class="ro-chip">
              <span class="dot" :style="{ background: colorOf(key) }"></span>{{ labelOf(key) }}
            </span>
          </div>
        </div>

        <div class="cal-wrap" @mouseleave="hovered = null">
          <div v-for="cal in shownCalendars" :key="cal.year" class="cal-year">
            <div class="cal-title">{{ cal.year }} <span class="cal-total">{{ cal.total }} days</span></div>
            <div class="cal-grid" :style="{ gridTemplateColumns: `repeat(${cal.weeks.length}, 11px)` }">
              <div v-for="(mo, mi) in cal.monthLabels" :key="mi" class="cal-month"
                   :style="{ gridColumn: mo.col + 1 }">{{ mo.label }}</div>
              <template v-for="(week, wi) in cal.weeks">
                <div v-for="(day, di) in week" :key="`${wi}-${di}`" class="cal-cell"
                     :class="{ on: day && day.intensity, here: day && hovered && hovered.date === day.date }"
                     :style="{ gridColumn: wi + 1, gridRow: di + 2, background: day ? cellColor(day.intensity) : 'transparent' }"
                     :tabindex="day ? 0 : -1"
                     @mouseenter="hovered = day" @focus="hovered = day"></div>
              </template>
            </div>
          </div>
        </div>

        <!-- The archive goes back to 1988 and is mostly a wall of empty years.
             The two that answer "is this current" come first; the rest are one
             click away rather than a scroll. -->
        <div v-if="hiddenYearCount" class="cal-more">
          <button class="btn" @click="showAllYears = !showAllYears">
            {{ showAllYears
              ? 'Show recent years only'
              : `Show ${hiddenYearCount} earlier year${hiddenYearCount === 1 ? '' : 's'}` }}
          </button>
          <span v-if="!showAllYears" class="cal-more-note">
            back to {{ calendars.at(-1).year }}
          </span>
        </div>
      </template>
    </template>
  </div>
</template>

<script setup>
import { PALETTE, UNCLUSTERED } from '~/composables/useObservations'
import { ref, computed, onMounted } from 'vue'

// A Plus Code box and three Vega charts used to sit between the layer cards and
// the calendar. They fetched /data/richness/, /data/ndvi/ and /data/temporal/ —
// three directories that do not exist and that nothing in the pipeline writes —
// so every request 404'd, the chart spec stayed null, and all three sat on
// "Loading…" permanently. They were built on vue-vega, whose only published
// version declares a Vue 2 peer, in a Vue 3 app. Removed rather than repaired:
// there is no data source to repair them against.

// The coverage summary written by raster_coverage.py. Everything below reads
// through `cov`, so it has to exist before the fetch assigns into it — without
// the ref the assignment threw and the catch silently showed "not found".
const cov = ref(null)
const error = ref(false)

onMounted(async () => {
  try {
    const res = await fetch('/data/coverage.json')
    if (!res.ok) throw new Error('not found')
    cov.value = await res.json()
  } catch {
    error.value = true
  }
})

const colors = {}
function colorOf(key) {
  if (!(key in colors)) colors[key] = PALETTE[Object.keys(colors).length % PALETTE.length] || UNCLUSTERED
  return colors[key]
}

const datedLayers = computed(() => (cov.value?.layers || []).filter((l) => l.dated && l.dates.length))
const datedCount = computed(() => Object.keys(cov.value?.date_index || {}).length)
const totalFiles = computed(() => (cov.value?.layers || []).reduce((s, l) => s + l.count, 0))
const matrixDates = computed(() => Object.keys(cov.value?.date_index || {}))

const dateSets = computed(() => {
  const m = {}
  for (const l of datedLayers.value) m[l.key] = new Set(l.dates)
  return m
})
function has(key, date) { return dateSets.value[key]?.has(date) }

// ── Calendar heatmap: one grid per year, cells shaded by how many layers have
// data on that date (GitHub-contributions style). ─────────────────────────────
const MONTH_ABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const iso = (d) => d.toISOString().slice(0, 10)

const maxIntensity = computed(() => {
  let m = 0
  for (const layers of Object.values(cov.value?.date_index || {})) m = Math.max(m, layers.length)
  return m || 1
})

const calendars = computed(() => {
  const idx = cov.value?.date_index || {}
  const years = [...new Set(Object.keys(idx).map((d) => d.slice(0, 4)))].sort((a, b) => Number(b) - Number(a))
  return years.map((year) => {
    const y = Number(year)
    const start = new Date(Date.UTC(y, 0, 1))
    const end = new Date(Date.UTC(y, 11, 31))
    const weeks = []
    let week = new Array(start.getUTCDay()).fill(null) // pad to the first weekday
    const monthLabels = []
    for (let d = new Date(start); d <= end; d.setUTCDate(d.getUTCDate() + 1)) {
      const ds = iso(d)
      if (d.getUTCDate() === 1) monthLabels.push({ col: weeks.length, label: MONTH_ABBR[d.getUTCMonth()] })
      week.push({ date: ds, intensity: (idx[ds] || []).length, layers: idx[ds] || [] })
      if (week.length === 7) { weeks.push(week); week = [] }
    }
    if (week.length) { while (week.length < 7) week.push(null); weeks.push(week) }
    const total = Object.keys(idx).filter((d) => d.startsWith(year)).length
    return { year, weeks, monthLabels, total }
  })
})

// ── The calendar's own state ─────────────────────────────────────────────────

/** The day under the cursor or the keyboard focus, or null. */
const hovered = ref(null)

/** Whether the years before the recent two are shown. */
const showAllYears = ref(false)

// Two: the current year and the one before it. That pair answers the question
// the page is usually open for — is this up to date, and what did a full year
// look like — without unrolling four decades of archive underneath it.
const RECENT_YEARS = 2

const shownCalendars = computed(() =>
  (showAllYears.value ? calendars.value : calendars.value.slice(0, RECENT_YEARS)))

const hiddenYearCount = computed(() => Math.max(0, calendars.value.length - RECENT_YEARS))

/** A layer's own label, for the readout, falling back to its key. */
const labelOf = (key) => (cov.value?.layers || []).find((l) => l.key === key)?.label || key

const longDate = (iso) => {
  const d = new Date(`${iso}T00:00:00Z`)
  return Number.isFinite(d.getTime())
    ? d.toLocaleDateString(undefined, {
      weekday: 'short', day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC',
    })
    : iso
}

function cellColor(intensity) {
  if (!intensity) return 'var(--surface-2)'
  // Discrete green ramp by fraction of the max layer count on any day.
  const t = intensity / maxIntensity.value
  const alpha = 0.35 + 0.65 * t
  return `color-mix(in srgb, var(--accent) ${Math.round(alpha * 100)}%, var(--surface-2))`
}

function mb(bytes) {
  if (!bytes) return '0 MB'
  const v = bytes / 1e6
  return v >= 1000 ? `${(v / 1000).toFixed(1)} GB` : `${v.toFixed(1)} MB`
}
function bboxLabel(b) {
  if (!b) return ', '
  const ns = (v) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'N' : 'S'}`
  const ew = (v) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? 'E' : 'W'}`
  return `${ns(b[1])}–${ns(b[3])}, ${ew(b[0])}–${ew(b[2])}`
}
</script>

<style scoped>
.coverage { padding: 16px 18px; max-width: 960px; margin: 0 auto; }
.head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 0; color: var(--muted); font-size: 0.82rem; }
.generated { color: var(--muted); font-size: 0.76rem; white-space: nowrap; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 12px; margin-bottom: 20px; }
.stat { border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; background: var(--surface); display: flex; flex-direction: column; gap: 2px; }
.stat .n { font-size: 1.3rem; font-weight: 700; color: var(--text); }
.stat .l { font-size: 0.76rem; color: var(--muted); }

.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; margin-bottom: 28px; }
.card { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--surface); }
.card-top { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.card-top h3 { margin: 0; font-size: 0.92rem; }
.dot { width: 11px; height: 11px; border-radius: 50%; flex: 0 0 auto; }
dl { margin: 0; display: flex; flex-direction: column; gap: 5px; }
dl > div { display: flex; justify-content: space-between; gap: 10px; font-size: 0.82rem; }
dt { color: var(--muted); }
dd { margin: 0; color: var(--text); font-variant-numeric: tabular-nums; text-align: right; }
.extent { font-size: 0.76rem; }

.section { margin: 4px 0 2px; font-size: 1rem; }
.cal-head { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; flex-wrap: wrap; }
.legend-scale { display: inline-flex; align-items: center; gap: 4px; font-size: 0.72rem; color: var(--muted); }
.legend-scale .swatch { width: 11px; height: 11px; border-radius: 2px; border: 1px solid var(--border-soft); }

.cal-wrap { display: flex; flex-direction: column; gap: 18px; margin-top: 12px; }
.cal-year { overflow-x: auto; }
.cal-title { font-size: 0.85rem; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.cal-total { font-weight: 400; color: var(--muted); font-size: 0.78rem; }
.cal-grid { display: grid; grid-auto-rows: 11px; gap: 2px; grid-template-rows: 14px repeat(7, 11px); width: max-content; }
.cal-month { font-size: 0.66rem; color: var(--muted); grid-row: 1; align-self: end; white-space: nowrap; }
.cal-cell { width: 11px; height: 11px; border-radius: 2px; }
/* Only the days that carry data are worth pointing at, so only those take a
   cursor. An empty square still focuses, because a keyboard user moving along
   a row needs to know it is empty rather than skipped. */
.cal-cell.on { cursor: pointer; }
.cal-cell:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
/* The square being read, marked on the grid as well as in the readout — with a
   ring rather than a colour change, since the colour IS the value. */
.cal-cell.here { box-shadow: 0 0 0 2px var(--surface), 0 0 0 3px var(--text); }

.cal-readout {
  border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface-2); padding: 8px 12px; margin-top: 12px;
  font-size: 0.8rem;
}
.cal-readout.empty { color: var(--muted); }
/* Fixed rows, so the grid below never moves. The chip row holds its height
   whether or not there are chips in it. */
.ro-line { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 10px; min-height: 1.3em; }
.ro-layers {
  display: flex; flex-wrap: wrap; gap: 4px 10px;
  min-height: 1.25em; margin-top: 4px; overflow: hidden;
}
.ro-date { font-size: 0.85rem; }
.ro-count { color: var(--muted); }
.ro-none { color: var(--muted); font-style: italic; }
.ro-hint { color: var(--muted); }
.ro-chip { display: inline-flex; align-items: center; gap: 5px; color: var(--muted); font-size: 0.76rem; }
.ro-chip .dot { width: 8px; height: 8px; border-radius: 50%; flex: 0 0 auto; }

.cal-more { display: flex; align-items: center; gap: 10px; margin-top: 14px; }
.cal-more .btn {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.8rem; font-weight: 600;
  cursor: pointer;
}
.cal-more .btn:hover { background: var(--surface-3); }
.cal-more-note { color: var(--muted); font-size: 0.76rem; }
.msg { padding: 16px; color: var(--muted); }
.msg code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
</style>
