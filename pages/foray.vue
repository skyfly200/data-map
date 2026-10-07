<template>
  <div class="foray-page">
    <header class="foray-head">
      <div>
        <h2>Plan a foray</h2>
        <p class="sub">Where past finds suggest the species in season are fruiting.</p>
      </div>
      <div class="seg" role="tablist" aria-label="Planner mode">
        <button v-for="m in MODE_LIST" :key="m.key" role="tab" :aria-selected="mode === m.key"
                :class="{ on: mode === m.key }" :title="m.blurb" @click="mode = m.key">{{ m.label }}</button>
      </div>
    </header>

    <details class="filters" :open="filtersOpen" @toggle="filtersOpen = $event.target.open">
      <summary>
        <span class="f-title">Filters</span>
        <span class="f-sum">{{ filterSummary }}</span>
      </summary>
      <section class="foray-controls">
        <label class="ctl">When
          <select v-model="timeSel">
            <option value="now">Now</option>
            <option v-for="(m, i) in MONTH_LABELS" :key="m" :value="String(i + 1)">{{ m }}</option>
          </select>
        </label>
        <label class="ctl">Land cover
          <select v-model="landCover">
            <option value="">All</option>
            <option v-for="lc in landCovers" :key="lc" :value="lc">{{ lc }}</option>
          </select>
        </label>
        <label class="ctl">Patch size
          <select v-model.number="cellSize">
            <option v-for="c in SIZE_OPTIONS" :key="c.value" :value="c.value">{{ c.label }}</option>
          </select>
        </label>
        <fieldset class="switches" :disabled="!access.loaded.value">
          <label :title="switchTitle"><input v-model="sw.free" type="checkbox"> Only free places</label>
          <label :title="switchTitle"><input v-model="sw.public" type="checkbox"> Only public lands</label>
          <label :title="switchTitle"><input v-model="sw.collecting" type="checkbox"> Collecting allowed only</label>
          <label :title="switchTitle"><input v-model="sw.includeLikely" type="checkbox" :disabled="!sw.collecting"> Include likely (BLM/USFS, estimated)</label>
          <label v-if="mode !== 'forager'" :title="switchTitle"><input v-model="sw.includeUnknown" type="checkbox"> Keep unknown</label>
        </fieldset>
      </section>
      <p v-if="access.message.value" class="notice warn" role="status">{{ access.message.value }}</p>
      <p v-else-if="anySwitch" class="notice" role="status">
        Unknown fee, land or collecting status is excluded{{ sw.includeUnknown ? ' (except where you chose to keep it)' : '' }}.
      </p>
      <p v-if="sw.includeLikely && sw.collecting" class="notice">{{ LIKELY_DISCLAIMER }}</p>
    </details>

    <p v-if="error" class="notice warn">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !features.length" class="notice">Loading observations…</p>

    <div class="foray-body">
      <div ref="mapCol" class="map-col">
        <div class="map-box">
          <ClientOnly>
            <ForayMap :cells="shown" :ranked="ranked" :selected-key="selectedKey" @select="selectedKey = $event" />
            <template #fallback><div class="map-fallback">Loading map…</div></template>
          </ClientOnly>
        </div>
        <div class="legend">
          <span v-for="b in LEGEND" :key="b.label" class="lg"><i :style="{ background: b.color }"></i>{{ b.label }}</span>
          <span class="lg"><i class="pin">1</i>Top places</span>
        </div>
        <details class="howto">
          <summary>How to read this map</summary>
          <ul>
            <li>Each dot is a {{ sizeLabel }} patch of ground (zoom in to see its outline) with at least {{ FORAY_MIN_SAMPLE }} past iNaturalist finds.</li>
            <li>Its colour is the foray score: how many of the finds there are species that fruit {{ whenText }}, weighted toward species near their peak. Darker green is better.</li>
            <li>Numbered pins are the top places in the list below. Tap a pin, dot or list entry to see what was found there.</li>
            <li>No colour means nobody has logged finds there, not that it is poor. Places with only a few finds are ranked cautiously.</li>
          </ul>
        </details>

        <article v-if="sel" class="place" aria-live="polite">
          <div class="p-head">
            <span v-if="selRank" class="p-rank">{{ selRank }}</span>
            <div class="p-title">
              <strong>{{ cellSiteName(sel) }}</strong>
              <span class="p-sub">{{ sel.landCover || 'Land cover unknown' }}<template v-if="sel.access.areaName"> · {{ sel.lat.toFixed(3) }}, {{ sel.lon.toFixed(3) }}</template></span>
            </div>
            <span class="band" :class="scoreBand(sel.t).toLowerCase()">{{ scoreBand(sel.t) }}</span>
          </div>
          <p class="p-why">
            <strong>{{ sel.seasonalN }}</strong> of <strong>{{ sel.n }}</strong> past finds here are species in season {{ whenText }}.
            <template v-if="sel.n < 8"> Only {{ sel.n }} finds, so treat this as a hint.</template>
          </p>
          <ul v-if="sel.components.length" class="p-species">
            <li v-for="x in sel.components.slice(0, 5)" :key="x.species">
              <a :href="taxonUrl(x.species)" target="_blank" rel="noopener"><em>{{ x.species }}</em></a>
              <span class="p-n">{{ x.count }} {{ x.count === 1 ? 'find' : 'finds' }}</span>
            </li>
          </ul>
          <p v-else class="p-why">None of the finds here are species in season {{ whenText }}.</p>
          <div class="acc">
            <span class="chip">{{ sel.access.covered ? sel.access.public_access : 'access unknown' }}</span>
            <span class="chip" :class="{ ok: sel.access.fee_status === 'free' }">fee: {{ feeLabel(sel.access) }}</span>
            <span class="chip" :class="{ ok: sel.access.collecting === 'allowed' }">collecting: {{ collectingLabel(sel.access) }}</span>
          </div>
          <div class="p-links">
            <a :href="directionsUrl(sel)" target="_blank" rel="noopener">Directions</a>
            <a :href="inatUrlFor(sel)" target="_blank" rel="noopener">Finds on iNaturalist</a>
          </div>
        </article>
      </div>

      <div class="list-col">
        <h3 v-if="ranked.length" class="list-h">Top {{ ranked.length }} places {{ whenText }}</h3>
        <div v-if="!shown.length && !pending" class="empty">
          <template v-if="!base.cells.length">No places with enough finds for this time and land cover. Only places that already have finds can be scored.</template>
          <template v-else>No places pass the current filters. Turn a switch off{{ access.loaded.value && !sw.includeUnknown ? ' or keep unknown places' : '' }}.</template>
        </div>

        <p v-if="cfg.showCaveats && shown.length" class="caveat">
          Scores come from past iNaturalist finds, not surveys, and cover only cells with at least {{ FORAY_MIN_SAMPLE }} finds.
          A blank area is unsampled, not poor. Score is each cell's share of finds that are in-season species, weighted by phenology
          (effort-neutral); ranking pulls small samples toward the average as if each cell had {{ FORAY_PRIOR_FINDS }} extra average finds.
          It is not a forecast and does not extrapolate beyond sampled cells.
        </p>

        <ol class="ranked">
          <li v-for="(c, i) in ranked" :key="c.key" :class="{ sel: c.key === selectedKey }" @click="pick(c.key)">
            <div class="row1">
              <span class="rk">{{ i + 1 }}</span>
              <strong class="site">{{ cellSiteName(c) }}</strong>
              <span class="band" :class="scoreBand(c.t).toLowerCase()">{{ cfg.showComponents ? `${Math.round(c.adj * 100)}%` : scoreBand(c.t) }}</span>
            </div>
            <div class="row2">
              <span v-if="c.components.length"><em>{{ topNames(c) }}</em></span>
              <span v-else>Few in-season finds</span>
              <span>{{ c.n }} finds<template v-if="cfg.showSample"> · {{ density(c) }}/km²</template></span>
              <span v-if="c.landCover && mode === 'forager'">{{ c.landCover }}</span>
              <span v-if="cfg.showAccessCols" class="acc">
                <span class="chip">{{ c.access.covered ? c.access.public_access : 'access unknown' }}</span>
                <span class="chip" :class="{ ok: c.access.fee_status === 'free' }">fee: {{ feeLabel(c.access) }}</span>
                <span v-if="c.access.collecting === 'likely_allowed'" class="chip" :title="LIKELY_DISCLAIMER">{{ LIKELY_LABEL }}</span>
                <span v-else class="chip" :class="{ ok: c.access.collecting === 'allowed' }">collecting: {{ collectingLabel(c.access) }}</span>
              </span>
            </div>
            <div v-if="cfg.showComponents" class="comp">
              <div>score {{ c.score.toFixed(2) }} = Σ weight × in-window finds ÷ {{ c.n }} finds; sample-adjusted {{ c.adj.toFixed(2) }}; in-season share {{ Math.round(c.share * 100) }}%</div>
              <ul>
                <li v-for="x in c.components.slice(0, 4)" :key="x.species">
                  <em>{{ x.species }}</em> · weight {{ x.weight.toFixed(2) }} × {{ x.count }} = {{ x.contribution.toFixed(3) }}
                </li>
              </ul>
            </div>
          </li>
        </ol>
        <p class="notice">{{ DISCLAIMER }}</p>

        <section v-if="cfg.showShortlist" class="shortlist" id="foray-shortlist">
          <div class="sl-head">
            <h3>Shortlist</h3>
            <div class="sl-actions">
              <button @click="exportCsv" :disabled="!rows.length">Export CSV</button>
              <button @click="copyText" :disabled="!rows.length">{{ copied ? 'Copied' : 'Copy text' }}</button>
              <button @click="printList" :disabled="!rows.length">Print</button>
            </div>
          </div>
          <div v-if="rows.length" class="table-wrap">
            <table>
              <thead>
                <tr><th>#</th><th>Site</th><th>Score</th><th>Access</th><th>Fee</th><th>Collecting</th><th>Notes</th></tr>
              </thead>
              <tbody>
                <tr v-for="(r, i) in rows" :key="ranked[i].key">
                  <td>{{ r.rank }}</td>
                  <td>{{ r.site }}<div v-if="ranked[i].access.areaName" class="coords">{{ r.lat }}, {{ r.lon }}</div></td>
                  <td>{{ Math.round(r.score * 100) }}%</td>
                  <td>{{ r.access_class }}</td>
                  <td>{{ r.fee_status }}</td>
                  <td>{{ r.collecting_status }}</td>
                  <td>
                    <input class="note-in" type="text" :value="notes[ranked[i].key] || ''" placeholder="Add a note"
                           :aria-label="`Notes for ${r.site}`" @input="setNote(ranked[i].key, $event.target.value)">
                    <span class="note-print">{{ r.notes }}</span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-else class="empty">Nothing to list yet.</p>
          <p class="print-disclaimer">{{ DISCLAIMER }}</p>
        </section>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useObservations } from '~/composables/useObservations'
import { useAccess } from '~/composables/useAccess'
import { CELL_SIZES, todayOfYear, useMapHeatmaps } from '~/composables/useMapHeatmaps'
import { rampColor } from '~/composables/ramps'
import {
  adjustForSample, FORAY_MIN_SAMPLE, FORAY_PRIOR_FINDS, MONTH_LABELS, normaliseScores, rankCells, resolveTimeSelection, scoreBand,
} from '~/composables/forayScore'
import {
  COLORADO_BBOX, DISCLAIMER, LIKELY_DISCLAIMER, LIKELY_LABEL, FORAY_MODES, attachAccess, buildShortlist, cellSiteName, collectingLabel,
  effectiveSwitches, feeLabel, filterByAccess, shortlistToCsv, shortlistToText, UNKNOWN_ACCESS,
} from '~/composables/forayPlanner'

useHead({ title: 'Plan a foray · Nexstrata' })

const MODE_LIST = Object.values(FORAY_MODES)
const { data, error, pending, load } = useObservations()
const access = useAccess()
const heat = useMapHeatmaps()

const mode = ref('forager')
const timeSel = ref('now')
const landCover = ref('')
const cellSize = ref(0.02)
const selectedKey = ref('')
const sw = reactive({ ...FORAY_MODES.forager.switches })
const notes = ref({})
const copied = ref(false)
const filtersOpen = ref(false)
const mapCol = ref(null)

const SIZE_OPTIONS = CELL_SIZES.filter((x) => x.value >= 0.01 && x.value <= 0.1)
// Readable on the light OSM base: the low end is a visible yellow-green, not near-white.
const ramp = ['#e6f598', '#a6d96a', '#41ab5d', '#00592b']
const LEGEND = ['Low', 'Fair', 'Good', 'Great'].map((label, i) => ({ label, color: rampColor(ramp, (i + 0.5) / 4) }))

const cfg = computed(() => FORAY_MODES[mode.value])

// Mode only changes defaults and which panels show; the score is shared.
watch(mode, (m) => {
  Object.assign(sw, FORAY_MODES[m].switches)
  try { localStorage.setItem('foray-mode', m) } catch { /* ignore */ }
})

onMounted(async () => {
  try {
    const m = localStorage.getItem('foray-mode')
    if (m && FORAY_MODES[m]) mode.value = m
    notes.value = JSON.parse(localStorage.getItem('foray-notes') || '{}') || {}
  } catch { /* defaults */ }
  const q = new URLSearchParams(location.search).get('mode')
  if (q && FORAY_MODES[q]) mode.value = q
  await load()
  access.load(COLORADO_BBOX)
})

const features = computed(() => data.value?.features || [])

const landCovers = computed(() => {
  const n = new Map()
  for (const f of features.value) {
    const l = f.properties?.land_cover_label
    if (l) n.set(l, (n.get(l) || 0) + 1)
  }
  return [...n.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k)
})

const time = computed(() => resolveTimeSelection(timeSel.value === 'now' ? 'now' : Number(timeSel.value), todayOfYear()))

const base = computed(() => heat.computeForayCells(features.value, {
  day: time.value.day, window: time.value.window,
  landCover: landCover.value || undefined, size: cellSize.value,
}))

const effective = computed(() => effectiveSwitches({ ...sw }, access.status.value))
const anySwitch = computed(() => effective.value.free || effective.value.public || effective.value.collecting)
const switchTitle = computed(() => (access.loaded.value ? '' : 'Needs access data for this area'))

const scored = computed(() => {
  const withAccess = access.loaded.value || access.status.value === 'partial'
    ? attachAccess(base.value.cells, access.areas.value)
    : base.value.cells.map((c) => ({ ...c, access: { ...UNKNOWN_ACCESS } }))
  return normaliseScores(adjustForSample(filterByAccess(withAccess, effective.value)), 'adj')
})
const shown = computed(() => scored.value.map((c) => ({
  ...c,
  color: rampColor(ramp, c.t),
  tip: `${scoreBand(c.t)} · ${c.n} finds${c.components.length ? ` · ${c.components[0].species}` : ''}`,
})))
const ranked = computed(() => rankCells(shown.value, cfg.value.listLimit, 'adj'))
const rows = computed(() => buildShortlist(ranked.value, notes.value, cfg.value.listLimit))

const sel = computed(() => shown.value.find((c) => c.key === selectedKey.value) || null)
const selRank = computed(() => ranked.value.findIndex((c) => c.key === selectedKey.value) + 1)
// Keep a place selected so the details card always has something to say.
watch(ranked, (r) => {
  if (!sel.value && r.length) selectedKey.value = r[0].key
})

const sizeLabel = computed(() => SIZE_OPTIONS.find((c) => c.value === cellSize.value)?.label || 'small')
const whenText = computed(() => {
  if (timeSel.value !== 'now') return `around mid-${time.value.label}`
  const d = new Date()
  return `within ${time.value.window} days of ${d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}`
})
const filterSummary = computed(() => [
  timeSel.value === 'now' ? 'Now' : time.value.label,
  landCover.value || 'All land cover',
  sizeLabel.value,
  ...(anySwitch.value ? ['access filters on'] : []),
].join(' · '))

function pick(key) {
  selectedKey.value = key
  // On a phone the map is above the list; bring it back into view.
  const el = mapCol.value
  if (el && window.matchMedia('(max-width: 800px)').matches && el.getBoundingClientRect().top < 0) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

const directionsUrl = (c) => `https://www.google.com/maps/dir/?api=1&destination=${c.lat.toFixed(5)},${c.lon.toFixed(5)}`
const inatUrlFor = (c) => {
  const km = Math.max(1, Math.round(cellSize.value * 111 * 0.75))
  return `https://www.inaturalist.org/observations?lat=${c.lat.toFixed(4)}&lng=${c.lon.toFixed(4)}&radius=${km}&iconic_taxa=Fungi`
}
const taxonUrl = (name) => `https://www.inaturalist.org/taxa/search?q=${encodeURIComponent(name)}`

const topNames = (c) => {
  const names = c.components.slice(0, 2).map((x) => x.species).join(', ')
  return c.components.length > 2 ? `${names} +${c.components.length - 2} more` : names
}
const density = (c) => {
  const km2 = (cellSize.value * 111) ** 2 * Math.cos((c.lat * Math.PI) / 180)
  return (c.n / km2).toFixed(1)
}

function setNote(key, v) {
  notes.value = { ...notes.value, [key]: v }
  try { localStorage.setItem('foray-notes', JSON.stringify(notes.value)) } catch { /* ignore */ }
}

function exportCsv() {
  const blob = new Blob([shortlistToCsv(rows.value)], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `foray-shortlist-${new Date().toISOString().slice(0, 10)}.csv`
  a.click()
  URL.revokeObjectURL(a.href)
}

async function copyText() {
  try {
    await navigator.clipboard.writeText(shortlistToText(rows.value))
    copied.value = true
    setTimeout(() => { copied.value = false }, 1500)
  } catch { /* clipboard blocked */ }
}

const printList = () => window.print()
</script>

<style scoped>
.foray-page { box-sizing: border-box; height: 100%; overflow-y: auto; padding: 0.8rem 1rem 1.5rem; display: flex; flex-direction: column; gap: 0.6rem; color: var(--text); }
.foray-head { display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 0.6rem; }
h2 { margin: 0; font-size: 1.2rem; color: var(--text-strong); }
.sub { margin: 0.1rem 0 0; font-size: 0.82rem; color: var(--muted); }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg button { font: inherit; font-size: 0.82rem; padding: 0.35rem 0.8rem; background: var(--surface); color: var(--text); border: 0; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--border); }
.seg button.on { background: var(--accent); color: var(--accent-ink, #fff); font-weight: 600; }

.filters { border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.4rem 0.7rem; background: var(--surface); }
.filters > summary { cursor: pointer; display: flex; gap: 0.6rem; align-items: baseline; flex-wrap: wrap; list-style: none; }
.filters > summary::before { content: '▸'; color: var(--muted); }
.filters[open] > summary::before { content: '▾'; }
.filters > summary::-webkit-details-marker { display: none; }
.f-title { font-weight: 600; font-size: 0.9rem; }
.f-sum { font-size: 0.78rem; color: var(--muted); }
.filters[open] { display: flex; flex-direction: column; gap: 0.5rem; }
.foray-controls { margin-top: 0.4rem; display: flex; flex-wrap: wrap; gap: 0.6rem 1rem; align-items: flex-end; }
.ctl { display: flex; flex-direction: column; gap: 0.15rem; font-size: 0.72rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.ctl select { font: inherit; font-size: 0.85rem; text-transform: none; letter-spacing: 0; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.3rem 0.4rem; }
.switches { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.4rem 0.7rem; margin: 0; font-size: 0.85rem; }
.switches:disabled { opacity: 0.5; }
.switches label { display: inline-flex; gap: 0.3rem; align-items: center; }

.notice { margin: 0; font-size: 0.78rem; color: var(--muted); }
.notice.warn { color: var(--text); background: color-mix(in srgb, #e0a800 18%, var(--surface)); border: 1px solid color-mix(in srgb, #e0a800 50%, var(--border)); border-radius: 6px; padding: 0.35rem 0.6rem; }

.foray-body { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 0.8rem; align-items: start; }
.map-col { display: flex; flex-direction: column; gap: 0.4rem; min-width: 0; scroll-margin-top: 0.5rem; }
.map-box { height: min(55vh, 520px); }
.map-fallback { display: grid; place-items: center; height: 100%; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 0.3rem 0.8rem; font-size: 0.75rem; color: var(--muted); }
.lg { display: inline-flex; align-items: center; gap: 0.3rem; }
.lg i { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.lg i.pin { width: 16px; height: 16px; font: 700 10px/16px system-ui, sans-serif; font-style: normal; text-align: center; background: #fff; color: #1b3a1f; border: 2px solid #1b5e20; }
.howto { font-size: 0.8rem; color: var(--muted); }
.howto summary { cursor: pointer; color: var(--accent); }
.howto ul { margin: 0.3rem 0 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.place { border: 1px solid var(--accent); border-radius: 8px; padding: 0.6rem 0.7rem; background: var(--surface-2); display: flex; flex-direction: column; gap: 0.4rem; }
.p-head { display: flex; align-items: center; gap: 0.5rem; }
.p-rank { display: grid; place-items: center; flex: 0 0 26px; height: 26px; border-radius: 50%; background: var(--accent); color: var(--accent-ink, #fff); font-weight: 700; font-size: 0.8rem; }
.p-title { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.p-title strong { font-size: 0.95rem; overflow-wrap: anywhere; }
.p-sub { font-size: 0.75rem; color: var(--muted); }
.p-why { margin: 0; font-size: 0.82rem; }
.p-species { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.15rem; font-size: 0.82rem; }
.p-species li { display: flex; justify-content: space-between; gap: 0.5rem; }
.p-species a { color: var(--text); }
.p-n { color: var(--muted); font-size: 0.75rem; white-space: nowrap; }
.p-links { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.p-links a { font-size: 0.8rem; padding: 0.3rem 0.7rem; border: 1px solid var(--border); border-radius: 6px; color: var(--text); text-decoration: none; background: var(--surface); }
.list-h { margin: 0; font-size: 0.95rem; }

.list-col { display: flex; flex-direction: column; gap: 0.6rem; min-width: 0; }
.empty { font-size: 0.85rem; color: var(--muted); padding: 0.6rem; border: 1px dashed var(--border); border-radius: 8px; }
.caveat { margin: 0; font-size: 0.75rem; color: var(--muted); border-left: 3px solid var(--border); padding-left: 0.6rem; }
.ranked { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 0.35rem; }
.ranked > li { background: var(--surface-2); border: 1px solid var(--border-soft); border-radius: 8px; padding: 0.45rem 0.6rem; cursor: pointer; }
.ranked > li:hover, .ranked > li.sel { border-color: var(--accent); }
.row1 { display: flex; align-items: center; gap: 0.5rem; }
.rk { font-variant-numeric: tabular-nums; color: var(--muted); width: 1.3rem; }
.site { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9rem; }
.xy { font-weight: 400; font-size: 0.72rem; color: var(--muted); }
.band { font-size: 0.72rem; font-weight: 700; border-radius: 999px; padding: 0.05rem 0.55rem; background: var(--surface-3); }
.band.great { background: var(--accent); color: var(--accent-ink, #fff); }
.band.good { background: color-mix(in srgb, var(--accent) 45%, var(--surface-3)); }
.row2 { font-size: 0.78rem; color: var(--muted); margin-top: 0.15rem; display: flex; flex-wrap: wrap; gap: 0.2rem 0.6rem; }
.acc { display: inline-flex; flex-wrap: wrap; gap: 0.25rem; }
.chip { font-size: 0.68rem; border: 1px solid var(--border); border-radius: 4px; padding: 0 0.3rem; color: var(--text); }
.chip.ok { border-color: var(--accent); }
.comp { font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }
.comp ul { margin: 0.2rem 0 0; padding-left: 1rem; }

.shortlist { border: 1px solid var(--border); border-radius: 8px; padding: 0.6rem; background: var(--surface); }
.sl-head { display: flex; justify-content: space-between; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.sl-head h3 { margin: 0; font-size: 0.95rem; }
.sl-actions { display: flex; gap: 0.4rem; }
.sl-actions button { font: inherit; font-size: 0.78rem; padding: 0.25rem 0.6rem; border: 1px solid var(--border); border-radius: 6px; background: var(--surface-2); color: var(--text); cursor: pointer; }
.sl-actions button:disabled { opacity: 0.5; cursor: default; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.78rem; margin-top: 0.5rem; }
th, td { text-align: left; padding: 0.25rem 0.3rem; border-bottom: 1px solid var(--border-soft); vertical-align: top; }
.coords { color: var(--muted); font-size: 0.68rem; }
.note-in { width: 100%; min-width: 7rem; font: inherit; font-size: 0.75rem; background: var(--input-bg, var(--surface)); color: var(--text); border: 1px solid var(--border); border-radius: 4px; padding: 0.15rem 0.3rem; }
.note-print, .print-disclaimer { display: none; }

@media (max-width: 800px) {
  .foray-body { grid-template-columns: 1fr; }
  .map-box { height: 52vh; }
}
</style>

<style>
@media print {
  body:has(.foray-page) .app-header,
  body:has(.foray-page) .foray-head,
  body:has(.foray-page) .filters,
  body:has(.foray-page) .list-h,
  body:has(.foray-page) .notice,
  body:has(.foray-page) .map-col,
  body:has(.foray-page) .ranked,
  body:has(.foray-page) .caveat,
  body:has(.foray-page) .empty,
  body:has(.foray-page) .sl-actions,
  body:has(.foray-page) .note-in { display: none !important; }
  body:has(.foray-page) .foray-page { height: auto; overflow: visible; color: #000; }
  body:has(.foray-page) .foray-body { display: block; }
  body:has(.foray-page) .note-print, body:has(.foray-page) .print-disclaimer { display: block; }
  body:has(.foray-page) .shortlist { border: 0; }
}
</style>
