<template>
  <div class="data-page">
    <div class="layout">
      <aside class="side">
        <FilterPanel />
      </aside>
      <section class="main">
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

    <div class="fetch-new">
      <label>Fetch a new species</label>
      <template v-if="configured && !isAuthed">
        <span class="fmsg">Live fetching is rate-protected.</span>
        <NuxtLink to="/login" class="signin-link">Sign in to fetch</NuxtLink>
      </template>
      <template v-else>
        <input v-model="newSpecies" type="text" placeholder="e.g. Amanita muscaria" :disabled="fetching" @keyup.enter="fetchNew" />
        <button :disabled="fetching || !newSpecies.trim()" @click="fetchNew">{{ fetching ? 'Fetching…' : 'Fetch' }}</button>
        <span v-if="fetchMsg && !fetching" :class="['fmsg', fetchOk ? 'ok' : 'err']">{{ fetchMsg }}</span>
      </template>
    </div>

    <div v-if="fetching" class="fetch-progress">
      <div class="pbar"><span class="pfill"></span></div>
      <span class="ptext">Fetching &amp; clustering “{{ fetchingName }}”… {{ elapsed }}s
        <em>(large species can take up to a minute)</em></span>
    </div>

    <div v-if="speciesOptions.length" class="species-search">
      <input v-model="speciesQuery" type="search" placeholder="Search species…" aria-label="Search species" />
      <span v-if="speciesQuery" class="found">{{ visibleSpecies.length }} match{{ visibleSpecies.length === 1 ? '' : 'es' }}</span>
    </div>

    <p v-if="error" class="msg error">Could not load observations ({{ error }}).</p>
    <p v-else-if="pending && !speciesOptions.length" class="msg">Loading…</p>
    <p v-else-if="!speciesOptions.length" class="msg">No species in the current dataset.</p>
    <p v-else-if="!visibleSpecies.length" class="msg">No species match “{{ speciesQuery }}”.</p>

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
          <tr v-for="opt in visibleSpecies" :key="opt.species" :class="{ off: !selected.has(opt.species) }" @click="toggle(opt.species)">
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
      </section>
    </div>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'

const { speciesOptions, speciesFilter, setSpeciesFilter, addInlineDataset, error, pending, load } = useObservations()
const { isAuthed, configured, accessToken } = useAuth()
const { filters } = useFilters()
onMounted(load)

// Turn the active location/time filters into iNaturalist query params so a
// scoped fetch pulls only what matches, instead of the whole history.
function fetchScopeParams() {
  const f = filters.value
  const p = new URLSearchParams()
  if (f.center && f.radiusKm) {
    p.set('lat', String(f.center.lat))
    p.set('lng', String(f.center.lng))
    p.set('radius', String(f.radiusKm))
  }
  if (f.dateFrom) p.set('d1', f.dateFrom)
  if (f.dateTo) p.set('d2', f.dateTo)
  // Whole-year / month shortcuts become a date range for iNaturalist.
  if (f.year && !f.dateFrom && !f.dateTo) {
    const mm = f.month ? String(f.month).padStart(2, '0') : null
    if (mm) {
      const last = new Date(Number(f.year), Number(f.month), 0).getDate()
      p.set('d1', `${f.year}-${mm}-01`); p.set('d2', `${f.year}-${mm}-${String(last).padStart(2, '0')}`)
    } else {
      p.set('d1', `${f.year}-01-01`); p.set('d2', `${f.year}-12-31`)
    }
  }
  return p
}

// ── Fetch a new species on demand (Netlify function → iNaturalist) ────────────
const newSpecies = ref('')
const fetching = ref(false)
const fetchMsg = ref('')
const fetchOk = ref(false)
const fetchingName = ref('')
const elapsed = ref(0)
let timer = null

function startTimer() {
  elapsed.value = 0
  const t0 = Date.now()
  timer = setInterval(() => { elapsed.value = Math.round((Date.now() - t0) / 1000) }, 250)
}
function stopTimer() { if (timer) { clearInterval(timer); timer = null } }
onBeforeUnmount(stopTimer)

async function fetchNew() {
  const q = newSpecies.value.trim()
  if (!q || fetching.value) return
  fetching.value = true
  fetchingName.value = q
  fetchMsg.value = ''
  startTimer()
  try {
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}
    const scope = fetchScopeParams()
    scope.set('species', q)

    let res
    try {
      res = await fetch(`/.netlify/functions/fetch-species?${scope.toString()}`, { headers })
    } catch {
      // fetch itself failed → the function isn't reachable (local dev, or not deployed).
      throw new Error('couldn’t reach the fetch function — it only runs on the deployed site.')
    }

    if (res.status === 401) {
      // Server requires auth but didn't accept us. Distinguish "no token sent"
      // (client can't read a session) from "token rejected".
      let detail = ''
      try { detail = (await res.json())?.error || '' } catch { /* ignore */ }
      if (!token) {
        throw new Error('the server requires sign-in, but the app couldn’t read your session. '
          + 'Make sure NUXT_PUBLIC_SUPABASE_URL and NUXT_PUBLIC_SUPABASE_ANON_KEY are set (and redeploy), then sign in again.')
      }
      // A token WAS sent but rejected → the function verifies against different
      // Supabase credentials than the browser signed in with.
      throw new Error('you’re signed in, but the server rejected the session. '
        + 'This usually means the function’s SUPABASE_URL / SUPABASE_ANON_KEY point at a different project (or wrong key) '
        + 'than the app’s NUXT_PUBLIC_SUPABASE_* — line those up and redeploy. '
        + (detail ? `(server: ${detail})` : ''))
    }
    if (!res.ok) throw new Error(`server returned ${res.status}.`)

    const data = await res.json()
    if (!data.ok) throw new Error(data.error || 'fetch failed')
    if (!data.count) { fetchOk.value = false; fetchMsg.value = `No research-grade observations found for “${q}”.`; return }
    const entry = { id: data.slug, label: `${data.species} (${data.count})`, path: data.path || `mem:${data.slug}` }
    addInlineDataset(entry, data.geojson)
    fetchOk.value = true
    fetchMsg.value = `Loaded ${data.count} observations for ${data.species}.` + (data.path ? '' : ' (session only — configure Supabase to persist)')
    newSpecies.value = ''
  } catch (e) {
    fetchOk.value = false
    fetchMsg.value = `Couldn’t fetch — ${e.message}`
  } finally {
    stopTimer()
    fetching.value = false
  }
}

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

// Search filter over the species list (display only; All/None still act on all).
const speciesQuery = ref('')
const visibleSpecies = computed(() => {
  const q = speciesQuery.value.trim().toLowerCase()
  if (!q) return speciesOptions.value
  return speciesOptions.value.filter((o) => o.species.toLowerCase().includes(q))
})

const allChecked = computed(() => selected.value.size === speciesOptions.value.length && speciesOptions.value.length > 0)
const someChecked = computed(() => selected.value.size > 0 && !allChecked.value)

const maxCount = computed(() => Math.max(1, ...speciesOptions.value.map((o) => o.count)))
function barWidth(n) { return `${(n / maxCount.value) * 100}%` }
</script>

<style scoped>
.data-page { padding: 16px 18px; max-width: 1200px; margin: 0 auto; }
.layout { display: grid; grid-template-columns: minmax(300px, 360px) 1fr; gap: 22px; align-items: start; }
.side { position: sticky; top: 16px; }
.main { min-width: 0; }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .side { position: static; }
}
.head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 12px; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 0; color: var(--muted); font-size: 0.82rem; }
.actions { display: flex; align-items: center; gap: 8px; }
.count { color: var(--muted); font-size: 0.82rem; }
.actions button { border: 1px solid var(--border); background: var(--surface); border-radius: 6px; padding: 4px 10px; font-size: 0.82rem; cursor: pointer; }
.actions button:hover { background: var(--surface-2); }

.table-wrap { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
thead th { background: var(--surface-2); text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); color: var(--text); }
tbody td { padding: 7px 10px; border-bottom: 1px solid var(--border-soft); }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--surface-2); }
tbody tr.off { color: #b0b6be; }
tbody tr.off .sp em { color: #b0b6be; }
.c-check { width: 34px; text-align: center; }
.c-num { text-align: right; width: 90px; font-variant-numeric: tabular-nums; }
.c-bar { width: 140px; }
.bar { display: block; height: 8px; background: #2a78d6; border-radius: 4px; }
tr.off .bar { background: #cbd5e1; }

.fetch-new {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px;
}
.fetch-new label { font-size: 0.82rem; font-weight: 600; color: var(--text); }
.fetch-new input { flex: 0 1 240px; border: 1px solid var(--border); border-radius: 6px; padding: 5px 9px; font-size: 0.85rem; }
.fetch-new button { border: 1px solid #2b7a3d; background: #2b7a3d; color: #fff; border-radius: 6px; padding: 5px 12px; font-size: 0.85rem; font-weight: 600; cursor: pointer; }
.fetch-new button:disabled { opacity: 0.55; cursor: default; }
.fmsg { font-size: 0.8rem; }
.fmsg.ok { color: #2b7a3d; }
.fmsg.err { color: #b00020; }
.signin-link { font-size: 0.82rem; font-weight: 600; color: #2b7a3d; text-decoration: none; border: 1px solid #2b7a3d; border-radius: 6px; padding: 4px 10px; }
.signin-link:hover { background: #f0fdf4; }

.fetch-progress { display: flex; align-items: center; gap: 10px; margin: -4px 0 12px; }
.pbar { position: relative; flex: 0 1 220px; height: 6px; background: var(--border); border-radius: 4px; overflow: hidden; }
.pfill { position: absolute; top: 0; left: 0; height: 100%; width: 40%; background: #2a78d6; border-radius: 4px; animation: indeterminate 1.1s ease-in-out infinite; }
@keyframes indeterminate { 0% { left: -40%; } 100% { left: 100%; } }
.ptext { font-size: 0.8rem; color: var(--text); }
.ptext em { color: var(--muted); font-style: normal; }
@media (prefers-reduced-motion: reduce) { .pfill { animation: none; width: 100%; opacity: 0.6; } }

.hint { margin-top: 12px; font-size: 0.8rem; color: var(--muted); }
.hint code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
.msg { padding: 16px; color: var(--muted); }
.msg.error { color: var(--danger); }

.species-search { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.species-search input {
  flex: 1 1 auto; border: 1px solid var(--border); border-radius: 8px; padding: 7px 11px;
  font-size: 0.88rem; background: var(--input-bg); color: var(--text);
}
.species-search .found { font-size: 0.8rem; color: var(--muted); white-space: nowrap; }
</style>
