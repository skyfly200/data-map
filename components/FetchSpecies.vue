<template>
  <div class="fetch">
    <div class="fetch-new">
      <label>Fetch a taxon from iNaturalist</label>
      <template v-if="configured && !isAuthed">
        <span class="fmsg">Live fetching is rate-protected.</span>
        <NuxtLink to="/login" class="signin-link">Sign in to fetch</NuxtLink>
      </template>
      <template v-else>
        <!-- Any rank: iNaturalist matches a taxon name at whatever level it
             sits, so "Amanitaceae" imports the family and "Fungi" the kingdom.
             The pipeline resolves each record's own ancestry on the way in, so
             a mixed import stays filterable at every rank. -->
        <input v-model="newSpecies" type="text" placeholder="e.g. Amanita muscaria, Amanitaceae, Fungi"
               :disabled="fetching" @keyup.enter="fetchNew" />
        <button :disabled="fetching || !newSpecies.trim()" @click="fetchNew">{{ fetching ? 'Fetching…' : 'Fetch' }}</button>
        <span v-if="fetchMsg && !fetching" :class="['fmsg', fetchOk ? 'ok' : 'err']">{{ fetchMsg }}</span>
      </template>
    </div>

    <div v-if="fetching" class="fetch-progress">
      <div class="pbar"><span class="pfill"></span></div>
      <span class="ptext">Fetching &amp; clustering “{{ fetchingName }}”… {{ elapsed }}s
        <em>(a genus or family can take a minute or more)</em></span>
    </div>

    <p class="hint">
      A fetch pulls research-grade observations for the taxon (scoped to any active location/time
      filters) and loads them into this session. Name it at any rank: a species, a genus, a family,
      or a whole kingdom, and every record comes back with its full ancestry, so the result stays
      filterable and groupable at each level. To add a taxon to the committed dataset, add it to
      the pipeline (<code>INAT_TAXON_NAME</code>) and re-run.
    </p>
  </div>
</template>

<script setup>
import { useObservations } from '~/composables/useObservations'

const { addInlineDataset } = useObservations()
const { isAuthed, configured, accessToken } = useAuth()
const { filters } = useFilters()

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
      throw new Error('couldn’t reach the fetch function, it only runs on the deployed site.')
    }

    if (res.status === 401) {
      let detail = ''
      try { detail = (await res.json())?.error || '' } catch { /* ignore */ }
      if (!token) {
        throw new Error('the server requires sign-in, but the app couldn’t read your session. '
          + 'Make sure NUXT_PUBLIC_SUPABASE_URL and NUXT_PUBLIC_SUPABASE_ANON_KEY are set (and redeploy), then sign in again.')
      }
      throw new Error('you’re signed in, but the server rejected the session. '
        + 'This usually means the function’s SUPABASE_URL / SUPABASE_ANON_KEY point at a different project (or wrong key) '
        + 'than the app’s NUXT_PUBLIC_SUPABASE_*, line those up and redeploy. '
        + (detail ? `(server: ${detail})` : ''))
    }
    if (!res.ok) throw new Error(`server returned ${res.status}.`)

    const data = await res.json()
    if (!data.ok) throw new Error(data.error || 'fetch failed')
    if (!data.count) { fetchOk.value = false; fetchMsg.value = `No research-grade observations found for “${q}”.`; return }
    const entry = { id: data.slug, label: `${data.species} (${data.count})`, path: data.path || `mem:${data.slug}` }
    addInlineDataset(entry, data.geojson)
    fetchOk.value = true
    fetchMsg.value = `Loaded ${data.count} observations for ${data.species}.` + (data.path ? '' : ' (session only, configure Supabase to persist)')
    newSpecies.value = ''
  } catch (e) {
    fetchOk.value = false
    fetchMsg.value = `Couldn’t fetch, ${e.message}`
  } finally {
    stopTimer()
    fetching.value = false
  }
}
</script>

<style scoped>
.fetch-new {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 14px;
}
.fetch-new label { font-size: 0.85rem; font-weight: 600; color: var(--text); }
.fetch-new input { flex: 0 1 260px; border: 1px solid var(--border); border-radius: 6px; padding: 6px 10px; font-size: 0.88rem; background: var(--input-bg); color: var(--text); }
.fetch-new button { border: 1px solid #2b7a3d; background: #2b7a3d; color: #fff; border-radius: 6px; padding: 6px 14px; font-size: 0.88rem; font-weight: 600; cursor: pointer; }
.fetch-new button:disabled { opacity: 0.55; cursor: default; }
.fmsg { font-size: 0.82rem; }
.fmsg.ok { color: var(--accent); }
.fmsg.err { color: var(--danger); }
.signin-link { font-size: 0.82rem; font-weight: 600; color: var(--accent); text-decoration: none; border: 1px solid var(--accent); border-radius: 6px; padding: 4px 10px; }

.fetch-progress { display: flex; align-items: center; gap: 10px; margin: 12px 0 0; }
.pbar { position: relative; flex: 0 1 220px; height: 6px; background: var(--border); border-radius: 4px; overflow: hidden; }
.pfill { position: absolute; top: 0; left: 0; height: 100%; width: 40%; background: #2a78d6; border-radius: 4px; animation: indeterminate 1.1s ease-in-out infinite; }
@keyframes indeterminate { 0% { left: -40%; } 100% { left: 100%; } }
.ptext { font-size: 0.82rem; color: var(--text); }
.ptext em { color: var(--muted); font-style: normal; }
@media (prefers-reduced-motion: reduce) { .pfill { animation: none; width: 100%; opacity: 0.6; } }

.hint { margin-top: 14px; font-size: 0.82rem; color: var(--muted); max-width: 640px; }
.hint code { background: var(--surface-2); padding: 1px 5px; border-radius: 4px; }
</style>
