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
        <input v-model="newSpecies" type="text" list="taxon-suggestions"
               placeholder="e.g. Amanita muscaria, Amanitaceae, Fungi"
               :disabled="fetching" @keyup.enter="fetchNew" />
        <!-- The saved list as suggestions. Typing is still free-form — the list
             is what you usually want, not a limit on what you can ask for. -->
        <datalist id="taxon-suggestions">
          <option v-for="t in taxa" :key="t" :value="t"></option>
        </datalist>
        <button :disabled="fetching || !newSpecies.trim()" @click="fetchNew">{{ fetching ? 'Fetching…' : 'Fetch' }}</button>
        <!-- One button for the whole saved list, because the alternative is
             typing forty genus names in turn and remembering which you did. -->
        <button v-if="taxa.length" class="ghost" :disabled="fetching"
                :title="`Fetch each of the ${taxa.length} taxa in your list, one after another`"
                @click="fetchList">
          Fetch my list ({{ taxa.length }})
        </button>
        <NuxtLink to="/options#inaturalist-taxa" class="fmsg edit-link">Edit list</NuxtLink>
        <span v-if="fetchMsg && !fetching" :class="['fmsg', fetchOk ? 'ok' : 'err']">{{ fetchMsg }}</span>
      </template>
    </div>

    <div v-if="fetching" class="fetch-progress">
      <div class="pbar"><span class="pfill" :class="{ known: run.running }"
                              :style="run.running ? { width: `${Math.round((run.done / run.total) * 100)}%` } : null"></span></div>
      <span class="ptext">
        <template v-if="run.running">
          {{ run.done }} of {{ run.total }} — fetching "{{ fetchingName }}"…
          {{ run.loaded.toLocaleString() }} so far, {{ elapsed }}s
          <button class="linkish" @click="run.stop = true">
            {{ run.stop ? 'Stopping after this one…' : 'Stop' }}
          </button>
        </template>
        <template v-else>
          Fetching &amp; clustering "{{ fetchingName }}"… {{ elapsed }}s
          <em>(a genus or family can take a minute or more)</em>
        </template>
      </span>
    </div>

    <p class="hint">
      A fetch pulls research-grade observations for the taxon (scoped to any active location/time
      filters) and loads them into this session. Name it at any rank: a species, a genus, a family,
      or a whole kingdom, and every record comes back with its full ancestry, so the result stays
      filterable and groupable at each level. A whole kingdom is a very large request from a
      browser — for that, set the list in
      <NuxtLink to="/options#inaturalist-taxa">Options</NuxtLink> and run the pipeline.
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

// The saved list, from Options. Read here so the fetch box can offer it as
// suggestions and run the whole of it, rather than being a box you have to
// remember forty genus names to use.
const { taxa, loadFromStorage } = useTaxonList()
onMounted(loadFromStorage)

/** State of a run through the whole list. */
const run = reactive({ running: false, stop: false, done: 0, total: 0, loaded: 0, failed: [] })

function startTimer() {
  elapsed.value = 0
  const t0 = Date.now()
  timer = setInterval(() => { elapsed.value = Math.round((Date.now() - t0) / 1000) }, 250)
}
function stopTimer() { if (timer) { clearInterval(timer); timer = null } }
onBeforeUnmount(stopTimer)

/**
 * One taxon, fetched and loaded. Throws with something worth reading.
 *
 * Split out of the button handler so the same path serves one name typed in and
 * a whole saved list run through in turn — a list that fetched differently from
 * a single name would eventually differ in a way nobody noticed.
 */
async function fetchOne(q) {
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
  if (data.count) {
    const entry = { id: data.slug, label: `${data.species} (${data.count})`, path: data.path || `mem:${data.slug}` }
    addInlineDataset(entry, data.geojson)
  }
  return data
}

/**
 * The saved taxon list, fetched one at a time.
 *
 * In turn rather than at once: each one is a serverless invocation that can run
 * for a minute, and forty in parallel is forty concurrent functions and a rate
 * limit at iNaturalist. Slower, and it finishes.
 *
 * A failure does not stop the run. Thirty-nine genera and one that timed out is
 * a useful afternoon; abandoning it at the first failure is not. What failed is
 * named at the end.
 */
async function fetchList() {
  if (fetching.value || run.running) return
  const list = [...taxa.value]
  if (!list.length) return

  Object.assign(run, { running: true, stop: false, done: 0, total: list.length, loaded: 0, failed: [] })
  fetchMsg.value = ''
  fetching.value = true
  startTimer()
  try {
    for (const name of list) {
      if (run.stop) break
      fetchingName.value = name
      try {
        const data = await fetchOne(name)
        run.loaded += data.count || 0
      } catch (e) {
        run.failed.push(`${name} (${e.message})`)
      }
      run.done += 1
    }
    fetchOk.value = run.failed.length === 0
    const ran = `${run.done} of ${run.total}`
    fetchMsg.value = run.failed.length
      ? `Fetched ${ran}, ${run.loaded.toLocaleString()} observations. Failed: ${run.failed.join('; ')}`
      : `Fetched ${ran}, ${run.loaded.toLocaleString()} observations.`
  } finally {
    run.running = false
    stopTimer()
    fetching.value = false
  }
}

async function fetchNew() {
  const q = newSpecies.value.trim()
  if (!q || fetching.value) return
  fetching.value = true
  fetchingName.value = q
  fetchMsg.value = ''
  startTimer()
  try {
    const data = await fetchOne(q)
    if (!data.count) {
      fetchOk.value = false
      fetchMsg.value = `No research-grade observations found for "${q}".`
      return
    }
    fetchOk.value = true
    fetchMsg.value = `Loaded ${data.count} observations for ${data.species}.`
      + (data.path ? '' : ' (session only, configure Supabase to persist)')
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
.fetch-new button.ghost {
  background: transparent; color: var(--text); border-color: var(--border);
  font-weight: 600;
}
.fetch-new button.ghost:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
.edit-link { color: var(--accent); text-decoration: none; }
.edit-link:hover { text-decoration: underline; }
.linkish {
  background: none; border: 0; padding: 0 0 0 6px; cursor: pointer;
  font: inherit; font-size: 0.8rem; color: var(--accent); text-decoration: underline;
}
/* A run through a list knows how far along it is; a single fetch does not, and
   an indeterminate bar that claims a percentage would be lying about it. */
.pfill.known { transition: width 0.3s ease; animation: none; }

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
