<template>
  <div class="offline-page">
    <div class="head">
      <div>
        <h2>Offline</h2>
        <p class="sub">
          What is kept in this browser so the app works with no signal. Everything here
          lives on this device only — it is never uploaded, and clearing the browser's
          site data removes it.
        </p>
      </div>
      <ClientOnly>
        <span class="net" :class="{ off: !online }">{{ online ? 'Online' : 'Offline' }}</span>
      </ClientOnly>
    </div>

    <ClientOnly>
      <p v-if="!supported" class="gate">
        This browser cannot store pages for offline use, so the app needs a connection.
      </p>
      <p v-else-if="!registered" class="gate">
        Offline storage activates on the next load of the app.
        <template v-if="isDev"> It is switched off while the dev server is running.</template>
      </p>

      <template v-else>
        <!-- ── Storage ──────────────────────────────────────────────────── -->
        <section class="panel">
          <h3>Storage</h3>
          <div class="usage">
            <div class="bar" :title="quotaTitle">
              <span class="fill" :style="{ width: quotaPct }"></span>
            </div>
            <p class="usage-line">
              <strong>{{ formatBytes(savedBytes) }}</strong>
              <template v-if="quotaBytes"> of {{ formatBytes(quotaBytes) }} available</template>
              <!-- Said plainly: the number above is the browser's, and the
                   per-area sizes further down are not measurable at all. -->
              <span class="usage-src">
                {{ bytesAreMeasured
                  ? 'Measured by the browser, across everything this site stores.'
                  : 'A floor, not a total: most tiles cannot be measured once saved.' }}
              </span>
            </p>
          </div>

          <div class="counts">
            <span><strong>{{ areas.length }}</strong> saved area{{ areas.length === 1 ? '' : 's' }}</span>
            <span><strong>{{ savedTiles.toLocaleString() }}</strong> tiles</span>
            <span :class="{ dim: !hasData }">{{ hasData ? 'Dataset saved' : 'No dataset saved' }}</span>
            <span :class="{ dim: !hasShell }">{{ hasShell ? 'App saved' : 'App not saved' }}</span>
          </div>
        </section>

        <!-- ── The app and the data ─────────────────────────────────────── -->
        <section class="panel">
          <h3>The app and its data</h3>
          <div class="thing">
            <div class="what">
              <strong>The app itself</strong>
              <small>Every page and the code behind it, so Charts and Analysis open offline too.</small>
            </div>
            <button class="btn" :disabled="!!busy" @click="doSaveShell">
              {{ busy === 'shell' ? 'Saving…' : hasShell ? 'Re-save' : 'Save' }}
            </button>
          </div>
          <div class="thing">
            <div class="what">
              <strong>Observations</strong>
              <small>
                The dataset the map, table and charts all read from.
                <template v-if="datasetLabel"> Currently {{ datasetLabel }}.</template>
              </small>
            </div>
            <button class="btn" :disabled="!!busy" @click="doSaveData">
              {{ busy === 'data' ? 'Saving…' : hasData ? 'Re-save' : 'Save' }}
            </button>
          </div>
        </section>

        <!-- ── Areas ────────────────────────────────────────────────────── -->
        <section class="panel">
          <div class="panel-head">
            <h3>Saved areas</h3>
            <NuxtLink to="/map" class="btn small">Save an area on the map</NuxtLink>
          </div>

          <p v-if="!areas.length" class="empty">
            No areas saved yet. Open the map, find the place you are going, and save it
            from the offline section of the map settings. Map tiles are the part that
            matters in the woods — the app and the dataset are small by comparison.
          </p>

          <ul v-else class="areas">
            <li v-for="a in described" :key="a.id" class="area">
              <div class="area-main">
                <!-- Renaming in place: the name is the only thing about a saved
                     area anyone wants to change, and a dialog for one field is
                     a dialog too many. -->
                <!-- Bound to a draft rather than straight to the record. With
                     :value alone, any re-render between the first keystroke and
                     the commit — a usage refresh is enough — put the old name
                     back under the cursor. -->
                <input
                  class="area-name" :value="drafts[a.id] ?? a.name" :aria-label="`Name of ${a.name}`"
                  @input="drafts[a.id] = $event.target.value"
                  @change="rename(a.id)"
                  @keyup.enter="$event.target.blur()"
                />
                <p class="area-meta">
                  {{ a.extent }} · {{ a.zooms }} ·
                  {{ (a.tiles || 0).toLocaleString() }} tiles ·
                  <span :title="'Estimated: saved tiles cannot be measured once stored'">
                    ~{{ formatBytes(a.bytes) }}
                  </span>
                </p>
                <p class="area-meta dim">
                  {{ a.layerNames }} · saved {{ fmtDate(a.savedAt) }}
                </p>
              </div>
              <div class="area-acts">
                <NuxtLink class="btn small" :to="mapLinkFor(a)" title="Open the map here">Open</NuxtLink>
                <button class="btn small" :disabled="!!busy || !online"
                        :title="online ? 'Fetch anything missing again' : 'Needs a connection'"
                        @click="resave(a)">
                  {{ busy === 'tiles' && resaving === a.id ? 'Saving…' : 'Re-save' }}
                </button>
                <button class="btn small danger" :disabled="!!busy" @click="remove(a)">Delete</button>
              </div>
            </li>
          </ul>

          <div v-if="busy" class="progress">
            <div class="pbar"><span class="pfill" :style="{ width: pct }"></span></div>
            <span class="ptext">{{ progress.done.toLocaleString() }} / {{ progress.total.toLocaleString() || '…' }}</span>
          </div>
          <p v-if="error" class="err">{{ error }}</p>
          <p v-if="note" class="ok">{{ note }}</p>
        </section>

        <!-- ── Clearing ─────────────────────────────────────────────────── -->
        <section class="panel">
          <h3>Clear</h3>
          <p class="sub small">
            Deleting an area above frees only the tiles no other area still needs.
            These clear whole categories at once.
          </p>
          <div class="clears">
            <button class="btn" :disabled="!!busy" @click="doClear('tiles', 'Map tiles and saved areas cleared.')">
              Map tiles and areas
            </button>
            <button class="btn" :disabled="!!busy" @click="doClear('data', 'Saved dataset cleared.')">
              Dataset
            </button>
            <button class="btn" :disabled="!!busy" @click="doClear('shell', 'Saved app cleared.')">
              The app
            </button>
            <button class="btn danger" :disabled="!!busy" @click="doClear('all', 'Everything saved offline was cleared.')">
              Everything
            </button>
          </div>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { AVG_TILE_BYTES, describeArea, formatBytes } from '~/composables/offlineTiles'

// The offline portal: one place to see what this device is holding and to
// manage it. The map's own offline panel saves the view in front of you; this
// is where the collection is read, renamed, refreshed and thrown away — which
// is not something that belongs in a popover over a map.

useHead({ title: 'Offline — Nexstrata' })

const offline = useOffline()
const {
  supported, registered, online, busy, progress, error, areas,
  savedBytes, savedTiles, hasData, hasShell, bytesAreMeasured, quotaBytes,
} = offline
const isDev = import.meta.dev

const { data, selectedDataset } = useObservations()
const note = ref('')
const resaving = ref('')

onMounted(async () => {
  await offline.register()
  await offline.refreshUsage()
  await offline.loadAreas()
})

const datasetLabel = computed(() => {
  const n = data.value?.features?.length || 0
  return n ? `${n.toLocaleString()} observations` : ''
})

const described = computed(() => areas.value.map((a) => {
  const d = describeArea(a)
  const names = (a.sources || []).map((s) => s.id || s.template)
  // What the worker recorded as actually stored, falling back to the geometry
  // for a record saved before the count existed.
  const tiles = Number.isFinite(a.tiles) ? a.tiles : d.tiles
  return {
    ...a, ...d,
    tiles,
    // Sized from that same count rather than from the geometry, or an area
    // whose tiles mostly failed reads as "0 tiles · ~30 KB" — two numbers
    // describing the same thing and disagreeing.
    bytes: tiles * AVG_TILE_BYTES,
    layerNames: names.length
      ? `${names.length} layer${names.length === 1 ? '' : 's'}`
      : 'no layers recorded',
  }
}))

const pct = computed(() => {
  const { done, total } = progress.value
  return total ? `${Math.min(100, Math.round((done / total) * 100))}%` : '0%'
})
const quotaPct = computed(() => {
  if (!quotaBytes.value) return '0%'
  return `${Math.min(100, Math.max(0.5, (savedBytes.value / quotaBytes.value) * 100))}%`
})
const quotaTitle = computed(() => (quotaBytes.value
  ? `${formatBytes(savedBytes.value)} of ${formatBytes(quotaBytes.value)} the browser allows this site`
  : 'The browser did not report a quota'))

const fmtDate = (iso) => {
  const d = new Date(iso)
  return Number.isFinite(d.getTime())
    ? d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
    : 'unknown'
}

/** Opening an area means putting the map over it, not just near it. */
function mapLinkFor(area) {
  const { north, south, east, west } = area.bounds
  const lat = ((north + south) / 2).toFixed(5)
  const lon = ((east + west) / 2).toFixed(5)
  return `/map?lat=${lat}&lon=${lon}&z=${area.minZoom}`
}

/** Names being typed, by area id, until they are committed. */
const drafts = ref({})

async function rename(id) {
  const value = drafts.value[id]
  if (value === undefined) return
  note.value = ''
  await offline.renameArea(id, value)
  // Dropped once it is stored, so the row goes back to showing the record and
  // a name the worker rejected does not linger as if it had been accepted.
  const { [id]: _committed, ...rest } = drafts.value
  drafts.value = rest
}

async function resave(area) {
  note.value = ''
  resaving.value = area.id
  const res = await offline.resaveArea(area)
  resaving.value = ''
  if (res) {
    note.value = res.failed
      ? `Re-saved ${(res.done - res.failed).toLocaleString()} of ${res.total.toLocaleString()} tiles; the rest could not be reached.`
      : `${area.name} is up to date.`
  }
  await offline.refreshUsage()
}

async function remove(area) {
  note.value = ''
  // Confirmed because it is not undoable and the thing being lost is the
  // reason someone drove somewhere with a phone.
  if (!window.confirm(`Delete “${area.name}”? Tiles another saved area still needs are kept.`)) return
  await offline.deleteArea(area.id)
  note.value = `“${area.name}” deleted.`
}

async function doSaveShell() {
  note.value = ''
  const res = await offline.saveShell()
  if (res) note.value = `App saved (${res.done - res.failed} of ${res.total} pages).`
}

async function doSaveData() {
  note.value = ''
  const url = selectedDataset.value
  if (!url || url.startsWith('mem:')) {
    note.value = 'This dataset lives only in this session, so there is nothing to save.'
    return
  }
  const res = await offline.saveData([url])
  if (res) note.value = res.failed ? 'Could not save the dataset.' : 'Observations saved.'
}

async function doClear(which, said) {
  note.value = ''
  if (!window.confirm(which === 'all'
    ? 'Clear everything saved for offline use on this device?'
    : 'Clear this? It cannot be undone.')) return
  await offline.clear(which)
  await offline.loadAreas()
  note.value = said
}
</script>

<style scoped>
.offline-page { max-width: 820px; margin: 0 auto; padding: 18px 16px 48px; color: var(--text); }

.head { display: flex; align-items: flex-start; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
.head h2 { margin: 0 0 6px; font-size: 1.3rem; }
.sub { margin: 0; color: var(--muted); font-size: 0.85rem; line-height: 1.55; max-width: 62ch; }
.sub.small { font-size: 0.78rem; margin-bottom: 10px; }

.net {
  flex: 0 0 auto; border: 1px solid var(--border); border-radius: 999px;
  padding: 3px 10px; font-size: 0.72rem; font-weight: 700; color: var(--accent);
}
.net.off { color: var(--danger); border-color: var(--danger); }

.gate {
  border: 1px solid var(--border); border-radius: 10px; padding: 16px;
  color: var(--muted); font-size: 0.85rem; line-height: 1.55;
}

.panel {
  border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  background: var(--surface); margin-bottom: 14px;
}
.panel h3 { margin: 0 0 10px; font-size: 0.95rem; }
.panel-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.panel-head h3 { margin: 0; }

.usage { margin-bottom: 12px; }
.bar { height: 8px; background: var(--surface-3); border-radius: 4px; overflow: hidden; margin-bottom: 8px; }
.fill { display: block; height: 100%; background: var(--accent); }
.usage-line { margin: 0; font-size: 0.85rem; }
.usage-src { display: block; color: var(--muted); font-size: 0.74rem; margin-top: 3px; line-height: 1.45; }

.counts { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.78rem; color: var(--muted); }
.counts strong { color: var(--text); }
.counts .dim { opacity: 0.6; }

.thing { display: flex; align-items: flex-start; gap: 12px; padding: 8px 0; }
.thing + .thing { border-top: 1px solid var(--border-soft); }
.what { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.what strong { font-size: 0.85rem; }
.what small { color: var(--muted); font-size: 0.76rem; line-height: 1.45; }

.empty { margin: 0; color: var(--muted); font-size: 0.82rem; line-height: 1.6; }

.areas { list-style: none; margin: 0; padding: 0; }
.area { display: flex; align-items: flex-start; gap: 12px; padding: 12px 0; }
.area + .area { border-top: 1px solid var(--border-soft); }
.area-main { flex: 1 1 auto; min-width: 0; }
.area-name {
  width: 100%; max-width: 32ch; background: transparent; border: 1px solid transparent;
  border-radius: 6px; padding: 3px 6px; margin: 0 0 4px -6px;
  color: var(--text); font: inherit; font-size: 0.92rem; font-weight: 600;
}
.area-name:hover { border-color: var(--border); }
.area-name:focus { border-color: var(--accent); outline: none; background: var(--surface-2); }
.area-meta { margin: 0 0 2px; color: var(--muted); font-size: 0.76rem; line-height: 1.45; }
.area-meta.dim { opacity: 0.75; }
.area-acts { flex: 0 0 auto; display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }

.btn {
  border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.8rem; font-weight: 600;
  cursor: pointer; text-decoration: none; display: inline-flex; align-items: center;
}
.btn:hover:not(:disabled) { background: var(--surface-3); }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn.small { padding: 4px 9px; font-size: 0.75rem; }
.btn.danger { color: var(--danger); border-color: var(--danger); }
.btn.danger:hover:not(:disabled) { background: var(--danger); color: #fff; }

.clears { display: flex; flex-wrap: wrap; gap: 8px; }

.progress { display: flex; align-items: center; gap: 8px; margin-top: 12px; }
.pbar { flex: 1 1 auto; height: 6px; background: var(--surface-3); border-radius: 3px; overflow: hidden; }
.pfill { display: block; height: 100%; background: var(--accent); transition: width 0.2s linear; }
.ptext { color: var(--muted); font-size: 0.72rem; font-variant-numeric: tabular-nums; }

.err { margin: 10px 0 0; color: var(--danger); font-size: 0.8rem; line-height: 1.45; }
.ok { margin: 10px 0 0; color: var(--accent); font-size: 0.8rem; line-height: 1.45; }

@media (max-width: 620px) {
  .area { flex-direction: column; align-items: stretch; }
  .area-acts { justify-content: flex-start; }
  .head { flex-direction: column; }
}
</style>
