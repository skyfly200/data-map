<template>
  <div class="off">
    <p v-if="!supported" class="off-note">
      This browser cannot store pages for offline use, so the app needs a connection.
    </p>
    <template v-else-if="!registered">
      <p class="off-note">
        Offline storage activates on the next load.
        <template v-if="isDev">It is disabled while running the dev server.</template>
      </p>
    </template>
    <template v-else>
      <p class="off-note">
        Saved data is kept in this browser, on this device. It is not uploaded anywhere,
        and clearing the browser's site data removes it.
      </p>

      <div class="off-row">
        <div class="off-what">
          <strong>The app</strong>
          <small>Every page and the code behind it, so Charts and Analysis open offline too.</small>
        </div>
        <button :disabled="!!busy" @click="saveShell">
          {{ busy === 'shell' ? 'Saving…' : 'Save' }}
        </button>
      </div>

      <div class="off-row">
        <div class="off-what">
          <strong>Observations</strong>
          <small>
            The dataset the map, table and charts all read from.
            <template v-if="datasetLabel"> Currently {{ datasetLabel }}.</template>
          </small>
        </div>
        <button :disabled="!!busy" @click="save">
          {{ busy === 'data' ? 'Saving…' : hasData ? 'Re-save' : 'Save' }}
        </button>
      </div>

      <!-- Only where the host view can say what area is on screen. -->
      <div v-if="bounds" class="off-area">
        <div class="off-what">
          <strong>Save this area</strong>
          <small>
            {{ estimate.tiles.toLocaleString() }} tiles, roughly {{ formatBytes(estimate.bytes) }}.
            Saves what is on screen now plus
            <span class="off-num">{{ extraZoom }}</span> zoom level{{ extraZoom === 1 ? '' : 's' }} closer,
            for the {{ sources.length }} layer{{ sources.length === 1 ? '' : 's' }} currently drawn.
          </small>
        </div>

        <!-- Named at the point of saving, not afterwards. A list of saved areas
             is read weeks later, and by then "39.74°N, 104.99°W" is a puzzle
             where "north ridge" is an answer. -->
        <label class="off-name">
          <span class="off-name-label">Name</span>
          <input v-model="areaName" type="text" :placeholder="suggestedName"
                 :disabled="!!busy" @keyup.enter="saveArea" />
        </label>

        <label class="off-zoom">
          Detail
          <input v-model.number="extraZoom" type="range" min="0" :max="MAX_EXTRA_ZOOM" step="1"
                 :disabled="!!busy" aria-label="How many zoom levels deeper to save" />
        </label>

        <div class="off-area-act">
          <span v-if="tooBig" class="off-warn">
            Past the {{ MAX_AREA_TILES.toLocaleString() }}-tile limit. Zoom in, or reduce the detail.
          </span>
          <button class="off-save" :disabled="!!busy || !estimate.tiles || tooBig" @click="saveArea">
            {{ busy === 'tiles' ? 'Saving…' : 'Save area' }}
          </button>
        </div>
      </div>

      <!-- Where the collection is managed. Renaming, re-saving and deleting a
           list of places is not something to do in a popover over a map. -->
      <NuxtLink to="/offline" class="off-portal">
        Manage saved areas<template v-if="areas.length"> ({{ areas.length }})</template> →
      </NuxtLink>

      <div v-if="busy" class="off-progress">
        <div class="pbar"><span class="pfill" :style="{ width: pct }"></span></div>
        <span class="ptext">{{ progress.done }} / {{ progress.total || '…' }}</span>
      </div>

      <p v-if="error" class="off-err">{{ error }}</p>
      <p v-if="lastResult" class="off-ok">{{ lastResult }}</p>

      <div class="off-foot">
        <!-- Sizes come from Content-Length, so they are transfer sizes — what
             it cost to fetch, not exactly what it occupies. Reading every body
             back to measure it would cost as much as the download again. -->
        <span class="off-usage" title="Approximate: measured as what was downloaded">
          Saved: ~{{ formatBytes(savedBytes) }}<template v-if="savedTiles"> · {{ savedTiles.toLocaleString() }} tiles</template>
          <template v-if="!online"> · <strong>offline now</strong></template>
        </span>
        <button class="off-clear" :disabled="!!busy" @click="clearAll">Clear</button>
      </div>
    </template>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import {
  MAX_AREA_TILES, MAX_EXTRA_ZOOM, countTilesInBounds, estimateSave, formatBytes, suggestAreaName,
} from '~/composables/offlineTiles'

const props = defineProps({
  // The area on screen, from the host view: { north, south, east, west, zoom }.
  // Absent on pages with no map, which then offer everything but the tiles.
  bounds: { type: Object, default: null },
  // The layers currently drawn, as { template, id } — so a save covers what the
  // viewer is actually looking at, and a layer whose URL expires is filed under
  // something that does not.
  sources: { type: Array, default: () => [] },
  datasetLabel: { type: String, default: '' },
})

const offline = useOffline()
const {
  supported, registered, online, busy, progress, error, savedBytes, savedTiles, hasData, areas,
} = offline
const isDev = import.meta.dev

const { data, selectedDataset } = useObservations()
const lastResult = ref('')
const extraZoom = ref(2)
const areaName = ref('')

onMounted(async () => {
  await offline.register()
  await offline.loadAreas()
})

const zoomRange = computed(() => {
  const z = Math.round(props.bounds?.zoom ?? 0)
  return { min: z, max: z + extraZoom.value }
})

// Counted rather than listed: this recomputes on every drag of the detail
// slider, and at four levels the list is tens of thousands of objects built and
// thrown away per frame.
const tileCount = computed(() => {
  if (!props.bounds || !props.sources.length) return 0
  return countTilesInBounds(props.bounds, zoomRange.value.min, zoomRange.value.max)
})
const estimate = computed(() => estimateSave(tileCount.value, Math.max(1, props.sources.length)))
const tooBig = computed(() => estimate.value.tiles > MAX_AREA_TILES)
const suggestedName = computed(() => suggestAreaName(props.bounds))

const pct = computed(() => {
  const { done, total } = progress.value
  return total ? `${Math.min(100, Math.round((done / total) * 100))}%` : '0%'
})

async function save() {
  lastResult.value = ''
  // The dataset is fetched by path, and that is what the worker keys on.
  const url = selectedDataset.value
  if (!url || url.startsWith('mem:')) {
    lastResult.value = 'This dataset lives only in this session, so there is nothing to save.'
    return
  }
  const res = await offline.saveData([url])
  if (res) lastResult.value = res.failed ? 'Could not save the dataset.' : 'Observations saved.'
}

async function saveShell() {
  lastResult.value = ''
  const res = await offline.saveShell()
  if (res) lastResult.value = `App saved (${res.done - res.failed} of ${res.total} pages).`
}

async function saveArea() {
  lastResult.value = ''
  if (!props.bounds || !props.sources.length) return
  const res = await offline.saveArea({
    name: areaName.value,
    bounds: props.bounds,
    minZoom: zoomRange.value.min,
    maxZoom: zoomRange.value.max,
    sources: props.sources,
  })
  if (!res) return
  areaName.value = ''
  lastResult.value = res.failed
    ? `Saved ${(res.done - res.failed).toLocaleString()} of ${res.total.toLocaleString()} tiles for “${res.area.name}”; the rest could not be reached.`
    : `“${res.area.name}” saved — ${res.done.toLocaleString()} tiles.`
}

async function clearAll() {
  lastResult.value = ''
  await offline.clear('all')
  lastResult.value = 'Saved data cleared.'
}
</script>

<style scoped>
.off { display: flex; flex-direction: column; gap: 10px; font-size: 0.82rem; color: var(--text); }
.off-note { margin: 0; color: var(--muted); font-size: 0.76rem; line-height: 1.45; }

.off-row { display: flex; align-items: flex-start; gap: 10px; }
.off-what { flex: 1 1 auto; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.off-what strong { font-weight: 600; }
.off-what small { color: var(--muted); font-size: 0.74rem; line-height: 1.45; }
.off-num { color: var(--text); font-weight: 600; }
.off-row button {
  flex: 0 0 auto; border: 1px solid var(--border); background: var(--surface-2); color: var(--text);
  border-radius: 6px; padding: 6px 12px; font-size: 0.8rem; font-weight: 600; cursor: pointer;
}
.off-row button:hover:not(:disabled) { background: var(--surface-3); }
.off-row button:disabled { opacity: 0.5; cursor: default; }

.off-zoom { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.74rem; }
.off-zoom input { flex: 1 1 auto; min-width: 0; }

.off-area {
  display: flex; flex-direction: column; gap: 7px;
  border: 1px solid var(--border-soft); border-radius: 8px; padding: 9px;
}
.off-name { display: flex; align-items: center; gap: 8px; font-size: 0.74rem; color: var(--muted); }
.off-name-label { flex: 0 0 auto; }
.off-name input {
  flex: 1 1 auto; min-width: 0; background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 7px;
  font: inherit; font-size: 0.78rem;
}
.off-name input:focus { border-color: var(--accent); outline: none; }
.off-area-act { display: flex; align-items: center; justify-content: flex-end; gap: 8px; flex-wrap: wrap; }
.off-warn { flex: 1 1 auto; color: var(--danger); font-size: 0.72rem; line-height: 1.4; }
.off-save {
  border: 1px solid var(--accent); background: var(--accent); color: #fff;
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.8rem; font-weight: 600;
  cursor: pointer;
}
.off-save:disabled { opacity: 0.5; cursor: default; }

.off-portal {
  color: var(--accent); font-size: 0.76rem; font-weight: 600; text-decoration: none;
}
.off-portal:hover { text-decoration: underline; }

.off-progress { display: flex; align-items: center; gap: 8px; }
.pbar { flex: 1 1 auto; height: 6px; background: var(--surface-3); border-radius: 3px; overflow: hidden; }
.pfill { display: block; height: 100%; background: var(--accent); transition: width 0.2s linear; }
.ptext { color: var(--muted); font-size: 0.72rem; font-variant-numeric: tabular-nums; }

.off-err { margin: 0; color: var(--danger); font-size: 0.76rem; line-height: 1.4; }
.off-ok { margin: 0; color: var(--accent); font-size: 0.76rem; line-height: 1.4; }

.off-foot {
  display: flex; align-items: center; justify-content: space-between; gap: 10px;
  border-top: 1px solid var(--border-soft); padding-top: 8px;
}
.off-usage { color: var(--muted); font-size: 0.74rem; }
.off-clear {
  border: 1px solid var(--border); background: transparent; color: var(--muted);
  border-radius: 6px; padding: 5px 10px; font-size: 0.76rem; cursor: pointer;
}
.off-clear:hover:not(:disabled) { color: var(--text); background: var(--surface-2); }
</style>
