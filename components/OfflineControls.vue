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
      <div v-if="bounds" class="off-row">
        <div class="off-what">
          <strong>Map tiles for this view</strong>
          <small>
            {{ estimate.tiles.toLocaleString() }} tiles, roughly {{ formatBytes(estimate.bytes) }}.
            Pan and zoom to the area you want first: this saves what is on screen now,
            plus <span class="off-num">{{ extraZoom }}</span> zoom level{{ extraZoom === 1 ? '' : 's' }} closer.
          </small>
          <label class="off-zoom">
            Detail
            <input v-model.number="extraZoom" type="range" min="0" :max="MAX_EXTRA_ZOOM" step="1"
                   :disabled="!!busy" aria-label="How many zoom levels deeper to save" />
          </label>
        </div>
        <button :disabled="!!busy || !estimate.tiles" @click="saveArea">
          {{ busy === 'tiles' ? 'Saving…' : 'Save' }}
        </button>
      </div>

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
  MAX_EXTRA_ZOOM, estimateSave, formatBytes, tileUrl, tilesInBounds,
} from '~/composables/offlineTiles'

const props = defineProps({
  // The area on screen, from the host view: { north, south, east, west, zoom }.
  // Absent on pages with no map, which then offer everything but the tiles.
  bounds: { type: Object, default: null },
  // Tile templates for the layers currently drawn, so a save covers what the
  // viewer is actually looking at rather than a basemap they switched away from.
  templates: { type: Array, default: () => [] },
  datasetLabel: { type: String, default: '' },
})

const offline = useOffline()
const {
  supported, registered, online, busy, progress, error, savedBytes, savedTiles, hasData,
} = offline
const isDev = import.meta.dev

const { data, selectedDataset } = useObservations()
const lastResult = ref('')
const extraZoom = ref(2)

onMounted(() => { offline.register() })

const tiles = computed(() => {
  const b = props.bounds
  if (!b || !props.templates.length) return []
  const z = Math.round(b.zoom)
  return tilesInBounds(b, z, z + extraZoom.value)
})
const estimate = computed(() => estimateSave(tiles.value.length, Math.max(1, props.templates.length)))

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
  const urls = []
  for (const t of tiles.value) for (const tpl of props.templates) urls.push(tileUrl(tpl, t))
  const res = await offline.saveTiles(urls)
  if (!res) return
  lastResult.value = res.failed
    ? `Saved ${res.done - res.failed} of ${res.total} tiles, the rest could not be reached.`
    : `${res.done.toLocaleString()} tiles saved for this area.`
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
