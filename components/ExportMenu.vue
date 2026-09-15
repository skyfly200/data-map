<template>
  <PopoverMenu icon="⤓" label="Export" :title="`Download ${countLabel}`" :badge="badge">
    <div class="ex">
      <p class="ex-count">
        <strong>{{ count.toLocaleString() }}</strong>
        {{ count === 1 ? 'record' : 'records' }}
        <span v-if="subtitle" class="ex-sub">{{ subtitle }}</span>
      </p>

      <div class="ex-formats" role="radiogroup" aria-label="Format">
        <label v-for="f in FORMATS" :key="f.id" class="ex-fmt">
          <input v-model="format" type="radio" :value="f.id" />
          <span><strong>{{ f.label }}</strong><em>{{ f.note }}</em></span>
        </label>
      </div>

      <!-- The enriched row is wide — a job that ran every stage produces around
           forty columns — so which ones is a real question rather than a
           preference. Defaults to the shown set, which is what somebody looking
           at the table just decided they cared about.

           Offered only when the columns are known. A source fetched on demand
           has none to list until the download happens, and a picker showing
           "Everything (0)" would both read as broken and, worse, narrow the
           export to nothing. -->
      <div v-if="columnsKnown" class="ex-cols">
        <label class="ex-row">
          <span>Columns</span>
          <select v-model="columnMode">
            <option value="shown">Shown in the table ({{ shownColumns.length }})</option>
            <option value="all">Everything ({{ allColumns.length }})</option>
            <option value="custom">Choose…</option>
          </select>
        </label>

        <div v-if="columnMode === 'custom'" class="ex-pick">
          <div class="ex-pick-head">
            <button class="linkish" @click="picked = [...allColumns]">All</button>
            <button class="linkish" @click="picked = []">None</button>
            <span class="ex-sub">{{ picked.length }} of {{ allColumns.length }}</span>
          </div>
          <label v-for="c in allColumns" :key="c" class="ex-col">
            <input v-model="picked" type="checkbox" :value="c" />
            <span>{{ c }}</span>
          </label>
        </div>
      </div>

      <p v-if="format === 'geojson'" class="ex-note">
        Coordinates travel in the geometry. Narrowing columns never drops it.
      </p>
      <p v-else class="ex-note">
        Longitude and latitude get their own columns, since a CSV has nowhere
        else to put them.
      </p>

      <p v-if="error" class="ex-error">{{ error }}</p>

      <button class="btn primary wide" :disabled="busy || !count || nothingChosen"
              @click="run">
        {{ busy ? 'Preparing…' : `Download ${format === 'csv' ? 'CSV' : 'GeoJSON'}` }}
      </button>
      <p v-if="!count" class="ex-note">Nothing to export with the current filters.</p>
      <p v-else-if="nothingChosen" class="ex-note">Choose at least one column.</p>
    </div>
  </PopoverMenu>
</template>

<script setup>
// The button that hands the data over.
//
// Everything about the encoding lives in composables/dataExport.js; this is the
// choosing. Two decisions are baked in rather than offered:
//
//   No gate. The reference dataset is already a public file the app fetches by
//   URL, so asking somebody to sign in to download what they could already curl
//   would be theatre. A job result is the member's own work, and they had to be
//   a member to produce it.
//
//   The features are resolved when the button is pressed, not when the menu is
//   opened. A job result has to be fetched, and fetching every result on the
//   page because somebody opened a menu is a lot of storage reads for a click
//   that may not come.

import { computed, ref, watch } from 'vue'
import { columnsOf, exportBlob, exportFilename } from '~/composables/dataExport'

const props = defineProps({
  /** The features, or a function returning them (possibly a promise). */
  source: { type: [Array, Function], required: true },
  /** Stem of the filename, before the date. */
  name: { type: String, default: 'export' },
  /** How many records, when the source is a function and counting is cheaper. */
  total: { type: Number, default: null },
  /** Columns the viewer is already looking at, used for the default. */
  shown: { type: Array, default: () => [] },
  subtitle: { type: String, default: '' },
})

const FORMATS = [
  { id: 'geojson', label: 'GeoJSON', note: 'Points with every property. For GIS.' },
  { id: 'csv', label: 'CSV', note: 'One row per record. For a spreadsheet.' },
]

const { download } = useImageExport()

const format = ref('geojson')
const columnMode = ref('shown')
const picked = ref([])
const busy = ref(false)
const error = ref('')

/** Features already to hand, for the counts and the column list. */
const known = computed(() => (Array.isArray(props.source) ? props.source : []))

const count = computed(() => (props.total ?? known.value.length))
const countLabel = computed(() => `${count.value.toLocaleString()} records`)
const badge = computed(() => (format.value === 'csv' ? 'CSV' : 'GeoJSON'))

const allColumns = computed(() => columnsOf(known.value))

/**
 * Whether the columns can be listed at all.
 *
 * False for a source that is only fetched when the button is pressed — a job
 * result, a saved dataset. There is nothing to offer until the file arrives, so
 * the picker is hidden and the export carries everything.
 */
const columnsKnown = computed(() => allColumns.value.length > 0)

/** The columns the table is showing, narrowed to ones that exist. */
const shownColumns = computed(() => {
  const present = new Set(allColumns.value)
  const hit = props.shown.filter((c) => present.has(c))
  return hit.length ? hit : allColumns.value
})

/**
 * The columns to write, or null for "everything there turns out to be".
 *
 * Null rather than an empty array, and the difference matters: an empty array
 * asks the encoder for a file with no columns in it, which is what an unknown
 * source used to produce.
 */
const columns = computed(() => {
  if (!columnsKnown.value) return null
  if (columnMode.value === 'all') return allColumns.value
  if (columnMode.value === 'custom') return picked.value.length ? picked.value : []
  return shownColumns.value
})

/** Only reachable by unticking every box; not the state an unknown source is in. */
const nothingChosen = computed(() => Array.isArray(columns.value) && columns.value.length === 0)

// Opening "Choose…" with nothing ticked reads as a bug; start from what was
// already selected.
watch(columnMode, (mode) => {
  if (mode === 'custom' && !picked.value.length) picked.value = [...shownColumns.value]
})

async function run() {
  busy.value = true
  error.value = ''
  try {
    const features = typeof props.source === 'function' ? await props.source() : props.source
    const list = Array.isArray(features) ? features : (features?.features || [])
    if (!list.length) throw new Error('There was nothing to export.')

    // A source fetched on demand may carry columns the page never saw, so a
    // selection is resolved against what actually arrived. No selection means
    // everything in the file, which is the unknown-source case.
    let wanted = null
    if (columns.value) {
      const available = new Set(columnsOf(list))
      const hit = columns.value.filter((c) => available.has(c))
      wanted = hit.length ? hit : null
    }

    download(
      exportBlob(list, format.value, { columns: wanted }),
      exportFilename(props.name, format.value),
    )
  } catch (e) {
    error.value = e.message || 'That export failed.'
  } finally {
    busy.value = false
  }
}
</script>

<style scoped>
.ex { display: grid; gap: 10px; min-width: 250px; max-width: 320px; }

.ex-count { margin: 0; font-size: 0.85rem; }
.ex-sub { color: var(--muted); font-size: 0.76rem; }

.ex-formats { display: grid; gap: 6px; }
.ex-fmt { display: flex; gap: 7px; align-items: flex-start; font-size: 0.8rem; }
.ex-fmt strong { display: block; font-weight: 600; }
.ex-fmt em { display: block; font-style: normal; color: var(--muted); font-size: 0.74rem; }

.ex-row { display: flex; align-items: center; justify-content: space-between; gap: 8px;
  font-size: 0.8rem; }
.ex-row select { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 7px; font: inherit; font-size: 0.78rem; }

/* Forty checkboxes is a scroll, not a wall. */
.ex-pick { margin-top: 7px; max-height: 190px; overflow-y: auto;
  border: 1px solid var(--border); border-radius: 6px; padding: 7px; }
.ex-pick-head { display: flex; align-items: center; gap: 10px; margin-bottom: 5px; }
.ex-col { display: flex; gap: 6px; align-items: center; font-size: 0.76rem;
  font-variant-numeric: tabular-nums; }
.ex-col span { overflow-wrap: anywhere; }

.ex-note { margin: 0; font-size: 0.73rem; color: var(--muted); }
.ex-error { margin: 0; font-size: 0.78rem; color: #b3492f; }

.linkish { background: none; border: none; color: var(--accent, #3d8b5f); font: inherit;
  font-size: 0.75rem; cursor: pointer; padding: 0; text-decoration: underline; }
.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.82rem; cursor: pointer; }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
.btn.wide { width: 100%; }
</style>
