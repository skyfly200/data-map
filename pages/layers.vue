<template>
  <div class="layers-page">
    <div class="head">
      <div>
        <h2>My custom layers</h2>
        <p class="sub">Register Earth Engine assets as map layers visible to yourself and other members.</p>
      </div>
    </div>

    <ClientOnly>
      <div v-if="!membership.isMember.value" class="gate">
        <p>Adding custom layers is a membership benefit.</p>
        <NuxtLink v-if="!membership.isAuthed.value" to="/login" class="btn primary">Sign in</NuxtLink>
      </div>

      <template v-else>
        <p v-if="error" class="msg error">{{ error }}</p>
        <p v-if="note" class="msg ok">{{ note }}</p>
        <p v-if="loading" class="msg">Loading…</p>

        <section class="panel">
          <p class="hint lead">
            Compute a layer in the Earth Engine Code Editor, export it with
            <code>Export.image.toAsset()</code>, then register the asset here. The app
            renders it like any built-in layer. An asset ID is stored, never a script.
          </p>

          <label class="stack wide"><span>Start from a preset</span>
            <select :value="presetId" @change="usePreset($event.target.value)">
              <option value="">Fill the rendering by hand</option>
              <option v-for="p in LAYER_PRESETS" :key="p.id" :value="p.id">{{ p.label }}</option>
            </select></label>
          <p v-if="presetHint" class="hint">{{ presetHint }}</p>

          <div class="grid">
            <label class="stack"><span>Name</span>
              <input v-model="draftLayer.name" placeholder="Chanterelle habitat index" /></label>
            <label class="stack"><span>Short name (URL)</span>
              <input v-model="draftLayer.slug" placeholder="chanterelle-habitat" /></label>
            <label class="stack"><span>Group</span>
              <input v-model="draftLayer.group" placeholder="Custom" /></label>
            <label class="stack"><span>Who can see it</span>
              <select v-model="draftLayer.tier">
                <option value="free">Everyone</option>
                <option value="member">Members</option>
                <option value="admin">Administrators</option>
              </select></label>
          </div>

          <label class="stack wide"><span>Earth Engine asset ID</span>
            <input v-model="draftLayer.asset_id" spellcheck="false"
                   placeholder="projects/your-project/assets/chanterelle-index" /></label>

          <div class="grid">
            <label class="stack"><span>Asset type</span>
              <select v-model="draftLayer.asset_type">
                <option value="image">Single image</option>
                <option value="image_collection">Image collection</option>
              </select></label>
            <label class="stack"><span>Band</span>
              <input v-model="draftLayer.band" spellcheck="false" placeholder="b1" /></label>
            <template v-if="draftLayer.asset_type === 'image_collection'">
              <label class="stack"><span>Combine with</span>
                <select v-model="draftLayer.reducer">
                  <option value="mosaic">Most recent (mosaic)</option>
                  <option value="mean">Mean</option>
                  <option value="median">Median</option>
                  <option value="max">Maximum</option>
                  <option value="min">Minimum</option>
                  <option value="first">First</option>
                </select></label>
              <label class="stack"><span>From</span>
                <input v-model="draftLayer.date_from" type="date" /></label>
              <label class="stack"><span>To</span>
                <input v-model="draftLayer.date_to" type="date" /></label>
            </template>
          </div>

          <div class="grid">
            <label class="stack"><span>Minimum</span>
              <input v-model="draftLayer.vis_min" type="number" step="any" /></label>
            <label class="stack"><span>Maximum</span>
              <input v-model="draftLayer.vis_max" type="number" step="any" /></label>
            <label class="stack"><span>Hide values below</span>
              <input v-model="draftLayer.mask_below" type="number" step="any" placeholder="0" /></label>
            <label class="stack"><span>Opacity</span>
              <input v-model="draftLayer.opacity" type="number" min="0.05" max="1" step="0.05" /></label>
          </div>

          <label class="stack wide"><span>Palette (hex colors, low to high)</span>
            <input v-model="draftLayer.palette" spellcheck="false"
                   placeholder="#f7fcb9, #addd8e, #31a354" /></label>
          <label class="stack wide"><span>Attribution</span>
            <input v-model="draftLayer.attribution" placeholder="FRMS, from Sentinel-2" /></label>
          <label class="stack wide"><span>Caveat shown with the key</span>
            <textarea v-model="draftLayer.note" rows="2"
                      placeholder="What it shows, and where it should not be trusted."></textarea></label>

          <div class="actions">
            <button class="btn primary" :disabled="saving" @click="saveLayer">
              {{ saving ? 'Saving…' : (draftLayer.id ? 'Save changes' : 'Add layer') }}
            </button>
            <button v-if="draftLayer.id" class="btn" @click="resetLayer">Cancel</button>
          </div>

          <p v-if="!layers.length && !loading" class="msg">No custom layers yet.</p>
          <ul v-else class="rows">
            <li v-for="l in layers" :key="l.id" class="row-item">
              <div class="row-top">
                <strong>{{ l.name }}</strong>
                <span class="tier-badge" :class="l.tier">{{ l.tier }}</span>
                <span class="usage">{{ l.group }} · <code>{{ l.asset_id }}</code></span>
                <button class="linkish" @click="editLayer(l)">Edit</button>
                <button class="linkish danger" @click="removeLayer(l)">Remove</button>
              </div>
            </li>
          </ul>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
import { computed, ref, reactive, onMounted } from 'vue'
import { LAYER_PRESETS, applyPreset } from '~/netlify/lib/ee-custom-layers.mjs'

const membership = useMembership()
const { accessToken } = useAuth()

const loading = ref(false)
const saving = ref(false)
const error = ref('')
const note = ref('')
const layers = ref([])

const BLANK_LAYER = {
  id: null, name: '', slug: '', group: 'Custom', tier: 'member',
  asset_id: '', asset_type: 'image', band: '', reducer: 'mosaic',
  date_from: '', date_to: '',
  vis_min: '', vis_max: '', mask_below: '', opacity: 0.8,
  palette: '#f7fcb9, #addd8e, #31a354', attribution: '', note: '',
}
const draftLayer = reactive({ ...BLANK_LAYER })

const presetId = ref('')
const presetHint = computed(() => {
  const p = LAYER_PRESETS.find((x) => x.id === presetId.value)
  return p ? `${p.hint} Example asset: ${p.assetHint}` : ''
})

function usePreset(id) {
  presetId.value = id
  if (!id) return
  Object.assign(draftLayer, applyPreset({ ...draftLayer }, id))
}

function resetLayer() {
  Object.assign(draftLayer, BLANK_LAYER)
  presetId.value = ''
}

function editLayer(l) {
  Object.assign(draftLayer, {
    ...BLANK_LAYER,
    ...l,
    palette: (l.palette || []).join(', '),
    date_from: l.date_from ? String(l.date_from).slice(0, 10) : '',
    date_to: l.date_to ? String(l.date_to).slice(0, 10) : '',
    vis_min: l.vis_min ?? '', vis_max: l.vis_max ?? '', mask_below: l.mask_below ?? '',
  })
}

async function call(path, options = {}) {
  const token = await accessToken()
  const res = await fetch(path, {
    ...options,
    headers: {
      'content-type': 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
      ...(options.headers || {}),
    },
  })
  const data = await res.json().catch(() => ({ ok: false, error: res.statusText }))
  if (!data.ok) throw new Error(data.error || 'Something went wrong.')
  return data
}

async function loadLayers() {
  loading.value = true
  error.value = ''
  try {
    const data = await call('/.netlify/functions/member-layers')
    layers.value = data.layers || []
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

async function saveLayer() {
  saving.value = true
  error.value = ''
  note.value = ''
  try {
    await call('/.netlify/functions/member-layers', {
      method: 'POST',
      body: JSON.stringify({ action: 'save-layer', ...draftLayer }),
    })
    note.value = draftLayer.id ? 'Layer saved.' : "Layer added. It appears in the map's layer picker."
    resetLayer()
    await loadLayers()
  } catch (e) {
    error.value = e.message
  } finally {
    saving.value = false
  }
}

async function removeLayer(l) {
  if (!confirm(`Remove the layer "${l.name}"?`)) return
  error.value = ''
  try {
    await call('/.netlify/functions/member-layers', {
      method: 'POST',
      body: JSON.stringify({ action: 'delete-layer', id: l.id }),
    })
    await loadLayers()
  } catch (e) {
    error.value = e.message
  }
}

onMounted(async () => {
  await membership.refresh()
  if (membership.isMember.value) await loadLayers()
})
</script>

<style scoped>
.layers-page { padding: 16px 18px; max-width: 900px; margin: 0 auto; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 16px; color: var(--muted); font-size: 0.82rem; }

.gate { border: 1px solid var(--border); border-radius: 10px; padding: 20px; background: var(--surface);
  text-align: center; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.gate p { margin: 0; color: var(--muted); font-size: 0.88rem; }

.panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--surface); }
.rows { list-style: none; margin: 12px 0 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.row-item { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.row-top { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
.usage { margin-left: auto; color: var(--muted); font-size: 0.75rem; }

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin: 10px 0; }
.stack { display: flex; flex-direction: column; gap: 4px; font-size: 0.82rem; }
.stack span { color: var(--muted); font-size: 0.77rem; }
.wide { grid-column: 1 / -1; }
.stack input, .stack select, .stack textarea {
  background: var(--input-bg, var(--surface)); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 8px; font: inherit; font-size: 0.85rem; color: var(--text);
}
.stack textarea { resize: vertical; }

.actions { display: flex; gap: 8px; margin: 12px 0; }
.btn { padding: 6px 14px; border-radius: 6px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text); font: inherit; font-size: 0.82rem; cursor: pointer; }
.btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

.linkish { background: none; border: none; font: inherit; font-size: 0.78rem;
  color: var(--accent); cursor: pointer; padding: 0; }
.linkish.danger { color: var(--danger, #c0392b); }

.hint { font-size: 0.8rem; color: var(--muted); margin: 6px 0; }
.hint.lead { margin-bottom: 14px; }
.msg { padding: 8px 12px; border-radius: 7px; font-size: 0.83rem; margin: 0 0 10px; }
.msg.error { background: #fde; color: #800; border: 1px solid #fbb; }
.msg.ok { background: #dfd; color: #080; border: 1px solid #afa; }

.tier-badge { font-size: 0.68rem; padding: 1px 6px; border-radius: 4px;
  border: 1px solid var(--border); color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
</style>
