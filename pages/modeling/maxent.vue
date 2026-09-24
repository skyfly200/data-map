<template>
  <div class="modeling">
    <MaxEntTutorial ref="tutorialRef" />

    <div class="head">
      <div class="head-text">
        <h2>MaxEnt Modeling</h2>
        <p class="sub">
          Predict <GlossaryTooltip term="habitat suitability" :definition="g('habitat suitability')">habitat suitability</GlossaryTooltip>
          by learning from environmental conditions at sighting locations.
          A model says where the environment resembles where the species was found — not where it is.
        </p>
      </div>
      <div class="head-actions">
        <button class="btn ghost small" aria-label="Open tutorial" @click="tutorialRef?.start()">Tutorial</button>
        <button v-if="selectedModels.length > 0" class="btn primary small" @click="showComparison = true">
          Compare ({{ selectedModels.length }})
        </button>
      </div>
    </div>

    <ClientOnly>
      <div v-if="membership.configured && !membership.isMember.value" class="gate">
        <p>MaxEnt modeling is a membership benefit.</p>
        <NuxtLink to="/login" class="btn primary">Sign in</NuxtLink>
      </div>

      <template v-else>
        <!-- Comparison Overlay -->
        <div v-if="showComparison" class="overlay" @click.self="showComparison = false">
          <div class="overlay-content">
            <button class="close-btn" @click="showComparison = false">✕</button>
            <ModelComparison :selected="selectedModels" />
          </div>
        </div>

        <!-- ─── Active Training Banner ──────────────────────────────────────── -->
        <div v-if="activeJob" class="job-banner">
          <div class="banner-body">
            <div class="banner-info">
              <strong class="banner-name">{{ activeJob.model_configs.title }}</strong>
              <span class="status-pill" :class="activeJob.status">{{ activeJob.status }}</span>
            </div>
            <div class="banner-bar">
              <div class="bar-fill" :style="{ width: progress + '%' }"></div>
            </div>
            <p v-if="activeJob.error_message" class="banner-error">{{ activeJob.error_message }}</p>
          </div>
          <button class="btn-text" @click="activeJob = null">Dismiss</button>
        </div>

        <!-- ─── Your Models ─────────────────────────────────────────────────── -->
        <section class="panel">
          <div class="panel-head">
            <h3>Your Models</h3>
            <button class="btn-text" :disabled="loading" @click="fetchModels">Refresh</button>
          </div>

          <p v-if="loading" class="loading-msg">Loading…</p>

          <!-- Empty state -->
          <div v-else-if="!models.length" class="empty-state">
            <p class="empty-title">No models yet</p>
            <p class="empty-sub">Register a surface you've already computed, or train a new model from an enriched dataset.</p>
            <div class="empty-paths">
              <button class="path-card primary" @click="openAddModel('register')">
                <span class="path-head">Register existing asset</span>
                <span class="path-desc">Have a suitability surface already in Earth Engine? Add it here — no data ingest needed.</span>
                <span class="path-cta">Get started →</span>
              </button>
              <button class="path-card" :disabled="!availableDatasets.length" @click="openAddModel('train')">
                <span class="path-head">Train from dataset</span>
                <span class="path-desc">{{ availableDatasets.length ? 'Fit a new MaxEnt model from presence points in an enriched dataset.' : 'Requires an enriched dataset from the Jobs page.' }}</span>
                <span class="path-cta" :class="{ muted: !availableDatasets.length }">
                  {{ availableDatasets.length ? 'Configure model →' : 'No datasets yet' }}
                </span>
              </button>
            </div>
          </div>

          <!-- Model cards -->
          <div v-else class="model-grid">
            <div v-for="m in models" :key="m.id" class="model-card" :class="{ selected: m.selected }">
              <div class="card-top">
                <label class="card-check">
                  <input type="checkbox" v-model="m.selected" />
                  <span class="card-title">{{ m.title }}</span>
                </label>
                <span class="vis-pill" :class="m.visibility">{{ VISIBILITY_LABELS[m.visibility] }}</span>
              </div>

              <div class="card-meta">
                <template v-if="m.predictors?.length">
                  <span>{{ m.predictors.length }} predictors</span>
                  <span class="dot">·</span>
                  <span>{{ (m.background_count || 0).toLocaleString() }} bg pts</span>
                </template>
                <span v-else class="reg-tag">Registered</span>

                <template v-if="m.results?.[0]?.auc != null">
                  <span class="dot">·</span>
                  <span class="auc-val" :class="m.results[0].grade">
                    {{ m.results[0].auc.toFixed(3) }} AUC
                  </span>
                </template>
                <template v-if="m.results?.[0]?.grade">
                  <span class="dot">·</span>
                  <span class="grade-tag" :class="m.results[0].grade">{{ m.results[0].grade }}</span>
                </template>
              </div>

              <p v-if="!m.predictors?.length && m.results?.[0]?.suitability_asset_path" class="asset-path">
                {{ m.results[0].suitability_asset_path }}
              </p>

              <div class="card-footer">
                <span class="card-date">{{ fmtWhen(m.created_at) }}</span>
                <div class="card-actions">
                  <button class="btn small primary" @click="openModelOnMap(m)">View on Map</button>
                  <button class="btn small danger" @click="confirmDelete(m)">Delete</button>
                </div>
              </div>
            </div>
          </div>

          <div class="row">
            <label class="check-row">
              <input type="checkbox" v-model="form.preciseOnly" />
              <span>
                Precise coordinates only
                <span class="field-hint">
                  Recommended for training. iNaturalist randomises obscured locations
                  inside a ~20&nbsp;km cell, so the sampled environment may not match
                  where the species was actually found.
                </span>
              </span>
            </label>
          </div>

          <div class="actions">
            <button class="btn primary" :disabled="pending || !canSubmit" @click="onSubmit">
              {{ pending ? 'Submitting…' : 'Queue Model' }}
            </button>
          </div>

          <!-- Add model trigger (when models exist) -->
          <div v-if="models.length" class="add-row">
            <button class="btn ghost small" @click="toggleAddModel">
              {{ showAddModel ? '▾ Close' : '+ Add model' }}
        </section>

        <!-- ─── Add Model Panel ─────────────────────────────────────────────── -->
        <section v-if="showAddModel" class="panel add-panel">
          <div class="tabs" role="tablist">
            <button role="tab" class="tab" :class="{ active: activeTab === 'register' }"
                    @click="activeTab = 'register'">
              Register existing asset
            </button>
            <button role="tab" class="tab" :class="{ active: activeTab === 'train' }"
                    @click="activeTab = 'train'">
              Train from dataset
            </button>
          </div>

          <!-- ── Register tab ─────────────────────────────────────────────── -->
          <div v-if="activeTab === 'register'" class="tab-body">
            <p class="tab-desc">
              Add a pre-computed suitability surface from your Earth Engine project.
              The asset path is the full project-scoped ID, e.g.
              <code>projects/my-project/assets/redbelted_suitability</code>.
            </p>

            <div class="form-row">
              <label for="reg-title">Model name</label>
              <input id="reg-title" v-model="registerForm.title" type="text"
                     placeholder="e.g. Red-belted Conk — 2024 run" />
            </div>
            <div class="form-row">
              <label for="reg-asset">Earth Engine asset path</label>
              <input id="reg-asset" v-model="registerForm.asset_path" type="text"
                     placeholder="projects/…/assets/…" />
            </div>
            <div class="form-row">
              <label for="reg-desc">Description <em class="opt-label">optional</em></label>
              <textarea id="reg-desc" v-model="registerForm.description"
                        placeholder="Notes about this run…" rows="2"></textarea>
            </div>
            <div class="form-row narrow">
              <label>Visibility</label>
              <select v-model="registerForm.visibility">
                <option v-for="v in visibilities" :key="v" :value="v">{{ VISIBILITY_LABELS[v] }}</option>
              </select>
            </div>

            <div class="form-actions">
              <button class="btn primary"
                      :disabled="pending || !registerForm.title.trim() || !registerForm.asset_path.trim()"
                      @click="onRegister">
                {{ pending ? 'Registering…' : 'Register asset' }}
              </button>
            </div>
            <p v-if="registerError" class="form-msg error">{{ registerError }}</p>
            <p v-if="registerNote" class="form-msg ok">{{ registerNote }}</p>
          </div>

          <!-- ── Train tab ────────────────────────────────────────────────── -->
          <div v-if="activeTab === 'train'" class="tab-body">

            <!-- CSV Import ─────────────────────────────────────────────────── -->
            <div class="csv-section" :class="{ open: showCsvImport }">
              <button class="csv-toggle" @click="showCsvImport = !showCsvImport">
                <span class="toggle-caret">{{ showCsvImport ? '▾' : '▸' }}</span>
                Import observations from a CSV file
                <span v-if="csvImport.rows" class="csv-badge">{{ csvImport.rows.toLocaleString() }} rows</span>
              </button>

              <div v-if="showCsvImport" class="csv-body">
                <p class="tab-desc">
                  Upload an iNaturalist export, GBIF DWC-A, or any CSV with latitude and longitude columns.
                  This creates a new dataset that will be available for training immediately.
                  MaxEnt works best with 50–2,000 presence points — thin large exports first.
                </p>
                <div class="form-row">
                  <label for="csv-file">CSV file</label>
                  <input id="csv-file" type="file" accept=".csv,text/csv" @change="onCsvFile" />
                </div>

                <div v-if="csvImport.headers.length" class="csv-detection">
                  <div class="det-pills">
                    <span class="det-pill" :class="csvImport.latCol ? 'ok' : 'warn'">
                      lat → {{ csvImport.latCol || 'not found' }}
                    </span>
                    <span class="det-pill" :class="csvImport.lonCol ? 'ok' : 'warn'">
                      lon → {{ csvImport.lonCol || 'not found' }}
                    </span>
                    <span v-if="csvImport.dateCol" class="det-pill ok">date → {{ csvImport.dateCol }}</span>
                    <span v-if="csvImport.speciesCol" class="det-pill ok">species → {{ csvImport.speciesCol }}</span>
                    <span class="det-pill rows">{{ csvImport.rows.toLocaleString() }} rows</span>
                  </div>
                  <details class="col-overrides">
                    <summary>Override column mapping</summary>
                    <div class="override-grid">
                      <div class="form-row">
                        <label>Latitude column</label>
                        <select v-model="csvImport.latCol">
                          <option value="">— not set —</option>
                          <option v-for="h in csvImport.headers" :key="h" :value="h">{{ h }}</option>
                        </select>
                      </div>
                      <div class="form-row">
                        <label>Longitude column</label>
                        <select v-model="csvImport.lonCol">
                          <option value="">— not set —</option>
                          <option v-for="h in csvImport.headers" :key="h" :value="h">{{ h }}</option>
                        </select>
                      </div>
                      <div class="form-row">
                        <label>Date column <em class="opt-label">optional</em></label>
                        <select v-model="csvImport.dateCol">
                          <option value="">— none —</option>
                          <option v-for="h in csvImport.headers" :key="h" :value="h">{{ h }}</option>
                        </select>
                      </div>
                      <div class="form-row">
                        <label>Species column <em class="opt-label">optional</em></label>
                        <select v-model="csvImport.speciesCol">
                          <option value="">— none —</option>
                          <option v-for="h in csvImport.headers" :key="h" :value="h">{{ h }}</option>
                        </select>
                      </div>
                    </div>
                  </details>
                </div>

                <template v-if="csvImport.rows > 0">
                  <div class="form-row">
                    <label for="csv-ds-title">Dataset name</label>
                    <input id="csv-ds-title" v-model="csvImport.title" type="text"
                           placeholder="e.g. Red-belted Conk — iNat 2024 export" />
                  </div>
                  <div class="form-row narrow">
                    <label>Visibility</label>
                    <select v-model="csvImport.visibility">
                      <option v-for="v in visibilities" :key="v" :value="v">{{ VISIBILITY_LABELS[v] }}</option>
                    </select>
                  </div>
                  <div class="form-actions">
                    <button class="btn primary"
                            :disabled="csvImport.pending || !csvImport.latCol || !csvImport.lonCol || !csvImport.title.trim()"
                            @click="onImportCsv">
                      {{ csvImport.pending ? 'Importing…' : `Import ${csvImport.rows.toLocaleString()} observations` }}
                    </button>
                    <span v-if="!csvImport.latCol || !csvImport.lonCol" class="form-hint">
                      Override column mapping above to set lat/lon.
                    </span>
                  </div>
                  <p v-if="csvImport.error" class="form-msg error">{{ csvImport.error }}</p>
                  <p v-if="csvImport.note" class="form-msg ok">{{ csvImport.note }}</p>
                </template>
              </div>
            </div>
            <!-- ─────────────────────────────────────────────────────────────── -->

            <div v-if="!availableDatasets.length && !showCsvImport" class="no-datasets">
              <p>
                No datasets available yet. Import a CSV above, or create one by running an enrichment job on the
                <NuxtLink to="/jobs">Jobs page</NuxtLink>.
              </p>
            </div>
            <template v-if="availableDatasets.length">
              <div class="form-divider">Train from an existing dataset</div>
              <div class="form-row">
                <label for="model-title">Model name</label>
                <input id="model-title" v-model="form.title" type="text"
                       placeholder="e.g. Red-belted Conk Suitability" />
              </div>
              <div class="form-row">
                <label for="model-desc">Description <em class="opt-label">optional</em></label>
                <textarea id="model-desc" v-model="form.description"
                          placeholder="Notes about the specific environment or year…" rows="2"></textarea>
              </div>
              <div class="form-row">
                <label>Source dataset</label>
                <select v-model="form.sourceDatasetId">
                  <option value="" disabled>Choose a dataset…</option>
                  <option v-for="d in availableDatasets" :key="d.id" :value="d.id">
                    {{ d.title }}{{ d.feature_count ? ` · ${d.feature_count.toLocaleString()} points` : '' }}
                  </option>
                </select>
              </div>
              <div class="form-row">
                <label>
                  <GlossaryTooltip term="predictors" :definition="g('predictors')">Predictors</GlossaryTooltip>
                  <span class="field-hint">Select at least {{ MIN_PREDICTORS }}</span>
                </label>
                <div class="pred-grid">
                  <label v-for="p in predictorList" :key="p.key" class="pred-item"
                         :title="PREDICTOR_DESCRIPTIONS[p.key]">
                    <input type="checkbox" :value="p.key" v-model="form.predictors" />
                    <span class="pred-text">
                      <strong>{{ p.label }}</strong>
                      <em>{{ PREDICTOR_DESCRIPTIONS[p.key] }}</em>
                    </span>
                  </label>
                </div>
              </div>
              <div class="form-pair">
                <div class="form-row">
                  <label for="bg-count">
                    <GlossaryTooltip term="background points" :definition="g('background points')">Background points</GlossaryTooltip>
                    <span class="field-hint">{{ MIN_BACKGROUND }}–{{ MAX_BACKGROUND }}</span>
                  </label>
                  <input id="bg-count" v-model.number="form.backgroundCount" type="number"
                         :min="MIN_BACKGROUND" :max="MAX_BACKGROUND" step="100" />
                  <span class="field-note">More = slower but more stable. 1,000 is a good start.</span>
                </div>
                <div class="form-row">
                  <label>Visibility</label>
                  <select v-model="form.visibility">
                    <option v-for="v in visibilities" :key="v" :value="v">{{ VISIBILITY_LABELS[v] }}</option>
                  </select>
                </div>
              </div>
              <div class="form-actions">
                <button class="btn primary" :disabled="pending || !canSubmit" @click="onSubmit">
                  {{ pending ? 'Submitting…' : 'Queue model' }}
                </button>
                <span v-if="form.predictors.length < MIN_PREDICTORS" class="form-hint">
                  Pick at least {{ MIN_PREDICTORS }} predictors.
                </span>
                <span v-else-if="!form.sourceDatasetId" class="form-hint">
                  Select a source dataset.
                </span>
              </div>
              <p v-if="error" class="form-msg error">{{ error }}</p>
              <p v-if="submitNote" class="form-msg ok">{{ submitNote }}</p>
            </template>
          </div>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
import { computed, nextTick, reactive, ref, onMounted } from 'vue'
import { PREDICTOR_KEYS, MAXENT_PREDICTORS, MIN_PREDICTORS, MAX_BACKGROUND, MIN_BACKGROUND, DEFAULT_PREDICTORS } from '~/netlify/lib/maxent.mjs'
import { useGlossary } from '~/composables/useGlossary'
import GlossaryTooltip from '~/components/GlossaryTooltip.vue'
import MaxEntTutorial from '~/components/MaxEntTutorial.vue'
import { VISIBILITY_LABELS } from '~/composables/useDatasets'
import ModelComparison from '~/components/ModelComparison.vue'

const { define: g } = useGlossary()
const tutorialRef = ref(null)

const PREDICTOR_DESCRIPTIONS = {
  elevation: 'Height above sea level (SRTM). Strong driver of temperature, moisture and vegetation zones.',
  slope: 'Steepness of terrain. Affects drainage, disturbance regime and micro-climate.',
  aspect: 'Direction a slope faces. Controls sun exposure and moisture retention.',
  ndvi: 'Vegetation greenness index from Sentinel-2 imagery (multi-year median). Proxy for habitat quality and food availability.',
  soil_moisture: 'Long-run average top-layer soil wetness from ERA5-Land. Use for moisture-dependent species.',
  precip_normal: 'Mean daily rainfall from CHIRPS (climate normal). Use for species with strong precipitation limits.',
  temp_normal: 'Mean daily 2 m air temperature from ERA5-Land (climate normal). Often the strongest climate predictor of range limits.',
}

const { models, activeJob, pending, error, fetchModels, trainModel, deleteModel, registerAsset } = useMaxEnt()
const { available: availableDatasets, refreshAvailable } = useDatasets()
const membership = useMembership()
const modelOverlay = useModelOverlay()
const router = useRouter()
const loading = ref(false)
const submitNote = ref('')
const showComparison = ref(false)
const selectedModels = computed(() => models.value.filter(m => m.selected))

// ── Add-model panel state ─────────────────────────────────────────────────
const showAddModel = ref(false)
const activeTab = ref('register')

function openAddModel(tab) {
  activeTab.value = tab
  showAddModel.value = true
  nextTick(() => {
    document.querySelector('.add-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  })
}

function toggleAddModel() {
  showAddModel.value = !showAddModel.value
  if (showAddModel.value) {
    nextTick(() => {
      document.querySelector('.add-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    })
  }
}

// ── Register form ─────────────────────────────────────────────────────────
const registerForm = reactive({ title: '', description: '', asset_path: '', visibility: 'private' })
const registerNote = ref('')
const registerError = ref('')

async function onRegister() {
  registerNote.value = ''
  registerError.value = ''
  const result = await registerAsset({ ...registerForm })
  if (result.ok) {
    registerNote.value = 'Asset registered successfully!'
    registerForm.title = ''
    registerForm.description = ''
    registerForm.asset_path = ''
    showAddModel.value = false
  } else {
    registerError.value = result.error || 'Registration failed.'
  }
}

// ── Training form ─────────────────────────────────────────────────────────
const visibilities = ['private', 'members', 'public']

const form = reactive({
  title: '',
  description: '',
  sourceDatasetId: '',
  predictors: [...DEFAULT_PREDICTORS],
  backgroundCount: 1000,
  visibility: 'private',
  preciseOnly: true,
})

const predictorList = PREDICTOR_KEYS.map((key) => ({ key, label: MAXENT_PREDICTORS[key].label }))

const canSubmit = computed(() =>
  form.title.trim() && form.sourceDatasetId && form.predictors.length >= MIN_PREDICTORS
)

async function onSubmit() {
  submitNote.value = ''
  const spec = {
    title: form.title,
    description: form.description,
    source_dataset_id: form.sourceDatasetId,
    predictors: form.predictors,
    background: form.backgroundCount,
    visibility: form.visibility,
    precise_only: form.preciseOnly,
  }
  const result = await trainModel(spec)
  if (result.ok) {
    submitNote.value = 'Model submitted — it will appear in your list when training finishes.'
    form.title = ''
    form.description = ''
  }
}

// ── CSV import ────────────────────────────────────────────────────────────
const { accessToken } = useAuth()
const showCsvImport = ref(false)

// Column auto-detection (mirrors the backend logic so the UI gives instant feedback).
const LAT_CANDIDATES = ['latitude', 'lat', 'decimallatitude', 'decimal_latitude', 'y']
const LON_CANDIDATES = ['longitude', 'lon', 'lng', 'decimallongitude', 'decimal_longitude', 'x']
const DATE_CANDIDATES = ['observed_on', 'date', 'eventdate', 'event_date', 'observedon', 'dateidentified', 'date_observed']
const SPECIES_CANDIDATES = ['scientific_name', 'scientificname', 'taxon_name', 'species', 'taxon', 'name', 'taxon_species_name']

function detectHeader(headers, candidates) {
  return candidates.find(c => headers.includes(c)) || ''
}

const csvImport = reactive({
  rawText: '',
  headers: [] as string[],
  rows: 0,
  latCol: '',
  lonCol: '',
  dateCol: '',
  speciesCol: '',
  title: '',
  visibility: 'private',
  pending: false,
  error: '',
  note: '',
})

function onCsvFile(event: Event) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (!file) return
  csvImport.error = ''
  csvImport.note = ''
  csvImport.rows = 0
  csvImport.headers = []
  const reader = new FileReader()
  reader.onload = (e) => {
    const text = String(e.target?.result || '')
    const firstLine = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')[0]
    if (!firstLine) { csvImport.error = 'Could not read CSV headers.'; return }
    // Simple header parse (handles quoted headers but not mid-value commas).
    const headers = firstLine.split(',').map(h => h.replace(/^"|"$/g, '').trim().toLowerCase())
    const lineCount = text.split('\n').filter(l => l.trim()).length - 1
    csvImport.rawText = text
    csvImport.headers = headers
    csvImport.rows = Math.max(0, lineCount)
    csvImport.latCol = detectHeader(headers, LAT_CANDIDATES)
    csvImport.lonCol = detectHeader(headers, LON_CANDIDATES)
    csvImport.dateCol = detectHeader(headers, DATE_CANDIDATES)
    csvImport.speciesCol = detectHeader(headers, SPECIES_CANDIDATES)
    if (!csvImport.title && file.name) csvImport.title = file.name.replace(/\.csv$/i, '')
  }
  reader.readAsText(file)
}

async function onImportCsv() {
  csvImport.pending = true
  csvImport.error = ''
  csvImport.note = ''
  try {
    const token = await accessToken()
    const res = await fetch('/.netlify/functions/datasets', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...(token ? { authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify({
        action: 'import_csv',
        csv: csvImport.rawText,
        title: csvImport.title,
        lat_col: csvImport.latCol,
        lon_col: csvImport.lonCol,
        date_col: csvImport.dateCol || undefined,
        species_col: csvImport.speciesCol || undefined,
        visibility: csvImport.visibility,
      }),
    })
    const data = await res.json()
    if (!data.ok) throw new Error(data.error || 'Import failed.')
    const skippedNote = data.skipped ? ` (${data.skipped} rows skipped for bad coordinates)` : ''
    csvImport.note = `Imported ${data.dataset.feature_count?.toLocaleString()} observations as "${data.dataset.title}"${skippedNote}.`
    csvImport.rawText = ''
    csvImport.rows = 0
    csvImport.headers = []
    csvImport.title = ''
    showCsvImport.value = false
    await refreshAvailable()
    // Auto-select the new dataset in the training form.
    form.sourceDatasetId = data.dataset.id
  } catch (e: any) {
    csvImport.error = e.message
  } finally {
    csvImport.pending = false
  }
}

// ── Model actions ─────────────────────────────────────────────────────────
async function confirmDelete(model) {
  if (confirm(`Delete "${model.title}"?`)) {
    const res = await deleteModel(model.id)
    if (res.ok) await fetchModels()
  }
}

function openModelOnMap(model) {
  modelOverlay.open({ configId: model.id, label: model.title || 'Suitability' })
  router.push('/map')
}

const progress = computed(() => {
  if (!activeJob.value) return 0
  if (activeJob.value.status === 'succeeded') return 100
  if (activeJob.value.status === 'failed') return 0
  return 50
})

function fmtWhen(dateStr) {
  if (!dateStr) return ''
  return new Date(dateStr).toLocaleDateString()
}

onMounted(async () => {
  await Promise.all([fetchModels(), refreshAvailable()])
  // Open the add-model panel by default when the user has no models yet.
  if (!models.value.length) {
    showAddModel.value = true
    activeTab.value = availableDatasets.value.length ? 'train' : 'register'
  }
})
</script>

<style scoped>
/* ── Base utilities ──────────────────────────────────────────────────────── */
.btn {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 14px; font: inherit; font-size: 0.82rem; cursor: pointer;
  display: inline-flex; align-items: center; gap: 5px;
}
.btn:hover:not(:disabled) { border-color: var(--muted); }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn.primary { background: var(--accent, #34c46a); color: #fff; border-color: transparent; }
.btn.primary:hover:not(:disabled) { opacity: 0.9; }
.btn.ghost { background: transparent; }
.btn.small { padding: 4px 10px; font-size: 0.78rem; }
.btn.danger { color: #b3492f; }
.btn.danger:hover:not(:disabled) { border-color: #b3492f; }
.btn-text {
  background: none; border: none; color: var(--muted); font: inherit; font-size: 0.78rem;
  cursor: pointer; padding: 0; text-decoration: underline; flex-shrink: 0;
}
.btn-text:hover { color: var(--text); }

/* ── Page ────────────────────────────────────────────────────────────────── */
.modeling { padding: 18px 20px; max-width: 1100px; margin: 0 auto; }

.head {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 16px; margin-bottom: 24px;
}
.head-text h2 { margin: 0 0 4px; }
.head-text .sub { margin: 0; color: var(--muted); font-size: 0.86rem; line-height: 1.5; max-width: 64ch; }
.head-actions { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }

/* ── Panel ───────────────────────────────────────────────────────────────── */
.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; margin-bottom: 16px; overflow: hidden;
}
.panel-head {
  display: flex; justify-content: space-between; align-items: center;
  padding: 16px 20px 14px; border-bottom: 1px solid var(--border);
}
.panel-head h3 { margin: 0; font-size: 0.95rem; }

/* ── Active training banner ──────────────────────────────────────────────── */
.job-banner {
  display: flex; align-items: flex-start; justify-content: space-between; gap: 12px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 12px 16px; margin-bottom: 14px;
  border-left: 3px solid var(--accent, #34c46a);
}
.banner-body { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 8px; }
.banner-info { display: flex; align-items: center; gap: 10px; }
.banner-name { font-size: 0.88rem; font-weight: 600; }
.banner-bar { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent, #34c46a); transition: width 0.4s ease; border-radius: 3px; }
.banner-error { margin: 4px 0 0; font-size: 0.78rem; color: #b3492f; }

.status-pill {
  font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 2px 7px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted);
}
.status-pill.running, .status-pill.queued { border-color: #3d6b8b; color: #3d6b8b; }
.status-pill.succeeded { border-color: #3d8b5f; color: #3d8b5f; }
.status-pill.failed { border-color: #b3492f; color: #b3492f; }

/* ── Loading ─────────────────────────────────────────────────────────────── */
.loading-msg { padding: 28px 20px; margin: 0; color: var(--muted); font-size: 0.86rem; }

/* ── Empty state ─────────────────────────────────────────────────────────── */
.empty-state { padding: 24px 20px; }
.empty-title { margin: 0 0 4px; font-weight: 600; font-size: 0.95rem; }
.empty-sub { margin: 0 0 20px; color: var(--muted); font-size: 0.84rem; line-height: 1.5; }

.empty-paths { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.path-card {
  display: flex; flex-direction: column; gap: 6px; text-align: left;
  padding: 16px 18px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--surface-2); cursor: pointer; font: inherit;
  transition: border-color 0.15s;
}
.path-card:hover:not(:disabled) { border-color: var(--muted); }
.path-card:disabled { opacity: 0.55; cursor: default; }
.path-card.primary { border-color: var(--accent, #34c46a); }
.path-card.primary:hover:not(:disabled) { background: color-mix(in srgb, var(--accent, #34c46a) 6%, var(--surface-2)); }
.path-head { font-weight: 600; font-size: 0.9rem; color: var(--text-strong, var(--text)); }
.path-card.primary .path-head { color: var(--accent, #34c46a); }
.path-desc { font-size: 0.8rem; color: var(--muted); line-height: 1.45; }
.path-cta { font-size: 0.78rem; font-weight: 600; color: var(--muted); margin-top: auto; }
.path-card.primary .path-cta { color: var(--accent, #34c46a); }
.path-cta.muted { color: var(--muted); opacity: 0.7; }

/* ── Model grid ──────────────────────────────────────────────────────────── */
.model-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(290px, 1fr));
  gap: 1px; background: var(--border);
}
.model-card {
  display: flex; flex-direction: column; gap: 8px;
  padding: 16px 18px; background: var(--surface);
  transition: background 0.12s;
}
.model-card:hover { background: var(--surface-2); }
.model-card.selected { background: color-mix(in srgb, var(--accent, #34c46a) 6%, var(--surface)); }

.card-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.card-check { display: flex; align-items: center; gap: 8px; cursor: pointer; flex: 1; min-width: 0; }
.card-check input[type="checkbox"] { flex-shrink: 0; margin: 0; }
.card-title { font-weight: 600; font-size: 0.9rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.vis-pill {
  font-size: 0.67rem; padding: 2px 7px; border-radius: 999px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.03em; flex-shrink: 0;
  border: 1px solid var(--border); color: var(--muted);
}
.vis-pill.public { border-color: #3d6b8b; color: #3d6b8b; }
.vis-pill.members { border-color: #3d8b5f; color: #3d8b5f; }

.card-meta { display: flex; align-items: center; flex-wrap: wrap; gap: 4px; font-size: 0.8rem; color: var(--muted); }
.dot { color: var(--border); }

.reg-tag {
  font-size: 0.72rem; padding: 1px 7px; border-radius: 4px; font-weight: 600;
  background: color-mix(in srgb, var(--accent, #34c46a) 12%, transparent);
  color: var(--accent, #34c46a);
}
.auc-val { font-weight: 600; }
.auc-val.excellent, .auc-val.good { color: #3d8b5f; }
.auc-val.fair { color: #8a5a1f; }
.auc-val.weak { color: #b3492f; }

.grade-tag {
  font-size: 0.68rem; padding: 1px 6px; border-radius: 4px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.04em;
  border: 1px solid currentColor;
}
.grade-tag.excellent { color: #3d8b5f; }
.grade-tag.good { color: #3d8b5f; }
.grade-tag.fair { color: #8a5a1f; }
.grade-tag.weak { color: #b3492f; }

.asset-path {
  margin: 0; font-size: 0.72rem; color: var(--muted); font-family: ui-monospace, monospace;
  word-break: break-all; line-height: 1.4;
}

.card-footer {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; margin-top: auto; padding-top: 4px;
}
.card-date { font-size: 0.74rem; color: var(--muted); }
.card-actions { display: flex; gap: 6px; }

/* ── Add row ─────────────────────────────────────────────────────────────── */
.add-row { padding: 12px 18px; border-top: 1px solid var(--border); }

/* ── Add model panel ─────────────────────────────────────────────────────── */
.add-panel { overflow: visible; }

.tabs {
  display: flex; border-bottom: 1px solid var(--border);
}
.tab {
  padding: 12px 18px; font: inherit; font-size: 0.84rem; font-weight: 500;
  background: none; border: none; border-bottom: 2px solid transparent;
  margin-bottom: -1px; cursor: pointer; color: var(--muted);
  transition: color 0.12s, border-color 0.12s;
}
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent, #34c46a); border-bottom-color: var(--accent, #34c46a); }

.tab-body { padding: 20px; }
.tab-desc {
  margin: 0 0 18px; font-size: 0.82rem; color: var(--muted); line-height: 1.55;
}
.tab-desc code {
  font-family: ui-monospace, monospace; font-size: 0.78rem;
  background: var(--surface-2); padding: 1px 5px; border-radius: 3px; border: 1px solid var(--border);
}

/* ── Form ────────────────────────────────────────────────────────────────── */
.form-row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
.form-row label { font-size: 0.8rem; font-weight: 600; color: var(--muted); display: flex; align-items: baseline; gap: 6px; }
.form-row.narrow { max-width: 240px; }

.opt-label { font-size: 0.74rem; font-weight: 400; color: var(--muted); font-style: italic; }
.field-hint { font-weight: 400; color: var(--muted); font-size: 0.74rem; }
.field-note { font-size: 0.74rem; color: var(--muted); line-height: 1.4; }

input[type="text"], input[type="number"], select, textarea {
  padding: 8px 11px; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 6px; font: inherit; font-size: 0.86rem;
}
textarea { resize: vertical; min-height: 56px; }
input:focus, select:focus, textarea:focus {
  outline: none; border-color: var(--accent, #34c46a);
}

.form-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

.pred-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 6px; margin-top: 4px;
}
.pred-item {
  display: flex; align-items: flex-start; gap: 8px; padding: 7px 10px;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
  font: inherit;
}
.pred-item:hover { background: var(--surface-3, var(--border)); }
.pred-item input[type="checkbox"] { margin-top: 2px; flex-shrink: 0; }
.pred-text { display: flex; flex-direction: column; gap: 2px; }
.pred-text strong { font-size: 0.82rem; }
.pred-text em { font-size: 0.72rem; color: var(--muted); font-style: normal; line-height: 1.3; }

.form-actions { display: flex; align-items: center; gap: 10px; padding-top: 4px; }
.form-hint { font-size: 0.76rem; color: var(--muted); }
.field-hint { font-weight: 400; color: var(--muted); font-size: 0.76rem; margin-left: 4px; }
.field-note { font-size: 0.76rem; color: var(--muted); line-height: 1.4; margin-top: 4px; }
.check-row { display: flex; align-items: flex-start; gap: 8px; font-size: 0.82rem; font-weight: 600;
  color: var(--text); cursor: pointer; }
.check-row input[type="checkbox"] { margin-top: 2px; flex-shrink: 0; }
.check-row .field-hint { display: block; margin: 2px 0 0; }

.form-msg { margin: 10px 0 0; font-size: 0.8rem; }
.form-msg.error { color: #b3492f; }
.form-msg.ok { color: #3d8b5f; }

.no-datasets {
  padding: 16px 20px; background: var(--surface-2); border: 1px dashed var(--border);
  border-radius: 8px; font-size: 0.84rem; color: var(--muted); line-height: 1.5; margin-bottom: 12px;
}
.no-datasets a { color: var(--accent, #34c46a); text-decoration: none; }
.no-datasets a:hover { text-decoration: underline; }

.form-divider {
  font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
  color: var(--muted); margin: 16px 0 14px; padding-top: 14px;
  border-top: 1px solid var(--border);
}

/* ── CSV import ──────────────────────────────────────────────────────────── */
.csv-section { border: 1px solid var(--border); border-radius: 8px; margin-bottom: 14px; overflow: hidden; }
.csv-section.open { border-color: color-mix(in srgb, var(--accent, #34c46a) 30%, var(--border)); }
.csv-toggle {
  width: 100%; text-align: left; background: var(--surface-2); border: none;
  padding: 10px 14px; font: inherit; font-size: 0.84rem; font-weight: 500; color: var(--text);
  cursor: pointer; display: flex; align-items: center; gap: 8px;
}
.csv-toggle:hover { background: var(--surface-3, var(--surface-2)); }
.toggle-caret { font-size: 0.68rem; color: var(--muted); }
.csv-badge {
  margin-left: auto; font-size: 0.7rem; padding: 1px 7px; border-radius: 999px; font-weight: 600;
  background: color-mix(in srgb, var(--accent, #34c46a) 12%, transparent);
  color: var(--accent, #34c46a);
}
.csv-body { padding: 16px 18px; border-top: 1px solid var(--border); }

.csv-detection { margin-bottom: 14px; }
.det-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.det-pill {
  font-size: 0.72rem; padding: 2px 8px; border-radius: 4px; font-family: ui-monospace, monospace;
  border: 1px solid var(--border); color: var(--muted);
}
.det-pill.ok { border-color: #3d8b5f; color: #3d8b5f; }
.det-pill.warn { border-color: #b3492f; color: #b3492f; }
.det-pill.rows { border-color: var(--border); color: var(--muted); font-family: inherit; }

.col-overrides { margin-top: 6px; }
.col-overrides summary { font-size: 0.78rem; color: var(--muted); cursor: pointer; }
.col-overrides summary:hover { color: var(--text); }
.override-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }

input[type="file"] {
  font: inherit; font-size: 0.84rem; color: var(--muted); cursor: pointer;
}

/* ── Gate ────────────────────────────────────────────────────────────────── */
.gate {
  padding: 40px; text-align: center; background: var(--surface);
  border: 1px solid var(--border); border-radius: 12px;
}

/* ── Overlay ─────────────────────────────────────────────────────────────── */
.overlay {
  position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,0.6);
  display: flex; align-items: center; justify-content: center; padding: 20px;
}
.overlay-content {
  background: var(--surface); border-radius: 12px; width: 100%; max-width: 1000px;
  max-height: 90vh; overflow-y: auto; position: relative; padding: 20px;
}
.close-btn {
  position: absolute; top: 12px; right: 12px; background: var(--surface-2);
  color: var(--text); border: 1px solid var(--border); border-radius: 50%;
  width: 26px; height: 26px; cursor: pointer; font-size: 12px;
}

/* ── Responsive ──────────────────────────────────────────────────────────── */
@media (max-width: 820px) {
  .modeling { padding: 14px 16px; }
  .head { flex-direction: column; gap: 10px; }
  .head-actions { align-self: flex-start; }
  .empty-paths { grid-template-columns: 1fr; }
  .model-grid { grid-template-columns: 1fr; }
  .form-pair { grid-template-columns: 1fr; gap: 0; }
  .overlay { padding: 10px; }
  .overlay-content { max-height: 94vh; padding: 16px; }
}

@media (max-width: 480px) {
  .pred-grid { grid-template-columns: 1fr; }
  .form-actions { flex-direction: column; align-items: stretch; }
  .form-actions .btn { justify-content: center; }
  .card-actions { flex-wrap: wrap; }
  .card-actions .btn { flex: 1; justify-content: center; }
  .tab { padding: 10px 12px; font-size: 0.8rem; }
}

@media (pointer: coarse) {
  input[type="text"], input[type="number"], select, textarea {
    padding: 11px 12px; font-size: 16px;
  }
  .btn { min-height: 44px; }
  .btn.small { min-height: 36px; }
  .pred-item { padding: 10px 12px; }
  .pred-item input[type="checkbox"],
  .card-check input[type="checkbox"] { width: 20px; height: 20px; }
  .close-btn { width: 36px; height: 36px; font-size: 14px; }
}
</style>
