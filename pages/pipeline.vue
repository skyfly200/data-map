<template>
  <div class="pipeline-page">
    <div class="page-head">
      <h2>Model Pipeline</h2>
      <p class="sub">Import observations → enrich with environment data → analyse predictors → train a habitat model → evaluate results.</p>
    </div>

    <!-- Step indicator -->
    <nav class="steps">
      <button v-for="(s, i) in STEPS" :key="s.key" class="step-btn"
              :class="{ active: step === s.key, done: completedSteps.has(s.key), reachable: canReach(s.key) }"
              :disabled="!canReach(s.key)"
              @click="goStep(s.key)">
        <span class="step-num">{{ i + 1 }}</span>
        <span class="step-label">{{ s.label }}</span>
      </button>
    </nav>

    <ClientOnly>
      <!-- Auth gate -->
      <div v-if="membership.configured && !membership.isAuthed.value" class="gate">
        <p>Sign in to use the model pipeline.</p>
        <NuxtLink to="/login" class="btn primary">Sign in</NuxtLink>
      </div>
      <div v-else-if="membership.configured && !membership.isMember.value" class="gate">
        <p>The model pipeline is a membership benefit.</p>
        <button class="btn" :disabled="rechecking" @click="recheck">{{ rechecking ? 'Checking…' : 'Just joined? Check again' }}</button>
      </div>

      <template v-else>
        <!-- ═══════════════════════════════════════════════════════
             STEP 1 — SOURCE
        ════════════════════════════════════════════════════════ -->
        <section v-if="step === 'source'" class="panel">
          <h3>Observations</h3>
          <p class="hint">Choose where your species observations come from.</p>

          <div class="row">
            <label>Source type</label>
            <div class="sources">
              <label class="pick">
                <input v-model="sourceForm.type" type="radio" value="dataset" />
                <span>Use a saved dataset</span>
              </label>
              <label class="pick">
                <input v-model="sourceForm.type" type="radio" value="bbox" />
                <span>Filter by area, date &amp; taxon</span>
              </label>
              <label class="pick">
                <input v-model="sourceForm.type" type="radio" value="fetch" />
                <span>Import from iNaturalist / GBIF</span>
              </label>
            </div>
          </div>

          <!-- Saved dataset -->
          <template v-if="sourceForm.type === 'dataset'">
            <div class="row">
              <label for="src-dataset">Dataset</label>
              <select id="src-dataset" v-model="sourceForm.datasetSlug">
                <option value="" disabled>Choose one…</option>
                <option v-for="d in datasetsApi.datasets.value" :key="d.id" :value="d.slug">
                  {{ d.title }}
                </option>
              </select>
            </div>
            <p v-if="!datasetsApi.datasets.value.length" class="hint warn">
              No saved datasets yet — try "Filter by area" or go to <NuxtLink to="/data">Data</NuxtLink> to import observations first.
            </p>
          </template>

          <!-- BBox -->
          <template v-else-if="sourceForm.type === 'bbox'">
            <div class="row">
              <label>Bounding box</label>
              <div class="bbox">
                <label class="mini">N <input v-model.number="sourceForm.north" type="number" step="0.1" /></label>
                <label class="mini">S <input v-model.number="sourceForm.south" type="number" step="0.1" /></label>
                <label class="mini">W <input v-model.number="sourceForm.west" type="number" step="0.1" /></label>
                <label class="mini">E <input v-model.number="sourceForm.east" type="number" step="0.1" /></label>
              </div>
            </div>
            <div class="row two">
              <label class="stack"><span>From</span><input v-model="sourceForm.dateFrom" type="date" /></label>
              <label class="stack"><span>To</span><input v-model="sourceForm.dateTo" type="date" /></label>
            </div>
            <div class="row">
              <label for="src-taxon">Taxon (optional)</label>
              <input id="src-taxon" v-model="sourceForm.taxon" type="text" placeholder="e.g. Amanita" />
            </div>
          </template>

          <!-- Fetch / import inline -->
          <template v-else-if="sourceForm.type === 'fetch'">
            <div class="row">
              <label for="fetch-taxon">Taxon</label>
              <input id="fetch-taxon" v-model="fetchForm.taxon" type="text"
                     placeholder="e.g. Morchella, Amanitaceae" :disabled="fetchForm.loading" />
            </div>
            <div class="row">
              <label>Source</label>
              <div class="src-tabs">
                <button v-for="s in FETCH_SOURCES" :key="s.key" class="src-tab"
                        :class="{ on: fetchForm.source === s.key }"
                        @click="fetchForm.source = s.key" type="button">{{ s.label }}</button>
              </div>
            </div>
            <div class="row two">
              <label class="stack"><span>From</span><input v-model="fetchForm.dateFrom" type="date" :disabled="fetchForm.loading" /></label>
              <label class="stack"><span>To</span><input v-model="fetchForm.dateTo" type="date" :disabled="fetchForm.loading" /></label>
            </div>
            <div class="actions" style="margin-top:0">
              <button class="btn secondary" :disabled="!fetchForm.taxon.trim() || fetchForm.loading"
                      @click="runFetch" type="button">
                {{ fetchForm.loading ? 'Fetching…' : 'Fetch observations' }}
              </button>
            </div>
            <p v-if="fetchForm.error" class="msg error">{{ fetchForm.error }}</p>
            <p v-if="fetchForm.result" class="msg ok">
              {{ fetchForm.result.count }} records fetched as "{{ fetchForm.result.title }}" — ready to continue.
            </p>
          </template>

          <div class="actions">
            <button class="btn primary" :disabled="!canAdvanceSource" @click="advanceSource">
              Continue →
            </button>
          </div>
        </section>

        <!-- ═══════════════════════════════════════════════════════
             STEP 2 — ENRICH
        ════════════════════════════════════════════════════════ -->
        <section v-else-if="step === 'enrich'" class="panel">
          <h3>Enrich with environment data</h3>
          <p class="hint">
            Sample terrain, climate, and vegetation values at each observation point.
            These become the predictors for the habitat model.
          </p>

          <div class="row">
            <label>Job name</label>
            <input v-model="enrichForm.title" type="text" placeholder="Autumn foray enrichment" />
          </div>

          <div class="row">
            <label>Environment layers</label>
            <div class="stages">
              <label v-for="s in stageList" :key="s.key" class="stage">
                <input type="checkbox" :value="s.key" v-model="enrichForm.stages" />
                <span><strong>{{ s.label }}</strong><em>{{ s.description }}</em></span>
              </label>
            </div>
          </div>

          <p v-if="enrichError" class="msg error">{{ enrichError }}</p>
          <p v-if="enrichNote" class="msg ok">{{ enrichNote }}</p>

          <div class="actions">
            <button class="btn secondary" @click="goStep('source')">← Back</button>
            <template v-if="!enrichJobId">
              <button class="btn primary"
                      :disabled="!enrichForm.stages.length || jobsApi.submitting.value"
                      @click="submitEnrich">
                {{ jobsApi.submitting.value ? 'Queuing…' : 'Start enrichment' }}
              </button>
            </template>
            <template v-else>
              <span class="job-status" :class="enrichJobStatus">{{ enrichStatusLabel }}</span>
              <button v-if="enrichJobStatus === 'succeeded'" class="btn primary" @click="goStep('explore')">
                Analyse predictors →
              </button>
            </template>
          </div>

          <!-- Live job progress -->
          <div v-if="enrichJob" class="job-progress">
            <div class="prog-bar"><div class="prog-fill" :style="{ width: (enrichJob.progress || 0) + '%' }"></div></div>
            <p class="prog-msg">{{ enrichJob.message || enrichJob.stage || enrichStatusLabel }}</p>
          </div>
        </section>

        <!-- ═══════════════════════════════════════════════════════
             STEP 3 — EXPLORE (pre-training analysis)
        ════════════════════════════════════════════════════════ -->
        <section v-else-if="step === 'explore'" class="panel">
          <h3>Analyse predictors</h3>
          <p class="hint">
            Review which environment variables have good coverage and are not highly correlated
            before committing to a full training run.
          </p>

          <template v-if="!exploreData">
            <p class="msg">Loading enriched dataset…</p>
          </template>
          <template v-else>
            <!-- Coverage table -->
            <div class="explore-section">
              <h4>Variable coverage</h4>
              <p class="hint">How many of your {{ exploreData.total }} observations have each variable sampled.</p>
              <table class="cov-table">
                <thead><tr><th>Variable</th><th>Coverage</th><th>Recommend</th></tr></thead>
                <tbody>
                  <tr v-for="v in exploreData.coverage" :key="v.key" :class="{ low: v.pct < 60 }">
                    <td>{{ v.label }}</td>
                    <td>
                      <span class="cov-bar-wrap">
                        <span class="cov-bar" :style="{ width: v.pct + '%', background: v.pct >= 80 ? '#22c55e' : v.pct >= 60 ? '#f59e0b' : '#ef4444' }"></span>
                        <span class="cov-pct">{{ v.pct.toFixed(0) }}%</span>
                      </span>
                    </td>
                    <td>
                      <span v-if="v.pct >= 80" class="badge ok">✓ Use</span>
                      <span v-else-if="v.pct >= 60" class="badge warn">Consider</span>
                      <span v-else class="badge bad">Skip</span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Correlation matrix -->
            <div v-if="exploreData.correlations.length" class="explore-section">
              <h4>Correlation between predictors</h4>
              <p class="hint">High correlation (|r| &gt; 0.7) between two variables means they carry redundant information — prefer one over the other.</p>
              <table class="corr-table">
                <thead>
                  <tr>
                    <th></th>
                    <th v-for="v in exploreData.coverage" :key="v.key">{{ v.shortLabel }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, ri) in exploreData.correlations" :key="ri">
                    <td class="row-label">{{ exploreData.coverage[ri]?.shortLabel }}</td>
                    <td v-for="(val, ci) in row" :key="ci"
                        class="corr-cell"
                        :class="{ high: Math.abs(val) > 0.7 && ri !== ci }"
                        :style="{ background: corrColor(val, ri === ci) }">
                      {{ ri === ci ? '—' : val.toFixed(2) }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!-- Auto-recommended predictors -->
            <div class="explore-section">
              <h4>Recommended predictors</h4>
              <p class="hint">Based on coverage and low redundancy — you can change these in the next step.</p>
              <div class="stages">
                <label v-for="p in predictorList" :key="p.key" class="stage">
                  <input type="checkbox" :value="p.key" v-model="trainForm.predictors" />
                  <span><strong>{{ p.label }}</strong></span>
                </label>
              </div>
            </div>
          </template>

          <div class="actions">
            <button class="btn secondary" @click="goStep('enrich')">← Back</button>
            <button class="btn primary" :disabled="trainForm.predictors.length < 2" @click="goStep('train')">
              Configure training →
            </button>
          </div>
        </section>

        <!-- ═══════════════════════════════════════════════════════
             STEP 4 — TRAIN
        ════════════════════════════════════════════════════════ -->
        <section v-else-if="step === 'train'" class="panel">
          <h3>Train habitat model</h3>
          <p class="hint">Configure and start the MaxEnt training run.</p>

          <div class="row">
            <label for="train-title">Model name</label>
            <input id="train-title" v-model="trainForm.title" type="text" placeholder="Front Range fungi – autumn 2024" />
          </div>

          <div class="row">
            <label>Predictors</label>
            <div class="stages">
              <label v-for="p in predictorList" :key="p.key" class="stage">
                <input type="checkbox" :value="p.key" v-model="trainForm.predictors" />
                <span><strong>{{ p.label }}</strong></span>
              </label>
            </div>
          </div>

          <div class="row">
            <label for="bg-count">Background points</label>
            <input id="bg-count" v-model.number="trainForm.background" type="number" min="100" max="10000" step="100" />
            <p class="hint">Random absence proxies. 1 000–5 000 is typical.</p>
          </div>

          <div class="row">
            <label>Options</label>
            <label class="pick">
              <input type="checkbox" v-model="trainForm.autoOptimize" />
              <span>Auto-select best predictors (scout run)</span>
            </label>
            <label class="pick">
              <input type="checkbox" v-model="trainForm.preciseOnly" />
              <span>Precise GPS coordinates only</span>
            </label>
          </div>

          <div class="row">
            <label>Projection region (optional)</label>
            <div class="bbox">
              <label class="mini">N <input v-model.number="trainForm.regionNorth" type="number" step="0.1" /></label>
              <label class="mini">S <input v-model.number="trainForm.regionSouth" type="number" step="0.1" /></label>
              <label class="mini">W <input v-model.number="trainForm.regionWest" type="number" step="0.1" /></label>
              <label class="mini">E <input v-model.number="trainForm.regionEast" type="number" step="0.1" /></label>
            </div>
          </div>

          <p v-if="trainError" class="msg error">{{ trainError }}</p>

          <div class="actions">
            <button class="btn secondary" @click="goStep('explore')">← Back</button>
            <template v-if="!trainJobId">
              <button class="btn primary"
                      :disabled="trainForm.predictors.length < 2 || !trainForm.title || maxEnt.pending.value"
                      @click="submitTrain">
                {{ maxEnt.pending.value ? 'Queuing…' : 'Train model' }}
              </button>
            </template>
            <template v-else>
              <span class="job-status" :class="trainStatus">{{ trainStatusLabel }}</span>
              <button v-if="trainStatus === 'succeeded'" class="btn primary" @click="goStep('evaluate')">
                Evaluate model →
              </button>
            </template>
          </div>

          <!-- Training progress -->
          <div v-if="maxEnt.activeJob.value" class="job-progress">
            <div class="prog-bar"><div class="prog-fill indeterminate"></div></div>
            <p class="prog-msg">{{ maxEnt.activeJob.value.status === 'pending' ? 'Queued…' : 'Training in progress — this takes a few minutes.' }}</p>
          </div>
        </section>

        <!-- ═══════════════════════════════════════════════════════
             STEP 5 — EVALUATE (post-training gate)
        ════════════════════════════════════════════════════════ -->
        <section v-else-if="step === 'evaluate'" class="panel">
          <h3>Evaluate model</h3>
          <p class="hint">
            Review how well the model performed before rendering the full suitability map (an
            expensive operation). A good model has AUC &gt; 0.75.
          </p>

          <div v-if="!evalData && !evalLoading && !evalError" class="actions">
            <button class="btn primary" @click="loadEval">Load evaluation data</button>
          </div>

          <p v-if="evalLoading" class="msg">Running evaluation — this may take 30–60 seconds…</p>
          <p v-if="evalError" class="msg error">{{ evalError }}</p>

          <template v-if="evalData || trainedModel">
            <!-- AUC summary card -->
            <div class="eval-summary">
              <div class="eval-stat" :class="aucGrade">
                <span class="stat-value">{{ aucDisplay }}</span>
                <span class="stat-label">AUC</span>
              </div>
              <div class="eval-stat">
                <span class="stat-value">{{ gradeDisplay }}</span>
                <span class="stat-label">Grade</span>
              </div>
              <div v-if="evalData?.presenceCount" class="eval-stat">
                <span class="stat-value">{{ evalData.presenceCount }}</span>
                <span class="stat-label">Presences</span>
              </div>
            </div>

            <p class="hint" :class="aucGrade">
              <template v-if="aucGrade === 'excellent'">Excellent — this model distinguishes habitat from non-habitat very well.</template>
              <template v-else-if="aucGrade === 'good'">Good — model performance is solid.</template>
              <template v-else-if="aucGrade === 'fair'">Fair — the model may be usable but treat predictions cautiously.</template>
              <template v-else-if="aucGrade === 'weak'">Weak — consider adding more observations or adjusting predictors before rendering.</template>
            </p>

            <!-- Variable contributions -->
            <div v-if="contributions.length" class="explore-section">
              <h4>Predictor contributions</h4>
              <VariableContribution :data="contributions" title="" x-label="Contribution (%)" />
            </div>

            <!-- ROC curve -->
            <div v-if="evalData?.roc" class="explore-section">
              <h4>ROC curve</h4>
              <ROCCurve :points="evalData.roc" :auc="evalData.auc" />
            </div>
          </template>

          <div class="actions">
            <button class="btn secondary" @click="goStep('train')">← Back</button>
            <button class="btn primary" :disabled="!trainedModel" @click="goStep('view')">
              View on map →
            </button>
          </div>
        </section>

        <!-- ═══════════════════════════════════════════════════════
             STEP 6 — VIEW
        ════════════════════════════════════════════════════════ -->
        <section v-else-if="step === 'view'" class="panel">
          <h3>View suitability map</h3>
          <p class="hint">Your model is ready. Open the map to see the habitat suitability surface.</p>

          <div v-if="trainedModel" class="model-card">
            <strong>{{ trainedModel.title }}</strong>
            <p class="sub">AUC {{ aucDisplay }} · {{ gradeDisplay }}</p>
          </div>

          <div class="actions">
            <button class="btn secondary" @click="goStep('evaluate')">← Back</button>
            <NuxtLink to="/" class="btn primary">Open map →</NuxtLink>
            <NuxtLink to="/modeling/maxent" class="btn secondary">All models</NuxtLink>
          </div>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
import { useEeJobs } from '~/composables/useEeJobs'
import { useDatasets } from '~/composables/useDatasets'
import { useMaxEnt } from '~/composables/useMaxEnt'

// ── Steps ────────────────────────────────────────────────────────────────────
const STEPS = [
  { key: 'source',   label: 'Source' },
  { key: 'enrich',   label: 'Enrich' },
  { key: 'explore',  label: 'Explore' },
  { key: 'train',    label: 'Train' },
  { key: 'evaluate', label: 'Evaluate' },
  { key: 'view',     label: 'View' },
]
const STEP_ORDER = STEPS.map(s => s.key)

const route = useRoute()
const router = useRouter()
const step = computed({
  get: () => (STEP_ORDER.includes(route.query.step) ? route.query.step : 'source'),
  set: (v) => router.replace({ query: { ...route.query, step: v } }),
})

const completedSteps = ref(new Set())

function canReach(key) {
  const idx = STEP_ORDER.indexOf(key)
  if (idx === 0) return true
  // Can navigate to any completed step, or the first incomplete step
  const prev = STEP_ORDER[idx - 1]
  return completedSteps.value.has(prev)
}

function goStep(key) {
  if (!canReach(key)) return
  step.value = key
}

// ── Composables ───────────────────────────────────────────────────────────────
const membership = useMembership()
const jobsApi = useEeJobs()
const datasetsApi = useDatasets()
const maxEnt = useMaxEnt()
useFilters()

const rechecking = ref(false)
async function recheck() {
  rechecking.value = true
  try { await membership.refresh?.() } finally { rechecking.value = false }
}

onMounted(() => {
  datasetsApi.refresh()
  maxEnt.fetchModels()
})

// ── Predictor / stage metadata ────────────────────────────────────────────────
const stageList = [
  { key: 'terrain',       label: 'Terrain',                description: 'Elevation, slope, aspect and exposure indices.' },
  { key: 'landcover',     label: 'Land cover',             description: 'ESA WorldCover class at each point (10 m).' },
  { key: 'soil_moisture', label: 'Soil moisture',          description: 'ERA5-Land volumetric soil water on the day of the record.' },
  { key: 'precip',        label: 'Rainfall lead-up',       description: 'CHIRPS daily rainfall for the 7 days up to each record.' },
  { key: 'temperature',   label: 'Temperature lead-up',    description: 'ERA5-Land daily max/min for the 7 days up to each record.' },
  { key: 'ndvi',          label: 'Vegetation (NDVI)',      description: 'Sentinel-2 NDVI and moisture index, cloud-screened.' },
  { key: 'soil',          label: 'Soil properties',        description: 'USDA texture class, percent sand, depth to bedrock.' },
  { key: 'soil_taxonomy', label: 'Soil taxonomy',          description: 'USDA great group and order at each point.' },
  { key: 'fire',          label: 'Fire history',           description: 'Most recent burn year and years since fire (MODIS, 2001+).' },
  { key: 'forest',        label: 'Forest type & structure', description: 'GAP forest type, canopy cover and stand height (CONUS).' },
]

const predictorList = [
  { key: 'elevation',     label: 'Elevation' },
  { key: 'slope',         label: 'Slope' },
  { key: 'aspect',        label: 'Aspect' },
  { key: 'ndvi',          label: 'Vegetation (NDVI)' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'precip_normal', label: 'Precipitation' },
  { key: 'temp_normal',   label: 'Temperature' },
]

// ── Step 1: Source ────────────────────────────────────────────────────────────
const sourceForm = reactive({
  type: 'dataset',
  datasetSlug: '',
  north: 49.0, south: 24.5, east: -66.9, west: -124.8,
  dateFrom: '', dateTo: '', taxon: '',
})

const canAdvanceSource = computed(() => {
  if (sourceForm.type === 'dataset') return !!sourceForm.datasetSlug
  if (sourceForm.type === 'bbox') {
    return sourceForm.north != null && sourceForm.south != null
        && sourceForm.east != null && sourceForm.west != null
  }
  if (sourceForm.type === 'fetch') return !!sourceForm.datasetSlug // set after successful fetch
  return false
})


const FETCH_SOURCES = [
  { key: 'auto', label: 'Auto' },
  { key: 'inat', label: 'iNaturalist' },
  { key: 'gbif', label: 'GBIF' },
]

const fetchForm = reactive({
  taxon: '',
  source: 'auto',
  dateFrom: '',
  dateTo: '',
  loading: false,
  error: '',
  result: null,
})

async function runFetch() {
  fetchForm.error = ''
  fetchForm.result = null
  fetchForm.loading = true
  try {
    const { accessToken } = useAuth()
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}

    const p = new URLSearchParams({ species: fetchForm.taxon.trim() })
    if (fetchForm.dateFrom) p.set('d1', fetchForm.dateFrom)
    if (fetchForm.dateTo) p.set('d2', fetchForm.dateTo)

    const src = fetchForm.source === 'auto'
      ? (fetchForm.dateFrom || fetchForm.dateTo ? 'inat' : 'inat')
      : fetchForm.source
    const fn = src === 'gbif' ? 'gbif-fetch' : 'fetch-species'

    const res = await fetch(`/.netlify/functions/${fn}?${p.toString()}`, { headers })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(data.error || `Fetch failed (${res.status})`)
    if (!data.count) throw new Error('No records found for that taxon.')

    // Register the uploaded file in saved_datasets so the pipeline can use it.
    const storagePath = `species/${data.slug}.geojson`
    const title = `${fetchForm.taxon.trim()} (${src === 'gbif' ? 'GBIF' : 'iNat'}, ${data.count})`
    const saveRes = await fetch('/.netlify/functions/datasets', {
      method: 'POST',
      headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify({
        action: 'save_fetched',
        path: storagePath,
        slug: data.slug,
        title,
        feature_count: data.count,
      }),
    })
    const saved = await saveRes.json().catch(() => ({}))
    if (!saveRes.ok || !saved.ok) throw new Error(saved.error || 'Could not register the fetched dataset.')

    sourceForm.datasetSlug = saved.dataset.slug
    fetchForm.result = { count: data.count, title }
    await datasetsApi.refresh()
  } catch (e) {
    fetchForm.error = e?.message || 'Fetch failed.'
  } finally {
    fetchForm.loading = false
  }
}

function advanceSource() {
  if (!canAdvanceSource.value) return
  completedSteps.value = new Set([...completedSteps.value, 'source'])
  goStep('enrich')
}

// ── Step 2: Enrich ────────────────────────────────────────────────────────────
const enrichForm = reactive({
  title: '',
  stages: ['terrain', 'landcover', 'soil_moisture', 'precip', 'temperature', 'ndvi'],
})
const enrichError = ref('')
const enrichNote = ref('')
const enrichJobId = ref(null)
const enrichDatasetSlug = ref(null)

const enrichJob = computed(() => enrichJobId.value
  ? jobsApi.jobs.value.find(j => j.id === enrichJobId.value) : null)

const enrichJobStatus = computed(() => enrichJob.value?.status || (enrichJobId.value ? 'pending' : null))
const enrichStatusLabel = computed(() => {
  const s = enrichJobStatus.value
  if (s === 'succeeded') return 'Enrichment complete ✓'
  if (s === 'failed') return 'Enrichment failed'
  if (s === 'running') return `Running (${enrichJob.value?.progress || 0}%)…`
  return 'Queued…'
})

watch(enrichJobStatus, async (s) => {
  if (s === 'succeeded') {
    completedSteps.value = new Set([...completedSteps.value, 'enrich'])
    loadExploreData()
    // Auto-save the enriched output as a named dataset so training can reference it by slug.
    try {
      const saved = await datasetsApi.saveJob(
        { id: enrichJobId.value, title: enrichForm.title || 'Pipeline enrichment' },
        { title: enrichForm.title || 'Pipeline enrichment' },
      )
      enrichDatasetSlug.value = saved.slug
    } catch (_) {
      // Non-fatal — training will fall back to the original source.
    }
  }
})

function buildSource() {
  if (sourceForm.type === 'dataset') {
    return { type: 'dataset', slug: sourceForm.datasetSlug }
  }
  return {
    type: 'bbox',
    bounds: { north: sourceForm.north, south: sourceForm.south, east: sourceForm.east, west: sourceForm.west },
    dateFrom: sourceForm.dateFrom || undefined,
    dateTo: sourceForm.dateTo || undefined,
    taxon: sourceForm.taxon || undefined,
  }
}

async function submitEnrich() {
  enrichError.value = ''
  enrichNote.value = ''
  const spec = {
    kind: 'enrich',
    title: enrichForm.title || 'Pipeline enrichment',
    stages: enrichForm.stages,
    source: buildSource(),
  }
  try {
    const result = await jobsApi.submit(spec)
    enrichJobId.value = result.job?.id || result.id
    enrichNote.value = 'Job queued — enrichment will run on the server.'
    // Start polling for this job
    pollEnrichJob()
  } catch (e) {
    enrichError.value = e?.message || 'Submission failed.'
  }
}

let enrichPollTimer = null
function pollEnrichJob() {
  if (enrichPollTimer) clearInterval(enrichPollTimer)
  enrichPollTimer = setInterval(async () => {
    await jobsApi.refresh()
    const job = jobsApi.jobs.value.find(j => j.id === enrichJobId.value)
    if (job && (job.status === 'succeeded' || job.status === 'failed' || job.status === 'cancelled')) {
      clearInterval(enrichPollTimer)
      enrichPollTimer = null
    }
  }, 4000)
}
onUnmounted(() => { if (enrichPollTimer) clearInterval(enrichPollTimer) })

// ── Step 3: Explore ────────────────────────────────────────────────────────────
const exploreData = ref(null)

const PREDICTOR_META = {
  elevation:     { label: 'Elevation',       shortLabel: 'Elev' },
  slope:         { label: 'Slope',           shortLabel: 'Slope' },
  aspect:        { label: 'Aspect',          shortLabel: 'Asp' },
  ndvi:          { label: 'NDVI',            shortLabel: 'NDVI' },
  soil_moisture: { label: 'Soil moisture',   shortLabel: 'Soil' },
  precip_normal: { label: 'Precipitation',   shortLabel: 'Prec' },
  temp_normal:   { label: 'Temperature',     shortLabel: 'Temp' },
}

async function loadExploreData() {
  const slug = enrichDatasetSlug.value || (sourceForm.type === 'dataset' ? sourceForm.datasetSlug : null)
  if (!slug) return

  try {
    const { accessToken } = useAuth()
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}
    const res = await fetch(`/.netlify/functions/datasets?slug=${encodeURIComponent(slug)}`, { headers })
    if (!res.ok) return
    const data = await res.json()
    const features = data.geojson?.features || data.features || []
    if (!features.length) return
    exploreData.value = computeExploreStats(features)
    const recommended = exploreData.value.coverage
      .filter(v => v.pct >= 80)
      .map(v => v.key)
    trainForm.predictors = recommended.length >= 2 ? recommended : predictorList.map(p => p.key)
  } catch { /* non-fatal */ }
}

function computeExploreStats(features) {
  const total = features.length
  const keys = Object.keys(PREDICTOR_META)

  const coverage = keys.map(key => {
    const present = features.filter(f => f.properties?.[key] != null && !isNaN(f.properties[key])).length
    return {
      key,
      label: PREDICTOR_META[key].label,
      shortLabel: PREDICTOR_META[key].shortLabel,
      pct: total ? (present / total) * 100 : 0,
    }
  }).filter(v => v.pct > 0)

  const correlations = computeCorrelationMatrix(features, coverage.map(v => v.key))
  return { total, coverage, correlations }
}

function computeCorrelationMatrix(features, keys) {
  const cols = keys.map(k => features.map(f => f.properties?.[k] ?? null).filter(v => v != null))
  const n = keys.length
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0))

  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) { matrix[i][j] = 1; continue }
      const xi = cols[i], xj = cols[j]
      const len = Math.min(xi.length, xj.length)
      if (len < 3) { matrix[i][j] = matrix[j][i] = 0; continue }
      const mx = xi.slice(0, len).reduce((a, b) => a + b, 0) / len
      const my = xj.slice(0, len).reduce((a, b) => a + b, 0) / len
      let num = 0, dx = 0, dy = 0
      for (let k = 0; k < len; k++) {
        const a = xi[k] - mx, b = xj[k] - my
        num += a * b; dx += a * a; dy += b * b
      }
      const r = (dx && dy) ? num / Math.sqrt(dx * dy) : 0
      matrix[i][j] = matrix[j][i] = parseFloat(r.toFixed(3))
    }
  }
  return matrix
}

function corrColor(val, isDiag) {
  if (isDiag) return 'transparent'
  const a = Math.abs(val)
  if (a > 0.7) return `rgba(239,68,68,${0.3 + a * 0.4})`
  if (a > 0.4) return `rgba(251,191,36,${0.2 + a * 0.3})`
  return `rgba(34,197,94,${0.1 + a * 0.2})`
}

// ── Step 4: Train ─────────────────────────────────────────────────────────────
const trainForm = reactive({
  title: '',
  predictors: predictorList.map(p => p.key),
  background: 1000,
  autoOptimize: true,
  preciseOnly: false,
  regionNorth: null, regionSouth: null, regionEast: null, regionWest: null,
})
const trainError = ref('')
const trainJobId = ref(null)

const trainStatus = computed(() => {
  const job = maxEnt.activeJob.value
  if (!trainJobId.value) return null
  return job?.status || null
})
const trainStatusLabel = computed(() => {
  const s = trainStatus.value
  if (s === 'succeeded') return 'Training complete ✓'
  if (s === 'failed') return maxEnt.activeJob.value?.error_message || 'Training failed'
  if (s === 'running') return 'Training in progress…'
  return 'Queued…'
})

watch(trainStatus, (s) => {
  if (s === 'succeeded') {
    completedSteps.value = new Set([...completedSteps.value, 'train'])
  }
})

const trainedModel = computed(() => {
  if (!trainJobId.value) return null
  return maxEnt.models.value.find(m => m.id === maxEnt.activeJob.value?.model_configs?.id
    || maxEnt.models.value[0])
    || maxEnt.models.value[0] || null
})

async function submitTrain() {
  trainError.value = ''
  maxEnt.error.value = ''

  const region = (trainForm.regionNorth != null && trainForm.regionSouth != null)
    ? { north: trainForm.regionNorth, south: trainForm.regionSouth,
        east: trainForm.regionEast,  west: trainForm.regionWest }
    : undefined

  const spec = {
    title: trainForm.title,
    predictors: trainForm.predictors,
    background: trainForm.background,
    autoOptimize: trainForm.autoOptimize,
    preciseOnly: trainForm.preciseOnly,
    region,
    source_dataset_id: enrichDatasetSlug.value
      || (sourceForm.type === 'dataset' ? sourceForm.datasetSlug : undefined),
    source: buildSource(),
  }

  const result = await maxEnt.trainModel(spec)
  if (!result.ok) {
    trainError.value = result.error || 'Training submission failed.'
    return
  }
  trainJobId.value = maxEnt.activeJob.value?.job_id
}

// ── Step 5: Evaluate ──────────────────────────────────────────────────────────
const evalData = ref(null)
const evalLoading = ref(false)
const evalError = ref('')

async function loadEval() {
  const jobId = trainJobId.value || maxEnt.activeJob.value?.job_id
  if (!jobId) return
  evalLoading.value = true
  evalError.value = ''
  try {
    const { accessToken } = useAuth()
    const token = await accessToken()
    const headers = token ? { authorization: `Bearer ${token}` } : {}
    const res = await fetch(`/.netlify/functions/modeling-maxent?evaluate=1&jobId=${encodeURIComponent(jobId)}`, { headers })
    const data = await res.json()
    if (!data.ok) throw new Error(data.error || 'Evaluation failed.')
    evalData.value = data
    completedSteps.value = new Set([...completedSteps.value, 'evaluate'])
  } catch (e) {
    evalError.value = e.message
  } finally {
    evalLoading.value = false
  }
}

const aucValue = computed(() => {
  if (evalData.value?.auc != null) return evalData.value.auc
  if (trainedModel.value?.results?.[0]?.auc != null) return trainedModel.value.results[0].auc
  return null
})
const aucDisplay = computed(() => aucValue.value != null ? aucValue.value.toFixed(3) : '—')
const gradeDisplay = computed(() => {
  const g = evalData.value?.grade || trainedModel.value?.results?.[0]?.grade
  return g ? g.charAt(0).toUpperCase() + g.slice(1) : '—'
})
const aucGrade = computed(() => {
  const g = evalData.value?.grade || trainedModel.value?.results?.[0]?.grade
  return g || (aucValue.value >= 0.9 ? 'excellent' : aucValue.value >= 0.75 ? 'good' : aucValue.value >= 0.6 ? 'fair' : 'weak')
})

const contributions = computed(() => {
  const c = evalData.value?.contributions || trainedModel.value?.results?.[0]?.contributions
  if (!c) return []
  return Object.entries(c).map(([key, value]) => ({
    label: PREDICTOR_META[key]?.label || key,
    value: typeof value === 'number' ? value : parseFloat(value),
  })).sort((a, b) => b.value - a.value)
})
</script>

<style scoped>
.pipeline-page { padding: 16px 18px; max-width: 960px; margin: 0 auto; }

.page-head { margin-bottom: 20px; }
.page-head h2 { margin: 0 0 4px; font-size: 1.3rem; }
.sub { margin: 0; color: var(--muted); font-size: 0.84rem; }

/* ── Step indicator ── */
.steps {
  display: flex; gap: 0; margin-bottom: 24px;
  border: 1px solid var(--border); border-radius: 10px; overflow: hidden;
}
.step-btn {
  flex: 1; display: flex; flex-direction: column; align-items: center; gap: 3px;
  padding: 10px 6px; border: 0; border-right: 1px solid var(--border);
  background: var(--surface); cursor: pointer; font-size: 0.78rem; color: var(--muted);
  transition: background 0.15s, color 0.15s;
}
.step-btn:last-child { border-right: 0; }
.step-btn:disabled { cursor: default; opacity: 0.5; }
.step-btn.reachable:hover { background: var(--surface-2); color: var(--text); }
.step-btn.done { color: var(--text); background: var(--surface-2); }
.step-btn.active { background: var(--accent); color: var(--accent-ink); }
.step-btn.active .step-num { background: rgba(0,0,0,0.15); }
.step-num {
  width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
  background: var(--border); font-size: 0.75rem; font-weight: 700;
}
.step-btn.done .step-num { background: #22c55e; color: #fff; }
.step-label { font-weight: 600; }

@media (max-width: 600px) { .step-label { display: none; } }

/* ── Panel ── */
.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 20px 22px; margin-bottom: 20px; }
.panel h3 { margin: 0 0 6px; font-size: 1.05rem; }

/* ── Form rows ── */
.row { display: grid; grid-template-columns: 160px 1fr; gap: 8px 14px; align-items: start; margin-bottom: 14px; }
.row label:first-child { padding-top: 6px; font-size: 0.84rem; font-weight: 600; color: var(--muted); }
.row input[type="text"], .row input[type="number"], .row input[type="date"], .row select {
  border: 1px solid var(--border); border-radius: 7px; padding: 7px 10px;
  font-size: 0.88rem; background: var(--input-bg); color: var(--text); width: 100%;
}
.row.two { grid-template-columns: 160px 1fr 1fr; }
.stack { display: flex; flex-direction: column; gap: 4px; font-size: 0.84rem; }
.stack span { font-weight: 600; color: var(--muted); }
.stack input, .stack select { border: 1px solid var(--border); border-radius: 7px; padding: 6px 9px; font-size: 0.88rem; background: var(--input-bg); color: var(--text); }
@media (max-width: 600px) { .row { grid-template-columns: 1fr; } .row.two { grid-template-columns: 1fr 1fr; } }

/* ── Source / stage pickers ── */
.sources, .stages { display: flex; flex-direction: column; gap: 6px; }
.pick { display: flex; align-items: flex-start; gap: 8px; font-size: 0.88rem; cursor: pointer; padding: 6px 10px; border: 1px solid var(--border); border-radius: 7px; background: var(--input-bg); }
.pick:hover { background: var(--surface-2); }
.pick input { margin-top: 2px; }
.stage { padding: 7px 10px; }
.stage span { display: flex; flex-direction: column; }
.stage em { font-size: 0.78rem; color: var(--muted); font-style: normal; }
.disabled { opacity: 0.5; pointer-events: none; }

/* ── BBox ── */
.bbox { display: flex; gap: 8px; flex-wrap: wrap; }
.mini { display: flex; align-items: center; gap: 4px; font-size: 0.82rem; color: var(--muted); }
.mini input { width: 76px; border: 1px solid var(--border); border-radius: 6px; padding: 5px 7px; font-size: 0.84rem; background: var(--input-bg); color: var(--text); }

/* ── Actions ── */
.actions { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
.btn { border: 1px solid var(--border); background: var(--surface-2); border-radius: 7px; padding: 8px 16px; font-size: 0.88rem; font-weight: 600; cursor: pointer; text-decoration: none; color: var(--text); }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.btn.secondary { background: var(--surface); }
.btn:hover:not(:disabled) { opacity: 0.85; }
.btn:disabled { opacity: 0.45; cursor: default; }

/* ── Job status ── */
.job-status { font-size: 0.85rem; font-weight: 600; padding: 5px 10px; border-radius: 6px; background: var(--surface-2); }
.job-status.succeeded { color: #16a34a; background: #dcfce7; }
.job-status.failed { color: #dc2626; background: #fee2e2; }
.job-status.running, .job-status.pending { color: #d97706; background: #fef3c7; }

.job-progress { margin-top: 14px; }
.prog-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 6px; }
.prog-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.4s; }
.prog-fill.indeterminate { width: 40%; animation: slide 1.6s infinite; }
@keyframes slide { 0% { transform: translateX(-100%); } 100% { transform: translateX(350%); } }
.prog-msg { font-size: 0.82rem; color: var(--muted); margin: 0; }

/* ── Explore / analysis ── */
.explore-section { margin-bottom: 22px; }
.explore-section h4 { margin: 0 0 8px; font-size: 0.92rem; }
.cov-table, .corr-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
.cov-table th, .corr-table th { text-align: left; padding: 6px 8px; background: var(--surface-2); border-bottom: 1px solid var(--border); color: var(--muted); font-weight: 600; }
.cov-table td, .corr-table td { padding: 6px 8px; border-bottom: 1px solid var(--border-soft); }
.cov-table tr.low td { color: var(--muted); }
.cov-bar-wrap { display: flex; align-items: center; gap: 8px; }
.cov-bar { display: inline-block; height: 10px; border-radius: 5px; min-width: 2px; }
.cov-pct { font-size: 0.8rem; color: var(--muted); }
.badge { font-size: 0.75rem; font-weight: 600; padding: 2px 7px; border-radius: 4px; }
.badge.ok   { background: #dcfce7; color: #16a34a; }
.badge.warn { background: #fef3c7; color: #d97706; }
.badge.bad  { background: #fee2e2; color: #dc2626; }
.corr-cell { text-align: center; font-size: 0.8rem; min-width: 42px; }
.corr-cell.high { font-weight: 700; }
.row-label { font-weight: 600; font-size: 0.8rem; color: var(--muted); }

/* ── Eval summary ── */
.eval-summary { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; }
.eval-stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 9px; padding: 14px 20px; text-align: center; min-width: 90px; }
.stat-value { display: block; font-size: 1.5rem; font-weight: 700; }
.stat-label { display: block; font-size: 0.76rem; color: var(--muted); margin-top: 2px; }
.eval-stat.excellent .stat-value { color: #16a34a; }
.eval-stat.good      .stat-value { color: #2563eb; }
.eval-stat.fair      .stat-value { color: #d97706; }
.eval-stat.weak      .stat-value { color: #dc2626; }

/* ── Hint / messages ── */
.hint { font-size: 0.82rem; color: var(--muted); margin: 2px 0 10px; }
.hint.warn { color: #d97706; }
.hint.excellent { color: #16a34a; }
.hint.good      { color: #2563eb; }
.hint.fair      { color: #d97706; }
.hint.weak      { color: #dc2626; }
.msg { padding: 10px 14px; border-radius: 7px; font-size: 0.88rem; }
.msg.error { background: #fee2e2; color: #dc2626; }
.msg.ok    { background: #dcfce7; color: #16a34a; }
.linkish { background: none; border: none; color: var(--accent); cursor: pointer; font-size: inherit; padding: 0; text-decoration: underline; }
.gate { padding: 32px; text-align: center; }

/* ── Model card ── */
.model-card { background: var(--surface-2); border: 1px solid var(--border); border-radius: 9px; padding: 14px 18px; margin-bottom: 16px; }
.model-card strong { font-size: 1rem; }
.model-card .sub { margin: 4px 0 0; font-size: 0.82rem; color: var(--muted); }
</style>
