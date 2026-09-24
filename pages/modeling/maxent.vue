<template>
  <div class="modeling">
    <MaxEntTutorial ref="tutorialRef" />

    <div class="head">
      <div class="title-row">
        <h2>MaxEnt Modeling</h2>
        <div class="title-actions">
          <button class="btn small ghost" aria-label="Open tutorial" @click="tutorialRef?.start()">
            ? Tutorial
          </button>
          <button v-if="selectedModels.length > 0" class="btn small primary" @click="showComparison = true">
            Compare Selected ({{ selectedModels.length }})
          </button>
        </div>
      </div>
      <p class="sub">
        Predict <GlossaryTooltip term="habitat suitability" :definition="g('habitat suitability')">habitat suitability</GlossaryTooltip>
        by learning from environmental conditions at sighting locations.
        A model is not a survey: it says where the environment resembles where the species was found.
      </p>
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

        <!-- ─── Submit Form ─────────────────────────────────────────────────── -->
        <section class="panel">
          <h3 class="ct">Configure New Model</h3>
          
          <div class="row">
            <label for="model-title">Model Name</label>
            <input id="model-title" v-model="form.title" type="text" placeholder="Example: Red-belted Conk Suitability" />
          </div>

          <div class="row">
            <label for="model-desc">Description (optional)</label>
            <textarea id="model-desc" v-model="form.description" placeholder="Note about the specific environment or year..."></textarea>
          </div>

          <div class="row">
            <label>Source Dataset</label>
            <select v-model="form.sourceDatasetId">
              <option value="" disabled>Choose a dataset…</option>
              <option v-for="d in availableDatasets" :key="d.id" :value="d.id">
                {{ d.title }}{{ d.feature_count ? ` · ${d.feature_count.toLocaleString()} points` : '' }}
              </option>
            </select>
          </div>

          <div class="row">
            <label>
              <GlossaryTooltip term="predictors" :definition="g('predictors')">Predictors</GlossaryTooltip>
              <span class="field-hint">Select at least {{ MIN_PREDICTORS }}</span>
            </label>
            <div class="stages">
              <label v-for="p in predictorList" :key="p.key" class="stage" :title="PREDICTOR_DESCRIPTIONS[p.key]">
                <input type="checkbox" :value="p.key" v-model="form.predictors" />
                <span>
                  <strong>{{ p.label }}</strong>
                  <em class="pred-desc">{{ PREDICTOR_DESCRIPTIONS[p.key] }}</em>
                </span>
              </label>
            </div>
          </div>

          <div class="row two">
            <label class="stack">
              <span>
                <GlossaryTooltip term="background points" :definition="g('background points')">Background Points</GlossaryTooltip>
                <span class="field-hint">{{ MIN_BACKGROUND }}–{{ MAX_BACKGROUND }}</span>
              </span>
              <input v-model.number="form.backgroundCount" type="number" :min="MIN_BACKGROUND" :max="MAX_BACKGROUND" step="100" />
              <span class="field-note">Random locations sampled to contrast against presences. More = slower but more stable; 1,000 is a reasonable start.</span>
            </label>
            <label class="stack">
              <span>Visibility</span>
              <select v-model="form.visibility">
                <option v-for="v in visibilities" :key="v" :value="v">
                  {{ VISIBILITY_LABELS[v] }}
                </option>
              </select>
            </label>
          </div>

          <div class="actions">
            <button class="btn primary" :disabled="pending || !canSubmit" @click="onSubmit">
              {{ pending ? 'Submitting…' : 'Queue Model' }}
            </button>
            <span v-if="form.predictors.length < MIN_PREDICTORS" class="hint">
              Pick at least {{ MIN_PREDICTORS }} predictors.
            </span>
            <span v-else-if="!form.sourceDatasetId" class="hint">
              Please select a source dataset.
            </span>
          </div>
          
          <p v-if="error" class="msg error">{{ error }}</p>
          <p v-if="submitNote" class="msg ok">{{ submitNote }}</p>
        </section>

        <!-- ─── Active Job ──────────────────────────────────────────────────── -->
        <section v-if="activeJob" class="panel active-job">
          <div class="panel-head">
            <h3>Training in Progress</h3>
            <button class="linkish" @click="activeJob = null">Dismiss</button>
          </div>

          <div class="job-status">
            <div class="info">
              <strong>{{ activeJob.model_configs.title }}</strong>
              <span class="status" :class="activeJob.status">{{ activeJob.status }}</span>
            </div>
            <div class="bar">
              <div class="fill" :style="{ width: progress + '%' }"></div>
              <span class="bar-text">{{ progress }}%</span>
            </div>
          </div>
          <p v-if="activeJob.error_message" class="msg error">{{ activeJob.error_message }}</p>
        </section>

        <!-- ─── Your Models ──────────────────────────────────────────────────── -->
        <section class="panel">
          <div class="panel-head">
            <h3>Your Saved Models</h3>
            <button class="linkish" :disabled="loading" @click="fetchModels">Refresh</button>
          </div>

          <p v-if="loading" class="msg">Loading…</p>
          <div v-else-if="!models.length" class="empty-state">
            <p class="empty-title">No models yet — here's how it works</p>
            <ol class="onboarding-steps">
              <li>
                <strong>Pick a dataset</strong>
                <span>Go to the <NuxtLink to="/data">Data page</NuxtLink> and select the species and filters you want to model. The form above will let you choose which dataset to train on.</span>
              </li>
              <li>
                <strong>Choose predictors</strong>
                <span>Tick the environmental layers that are ecologically meaningful for your species. Terrain and vegetation are a good baseline; add climate normals if your species is temperature- or rainfall-sensitive.</span>
              </li>
              <li>
                <strong>Queue the model</strong>
                <span>Hit <em>Queue Model</em> above. Training runs on Google Earth Engine and usually takes a few minutes. When it finishes you'll see your model here with an AUC score and a "View on Map" button.</span>
              </li>
            </ol>
          </div>

          <ul v-else class="model-list">
            <li v-for="m in models" :key="m.id" class="model-item" :class="{ selected: m.selected }">
              <div class="model-top">
                <label class="model-select">
                  <input type="checkbox" v-model="m.selected" />
                  <strong class="model-name">{{ m.title }}</strong>
                </label>
                <span class="vis-badge" :class="m.visibility">{{ VISIBILITY_LABELS[m.visibility] }}</span>
                <span class="when">{{ fmtWhen(m.created_at) }}</span>
              </div>
              <p class="model-meta">
                {{ m.predictors.length }} predictors · {{ m.background_count }} background points
                <template v-if="m.results">
                  · <span class="cv-score">{{ m.results[0]?.auc }} <GlossaryTooltip term="AUC" :definition="g('AUC')">AUC</GlossaryTooltip></span>
                </template>
              </p>
              <div class="model-actions">
                <button class="btn small primary" @click="openModelOnMap(m)">
                  View on Map
                </button>
                <button class="btn small danger" @click="confirmDelete(m)">Delete</button>
              </div>
            </li>
          </ul>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
import { computed, reactive, ref, onMounted } from 'vue'
import { PREDICTOR_KEYS, MAXENT_PREDICTORS, MIN_PREDICTORS, MAX_BACKGROUND, MIN_BACKGROUND, DEFAULT_PREDICTORS } from '~/netlify/lib/maxent.mjs'
import { useGlossary } from '~/composables/useGlossary'
import GlossaryTooltip from '~/components/GlossaryTooltip.vue'
import MaxEntTutorial from '~/components/MaxEntTutorial.vue'

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
import { VISIBILITY_LABELS } from '~/composables/useDatasets'
import ModelComparison from '~/components/ModelComparison.vue'

const { models, activeJob, pending, error, fetchModels, trainModel, deleteModel } = useMaxEnt()
const { available: availableDatasets, refreshAvailable } = useDatasets()
const membership = useMembership()
const modelOverlay = useModelOverlay()
const router = useRouter()
const loading = ref(false)
const submitNote = ref('')

const visibilities = ['private', 'members', 'public']

const form = reactive({
  title: '',
  description: '',
  sourceDatasetId: '',
  predictors: [...DEFAULT_PREDICTORS],
  backgroundCount: 1000,
  visibility: 'private',
})

const predictorList = PREDICTOR_KEYS.map((key) => ({ key, label: MAXENT_PREDICTORS[key].label }))

const canSubmit = computed(() => {
  return form.title.trim() && form.sourceDatasetId && form.predictors.length >= MIN_PREDICTORS
})

async function onSubmit() {
  submitNote.value = ''
  const spec = {
    title: form.title,
    description: form.description,
    source_dataset_id: form.sourceDatasetId,
    predictors: form.predictors,
    background: form.backgroundCount,
    visibility: form.visibility,
  }

  const result = await trainModel(spec)
  if (result.ok) {
    submitNote.value = 'Model submitted successfully! It will appear in your list when finished.'
    form.title = ''
    form.description = ''
  }
}

async function confirmDelete(model) {
  if (confirm(`Are you sure you want to delete "${model.title}"?`)) {
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
  return 50 // Simplified for prototype
})

function fmtWhen(dateStr) {
  if (!dateStr) return ''
  const d = new Date(dateStr)
  return d.toLocaleDateString()
}

// --- Selection Logic ---
const showComparison = ref(false)
const selectedModels = computed(() => models.value.filter(m => m.selected))

onMounted(() => { fetchModels(); refreshAvailable() })
</script>

<style scoped>
.modeling { padding: 16px 18px; max-width: 1200px; margin: 0 auto; }
.head { margin-bottom: 24px; }
.head .title-row { display: flex; justify-content: space-between; align-items: center; }
.title-actions { display: flex; align-items: center; gap: 8px; }
.head h2 { margin: 0; }
.head .sub { color: var(--muted); font-size: 0.86rem; line-height: 1.4; }

.panel {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 20px; margin-bottom: 20px;
}
.panel-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
.panel-head h3 { margin: 0; font-size: 1rem; }

.ct { text-align: center; margin-bottom: 16px; }

.row { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
.row.two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.row label { font-size: 0.82rem; font-weight: 600; color: var(--muted); }

input, select, textarea {
  padding: 8px 12px; border: 1px solid var(--border); background: var(--surface-2);
  color: var(--text); border-radius: 6px; font-size: 0.86rem;
}
textarea { min-height: 60px; resize: vertical; }

.stages { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 8px; margin-top: 4px; }
.stage {
  display: flex; align-items: flex-start; gap: 8px; padding: 6px 10px;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
}
.stage:hover { background: var(--surface-3); }
.stage input { margin: 2px 0 0; flex: 0 0 auto; }
.stage span { display: flex; flex-direction: column; gap: 2px; }
.pred-desc { font-size: 0.74rem; color: var(--muted); font-style: normal; line-height: 1.3; }

.field-hint { font-weight: 400; color: var(--muted); font-size: 0.76rem; margin-left: 4px; }
.field-note { font-size: 0.76rem; color: var(--muted); line-height: 1.4; margin-top: 4px; }

.empty-state {
  padding: 20px; background: var(--surface-2); border-radius: 8px;
  border: 1px dashed var(--border);
}
.empty-title { font-weight: 600; margin: 0 0 14px; font-size: 0.9rem; color: var(--text-strong); }
.onboarding-steps {
  margin: 0; padding: 0 0 0 18px; display: flex; flex-direction: column; gap: 12px;
}
.onboarding-steps li { font-size: 0.84rem; color: var(--muted); line-height: 1.5; }
.onboarding-steps li strong { color: var(--text); display: block; margin-bottom: 2px; }
.onboarding-steps a { color: var(--accent); text-decoration: none; }
.onboarding-steps a:hover { text-decoration: underline; }

.actions { display: flex; align-items: center; gap: 12px; margin-top: 20px; }
.hint { font-size: 0.76rem; color: var(--muted); }

.msg { padding: 12px; border-radius: 6px; font-size: 0.82rem; margin-top: 12px; }
.msg.error { background: var(--danger-soft); color: var(--danger); }
.msg.ok { background: var(--accent-soft); color: var(--accent); }

.active-job { border: 2px solid var(--accent); background: var(--accent-soft-2); }
.job-status { display: flex; flex-direction: column; gap: 12px; margin-top: 12px; }
.job-status .info { display: flex; justify-content: space-between; align-items: center; }
.job-status .status { font-size: 0.76rem; font-weight: 600; text-transform: uppercase; padding: 2px 6px; border-radius: 4px; background: var(--border); }
.job-status .status.running { background: var(--accent); color: var(--accent-ink); }
.job-status .status.succeeded { background: var(--success); color: var(--success-ink); }
.job-status .status.failed { background: var(--danger); color: var(--danger-ink); }

.bar { height: 8px; background: var(--border); border-radius: 4px; position: relative; overflow: hidden; }
.bar .fill { height: 100%; background: var(--accent); transition: width 0.3s ease; }
.bar .bar-text {
  position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
  font-size: 0.7rem; font-weight: 600; color: var(--text);
}

.model-list { list-style: none; padding: 0; margin: 0; }
.model-item {
  display: flex; flex-direction: column; gap: 8px; padding: 16px;
  border-bottom: 1px solid var(--border);
}
.model-item:last-child { border-bottom: none; }
.model-item.selected { background: var(--accent-soft-2); border-left: 4px solid var(--accent); }
.model-top { display: flex; justify-content: space-between; align-items: center; }
.model-select { display: flex; align-items: center; gap: 8px; cursor: pointer; }
.model-select input { margin: 0; }
.model-name { font-weight: 600; }
.model-meta { font-size: 0.82rem; color: var(--muted); }
.vis-badge {
  font-size: 0.7rem; padding: 2px 6px; border-radius: 4px; background: var(--border);
  text-transform: uppercase; font-weight: 600;
}
.model-actions { display: flex; gap: 8px; margin-top: 8px; }

.gate {
  padding: 40px; text-align: center; background: var(--surface);
  border: 1px solid var(--border); border-radius: 12px;
}

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
  width: 24px; height: 24px; cursor: pointer; font-size: 12px;
}

/* ─── Responsive: tablets ─────────────────────────────────────────────────
   The modeling interface is form-heavy; on a tablet the two-up rows and the
   badge/date header run out of room, so they stack rather than crush. */
@media (max-width: 820px) {
  .modeling { padding: 14px 14px; }
  .head .title-row { flex-wrap: wrap; gap: 10px; }
  .panel { padding: 16px; }
  .row.two { grid-template-columns: 1fr; gap: 12px; }
  .model-top { flex-wrap: wrap; gap: 6px; }
  .model-top .when { width: 100%; order: 3; }
  .overlay { padding: 12px; }
  .overlay-content { max-height: 94vh; padding: 16px; }
}

/* ─── Responsive: phones ──────────────────────────────────────────────────
   One predictor per line, and actions go full-width so they are easy to tap. */
@media (max-width: 520px) {
  .stages { grid-template-columns: 1fr; }
  .actions { flex-direction: column; align-items: stretch; }
  .actions .btn { width: 100%; }
  .actions .hint { text-align: center; }
  .model-actions { flex-wrap: wrap; }
  .model-actions .btn { flex: 1 1 auto; }
}

/* Touch devices: roomier tap targets for inputs, checkboxes and buttons so the
   modeling form is usable with a finger rather than a mouse pointer. */
@media (pointer: coarse) {
  input, select, textarea { padding: 11px 13px; font-size: 16px; }
  .btn { min-height: 44px; }
  .stage { padding: 10px 12px; }
  .stage input[type="checkbox"],
  .model-select input[type="checkbox"] { width: 20px; height: 20px; }
  .close-btn { width: 34px; height: 34px; font-size: 15px; }
}
</style>
