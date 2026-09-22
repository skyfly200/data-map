<template>
  <div class="modeling">
    <div class="head">
      <div class="title-row">
        <h2>MaxEnt Modeling</h2>
        <button v-if="selectedModels.length > 0" class="btn small primary" @click="showComparison = true">
          Compare Selected ({{ selectedModels.length }})
        </button>
      </div>
      <p class="sub">
        Predict habitat suitability by learning from environmental conditions at sighting locations.
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
        <div v-if="showComparison" class="overlay">
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
            <label>Predictors</label>
            <div class="stages">
              <label v-for="p in predictorList" :key="p.key" class="stage">
                <input type="checkbox" :value="p.key" v-model="form.predictors" />
                <span><strong>{{ p.label }}</strong></span>
              </label>
            </div>
          </div>

          <div class="row two">
            <label class="stack">
              <span>Background Points</span>
              <input v-model.number="form.backgroundCount" type="number" step="100" />
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
          <p v-else-if="!models.length" class="msg">No saved models yet.</p>

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
                  · <span class="cv-score">{{ m.results[0]?.auc }} AUC</span>
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
import { PREDICTOR_KEYS, MAXENT_PREDICTORS, MIN_PREDICTORS, DEFAULT_PREDICTORS } from '~/netlify/lib/maxent.mjs'
import { VISIBILITY_LABELS } from '~/composables/useDatasets'
import ModelComparison from '~/components/ModelComparison.vue'

const { models, activeJob, pending, error, fetchModels, trainModel, deleteModel } = useMaxEnt()
const { availableDatasets } = useDatasets()
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
  if (confirm(`Are you sure you want to delete “${model.title}”?`)) {
    const res = await deleteModel(model.id)
    if (res.ok) await fetchModels()
  }
}

function openModelOnMap(model) {
  modelOverlay.setModel(model)
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

onMounted(fetchModels)
</script>

<style scoped>
.modeling { padding: 16px 18px; max-width: 1200px; margin: 0 auto; }
.head { margin-bottom: 24px; }
.head .title-row { display: flex; justify-content: space-between; align-items: center; }
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
  display: flex; align-items: center; gap: 8px; padding: 6px 10px;
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; cursor: pointer;
}
.stage:hover { background: var(--surface-3); }
.stage input { margin: 0; }

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
</style>
