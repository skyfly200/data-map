<template>
  <div class="modeling">
    <MaxEntTutorial ref="tutorialRef" />

    <div class="head">
      <div class="head-text">
        <h2>Habitat Suitability Models</h2>
        <p class="sub">
          Trained <GlossaryTooltip term="habitat suitability" :definition="g('habitat suitability')">habitat suitability</GlossaryTooltip>
          models. Use the <NuxtLink to="/pipeline">Pipeline</NuxtLink> to train a new model.
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
            <p class="empty-sub">Use the <NuxtLink to="/pipeline">Pipeline</NuxtLink> to train a model, or link a pre-computed EE asset below.</p>
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

        </section>

        <!-- ─── Link pre-computed asset ──────────────────────────────────────── -->
        <section class="panel add-panel">
          <div class="panel-head">
            <h3>Link a pre-computed model</h3>
          </div>
          <div class="tab-body">
            <p class="tab-desc">
              Add a suitability surface already computed in Earth Engine by asset path, e.g.
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
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, ref, onMounted } from 'vue'
import { useGlossary } from '~/composables/useGlossary'
import GlossaryTooltip from '~/components/GlossaryTooltip.vue'
import MaxEntTutorial from '~/components/MaxEntTutorial.vue'
import { VISIBILITY_LABELS } from '~/composables/useDatasets'
import ModelComparison from '~/components/ModelComparison.vue'

const { define: g } = useGlossary()
const tutorialRef = ref(null)


const { models, activeJob, pending, error, fetchModels, deleteModel, registerAsset } = useMaxEnt()
const membership = useMembership()
const modelOverlay = useModelOverlay()
const router = useRouter()
const loading = ref(false)
const showComparison = ref(false)
const selectedModels = computed(() => models.value.filter(m => m.selected))

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

const visibilities = ['private', 'members', 'public']

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
  await fetchModels()
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
.workflow-hint {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  margin-bottom: 18px; font-size: 0.8rem;
}
.workflow-hint .step { color: var(--muted); text-decoration: none; }
.workflow-hint .step:hover { color: var(--text); text-decoration: underline; }
.workflow-hint .step.active { color: var(--accent, #34c46a); font-weight: 600; }
.workflow-hint .step-arrow { color: var(--border); }
.head-text .sub a { color: var(--accent, #34c46a); text-decoration: none; }
.head-text .sub a:hover { text-decoration: underline; }

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
