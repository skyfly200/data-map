<template>
  <div class="jobs">
    <div class="head">
      <div>
        <h2>Pipeline jobs</h2>
        <p class="sub">
          Enrich a set of observations with terrain, weather and vegetation sampled from
          Earth Engine. A job runs on the server; you can leave this page.
        </p>
      </div>
      <!-- Inside ClientOnly with everything else that depends on the session:
           the server has no token to read a tier from, so rendering this during
           SSR guarantees a hydration mismatch. -->
      <ClientOnly>
        <span v-if="membership.configured && membership.isMember.value"
              class="tier-badge" :class="membership.tier.value">
          {{ membership.label.value }}
        </span>
      </ClientOnly>
    </div>

    <ClientOnly>
      <!-- Signed out, or dues not paid: say which, and what to do about it.
           With no Supabase at all there are no accounts to be a member of and
           the app is open, so the form is shown rather than a gate nobody can
           get past. -->
      <div v-if="membership.configured && !membership.isAuthed.value" class="gate">
        <p>Pipeline jobs are a membership benefit.</p>
        <NuxtLink to="/login" class="btn primary">Sign in</NuxtLink>
      </div>

      <div v-else-if="membership.configured && !membership.isMember.value" class="gate">
        <p v-if="lapsed">
          Your membership lapsed on {{ fmtDate(membership.lapsesAt.value) }}. Renew it to run jobs.
        </p>
        <p v-else>Running pipeline jobs is a benefit of Front Range Mycological Society membership.</p>
        <!-- A tier change only reaches the browser at the next token refresh,
             so someone who has just paid would otherwise see this and think it
             had not worked. -->
        <button class="btn" :disabled="rechecking" @click="recheck">
          {{ rechecking ? 'Checking…' : 'Just joined? Check again' }}
        </button>
      </div>

      <template v-else>
        <!-- ── Submit ─────────────────────────────────────────────────── -->
        <section class="panel">
          <h3>New job</h3>

          <div class="row">
            <label for="job-title">Name</label>
            <input id="job-title" v-model="form.title" type="text" placeholder="Autumn foray, Front Range" />
          </div>

          <!-- Running a job over a previous job's output is what makes these
               compose: enrich a set, save it, enrich the result differently. -->
          <div class="row">
            <label>Run over</label>
            <div class="sources">
              <label class="pick">
                <input v-model="form.sourceType" type="radio" value="bbox" />
                <span>Observations in an area</span>
              </label>
              <label class="pick" :class="{ disabled: !sourceChoices.length }">
                <input v-model="form.sourceType" type="radio" value="dataset"
                       :disabled="!sourceChoices.length" />
                <span>A saved dataset</span>
              </label>
            </div>
          </div>

          <template v-if="form.sourceType === 'dataset'">
            <div class="row">
              <label for="job-dataset">Dataset</label>
              <select id="job-dataset" v-model="form.datasetSlug">
                <option value="" disabled>Choose one…</option>
                <option v-for="d in sourceChoices" :key="d.id" :value="d.slug">
                  {{ d.title }}{{ d.feature_count ? ` · ${d.feature_count.toLocaleString()} points` : '' }}
                </option>
              </select>
            </div>
            <p class="hint">
              The dataset's own points and dates are used as they are. Area,
              dates and taxon do not apply — those chose the points in the first
              place.
            </p>
          </template>

          <template v-else>
            <div class="row">
              <label>Area</label>
              <div class="bbox">
                <label class="mini">N <input v-model.number="form.north" type="number" step="0.1" /></label>
                <label class="mini">S <input v-model.number="form.south" type="number" step="0.1" /></label>
                <label class="mini">W <input v-model.number="form.west" type="number" step="0.1" /></label>
                <label class="mini">E <input v-model.number="form.east" type="number" step="0.1" /></label>
              </div>
            </div>
            <p class="hint">
              <button class="linkish" @click="useMapView">Use the current map view</button>
            </p>

            <div class="row two">
              <label class="stack">
                <span>From</span>
                <input v-model="form.dateFrom" type="date" />
              </label>
              <label class="stack">
                <span>To</span>
                <input v-model="form.dateTo" type="date" />
              </label>
            </div>

            <div class="row">
              <label for="job-taxon">Taxon</label>
              <input id="job-taxon" v-model="form.taxon" type="text"
                     placeholder="Any — or a genus, family, order…" />
            </div>
          </template>

          <div class="row">
            <label>Layers</label>
            <div class="stages">
              <label v-for="s in stageList" :key="s.key" class="stage">
                <input type="checkbox" :value="s.key" v-model="form.stages" />
                <span>
                  <strong>{{ s.label }}</strong>
                  <em>{{ s.description }}</em>
                </span>
              </label>
            </div>
          </div>

          <p v-if="submitError" class="msg error">{{ submitError }}</p>
          <p v-if="submitNote" class="msg ok">{{ submitNote }}</p>

          <div class="actions">
            <button class="btn primary" :disabled="jobsApi.submitting.value || !canSubmit"
                    @click="onSubmit">
              {{ jobsApi.submitting.value ? 'Submitting…' : 'Queue job' }}
            </button>
            <span v-if="!form.stages.length" class="hint">Pick at least one layer.</span>
            <span v-else-if="form.sourceType === 'dataset' && !form.datasetSlug" class="hint">
              Choose a dataset.
            </span>
          </div>
        </section>

        <!-- ── The queue ──────────────────────────────────────────────── -->
        <section class="panel">
          <div class="panel-head">
            <h3>Your jobs</h3>
            <button class="linkish" :disabled="jobsApi.loading.value" @click="jobsApi.refresh()">Refresh</button>
          </div>

          <p v-if="jobsApi.error.value" class="msg error">{{ jobsApi.error.value }}</p>
          <p v-else-if="!jobsApi.jobs.value.length" class="msg">Nothing submitted yet.</p>

          <ul v-else class="job-list">
            <li v-for="job in jobsApi.jobs.value" :key="job.id" class="job" :class="job.status">
              <div class="job-top">
                <span class="status" :class="job.status">{{ statusLabel(job) }}</span>
                <strong class="job-name">{{ job.title || describe(job) }}</strong>
                <span class="when">{{ fmtWhen(job.created_at) }}</span>
              </div>

              <p class="job-meta">
                {{ (job.params?.points || 0).toLocaleString() }} points ·
                {{ (job.params?.stages || []).length }} layers ·
                {{ job.cost_units || job.estimated_units || 0 }} units
              </p>

              <!-- Progress is written by the worker as each stage advances, so
                   a ten-minute job visibly moves rather than looking stuck. -->
              <div v-if="running(job)" class="bar" :aria-label="`${Math.round(job.progress * 100)}%`">
                <div class="fill" :style="{ width: `${Math.max(2, job.progress * 100)}%` }"></div>
                <span class="bar-text">{{ job.message || 'Working…' }} {{ Math.round(job.progress * 100) }}%</span>
              </div>

              <p v-if="job.status === 'failed'" class="msg error">{{ job.error }}</p>

              <p v-if="job.status === 'succeeded' && skippedNote(job)" class="msg warn">
                {{ skippedNote(job) }}
              </p>

              <div class="job-actions">
                <button v-if="job.status === 'succeeded'" class="btn small primary"
                        :disabled="opening === job.id" @click="openOnMap(job)">
                  {{ opening === job.id ? 'Loading…' : 'Open on map' }}
                </button>
                <button v-if="job.status === 'succeeded' && !savedFrom(job)" class="btn small"
                        @click="startSave(job)">
                  Save as dataset
                </button>
                <span v-else-if="job.status === 'succeeded'" class="saved-as">
                  Saved as <strong>{{ savedFrom(job).title }}</strong>
                </span>
                <button v-if="running(job)" class="btn small" @click="jobsApi.cancel(job.id)">Cancel</button>
              </div>

              <!-- Naming it is the whole step, so it is one field and a button
                   rather than a dialog. Visibility defaults to private and is
                   only offered as a change from that. -->
              <div v-if="saving.job === job.id" class="save-form">
                <label class="stack">
                  <span>Name this dataset</span>
                  <input v-model="saving.title" type="text" placeholder="Autumn 2026, enriched"
                         @keyup.enter="confirmSave" />
                </label>
                <label class="stack">
                  <span>Who can use it</span>
                  <select v-model="saving.visibility">
                    <option v-for="v in datasetsApi.visibilities" :key="v" :value="v">
                      {{ VISIBILITY_LABELS[v] }}
                    </option>
                  </select>
                </label>
                <div class="actions">
                  <button class="btn small primary" :disabled="saving.busy || !saving.title.trim()"
                          @click="confirmSave">
                    {{ saving.busy ? 'Saving…' : 'Save' }}
                  </button>
                  <button class="btn small" @click="saving.job = ''">Cancel</button>
                </div>
                <p v-if="saving.error" class="msg error">{{ saving.error }}</p>
              </div>
            </li>
          </ul>
        </section>

        <!-- ── Saved datasets ─────────────────────────────────────────── -->
        <section class="panel">
          <div class="panel-head">
            <h3>Your datasets</h3>
            <button class="linkish" :disabled="datasetsApi.loading.value"
                    @click="datasetsApi.refresh()">Refresh</button>
          </div>

          <p v-if="datasetsApi.error.value" class="msg error">{{ datasetsApi.error.value }}</p>
          <p v-else-if="!datasetsApi.datasets.value.length" class="msg">
            Nothing saved yet. A finished job can be saved as a dataset, and then
            used as the starting point for another one.
          </p>

          <ul v-else class="ds-list">
            <li v-for="d in datasetsApi.datasets.value" :key="d.id" class="ds">
              <div class="ds-top">
                <strong>{{ d.title }}</strong>
                <span class="vis-badge" :class="d.visibility">{{ VISIBILITY_LABELS[d.visibility] }}</span>
                <span class="when">{{ fmtWhen(d.created_at) }}</span>
              </div>
              <p class="ds-meta">
                <code>{{ d.slug }}</code>
                <template v-if="d.feature_count"> · {{ d.feature_count.toLocaleString() }} points</template>
              </p>
              <div class="job-actions">
                <select :value="d.visibility" @change="setVisibility(d, $event.target.value)">
                  <option v-for="v in datasetsApi.visibilities" :key="v" :value="v">
                    {{ VISIBILITY_LABELS[v] }}
                  </option>
                </select>
                <button class="btn small" @click="useAsSource(d)">Run a job on it</button>
                <button class="btn small danger" @click="removeDataset(d)">Delete</button>
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
import { STAGES, DEFAULT_STAGES } from '~/netlify/lib/ee-pipeline.mjs'
import { VISIBILITY_LABELS } from '~/composables/useDatasets'

const membership = useMembership()
const jobsApi = useEeJobs()
const datasetsApi = useDatasets()
const { addInlineDataset } = useObservations()
const router = useRouter()

const stageList = Object.entries(STAGES).map(([key, s]) => ({ key, ...s }))

const form = reactive({
  title: '',
  sourceType: 'bbox',
  datasetSlug: '',
  north: 40.5, south: 39.2, west: -106.2, east: -104.8,
  dateFrom: '', dateTo: '', taxon: '',
  stages: [...DEFAULT_STAGES],
})

/** Datasets this member may run a job over: their own, plus anything shared. */
const sourceChoices = computed(() => datasetsApi.available.value)

const canSubmit = computed(() => {
  if (!form.stages.length) return false
  if (form.sourceType === 'dataset') return Boolean(form.datasetSlug)
  return true
})

const submitError = ref('')
const submitNote = ref('')
const opening = ref('')
const rechecking = ref(false)

const lapsed = computed(() => {
  const at = membership.lapsesAt.value
  return at && at < new Date()
})

const running = (job) => job.status === 'queued' || job.status === 'running'

function statusLabel(job) {
  if (job.status === 'queued') return 'Queued'
  if (job.status === 'running') return 'Running'
  if (job.status === 'succeeded') return 'Done'
  if (job.status === 'cancelled') return 'Cancelled'
  return 'Failed'
}

function describe(job) {
  const p = job.params || {}
  const where = p.source?.type === 'dataset' ? p.source.slug : 'area'
  return `${p.stages?.length || 0} layers over ${where}`
}

/** How much of a finished job came back thin, if any did. */
function skippedNote(job) {
  const skipped = job.result_meta?.skipped || {}
  const entries = Object.entries(skipped)
  if (!entries.length) return ''
  const which = entries.map(([k, n]) => `${STAGES[k]?.label || k} (${n} points)`).join(', ')
  return `Some samples were skipped after repeated Earth Engine failures: ${which}. `
    + 'Running the job again fills only the gaps.'
}

const fmtDate = (d) => (d ? d.toLocaleDateString() : '')
const fmtWhen = (iso) => {
  const d = new Date(iso)
  return Number.isFinite(d.getTime()) ? d.toLocaleString() : ''
}

/** Copy the map's current viewport into the area fields. */
function useMapView() {
  try {
    const saved = JSON.parse(localStorage.getItem('map-last-view') || 'null')
    if (!saved?.bounds) { submitError.value = 'Open the map first, then come back.'; return }
    Object.assign(form, {
      north: Number(saved.bounds.north.toFixed(3)),
      south: Number(saved.bounds.south.toFixed(3)),
      east: Number(saved.bounds.east.toFixed(3)),
      west: Number(saved.bounds.west.toFixed(3)),
    })
    submitError.value = ''
  } catch {
    submitError.value = 'Could not read the map view.'
  }
}

async function recheck() {
  rechecking.value = true
  try { await membership.refreshSession() } finally { rechecking.value = false }
}

async function onSubmit() {
  submitError.value = ''
  submitNote.value = ''
  try {
    // A dataset already is a set of points with dates; the area, date and
    // taxon fields chose points in the first place and mean nothing applied to
    // one. Sending them anyway would look like a filter that did nothing.
    const source = form.sourceType === 'dataset'
      ? { type: 'dataset', slug: form.datasetSlug }
      : {
        type: 'bbox',
        bounds: { north: form.north, south: form.south, east: form.east, west: form.west },
        dateFrom: form.dateFrom || null,
        dateTo: form.dateTo || null,
        taxon: form.taxon || '',
      }

    const result = await jobsApi.submit({
      kind: 'enrich',
      title: form.title,
      stages: form.stages,
      source,
    })
    submitNote.value = `Queued: ${result.points.toLocaleString()} points, about `
      + `${result.estimate} units. ${result.remaining} left this month.`
  } catch (e) {
    submitError.value = e.message
  }
}

/**
 * Load a finished result and show it.
 *
 * It becomes a dataset like any other, so the map, the charts and the phenology
 * panel all read it without knowing it came from a job.
 */
async function openOnMap(job) {
  opening.value = job.id
  submitError.value = ''
  try {
    const geojson = await jobsApi.fetchResult(job)
    const count = geojson?.features?.length || 0
    addInlineDataset({
      id: `job-${job.id}`,
      label: `${job.title || 'Pipeline job'} (${count})`,
      path: `mem:job-${job.id}`,
    }, geojson)
    router.push('/map')
  } catch (e) {
    submitError.value = e.message
  } finally {
    opening.value = ''
  }
}

// ── Saving a result as a dataset ─────────────────────────────────────────────

const saving = reactive({ job: '', title: '', visibility: 'private', busy: false, error: '' })

/** The dataset a job was already saved as, if it was. */
const savedFrom = (job) => datasetsApi.datasets.value.find((d) => d.job_id === job.id) || null

function startSave(job) {
  saving.job = job.id
  saving.title = job.title || describe(job)
  // Private unless they change it here. The server defaults the same way, so
  // this field is showing the default rather than deciding it.
  saving.visibility = 'private'
  saving.error = ''
}

async function confirmSave() {
  const job = jobsApi.jobs.value.find((j) => j.id === saving.job)
  if (!job || !saving.title.trim()) return
  saving.busy = true
  saving.error = ''
  try {
    await datasetsApi.saveJob(job, { title: saving.title.trim(), visibility: saving.visibility })
    await datasetsApi.refreshAvailable()
    saving.job = ''
  } catch (e) {
    saving.error = e.message
  } finally {
    saving.busy = false
  }
}

async function setVisibility(dataset, visibility) {
  if (visibility === dataset.visibility) return
  try {
    await datasetsApi.update(dataset.id, { visibility })
    await datasetsApi.refreshAvailable()
  } catch (e) {
    datasetsApi.error.value = e.message
  }
}

async function removeDataset(dataset) {
  if (!confirm(`Delete “${dataset.title}”? The job result itself is kept.`)) return
  try {
    await datasetsApi.remove(dataset.id)
    await datasetsApi.refreshAvailable()
    if (form.datasetSlug === dataset.slug) form.datasetSlug = ''
  } catch (e) {
    datasetsApi.error.value = e.message
  }
}

/** Point the form at this dataset and scroll back up to it. */
function useAsSource(dataset) {
  form.sourceType = 'dataset'
  form.datasetSlug = dataset.slug
  if (import.meta.client) window.scrollTo({ top: 0, behavior: 'smooth' })
}

onMounted(() => {
  membership.loadProfile()
  jobsApi.refresh()
  datasetsApi.refresh()
  datasetsApi.refreshAvailable()
})
</script>

<style scoped>
.jobs { padding: 16px 18px; max-width: 860px; margin: 0 auto; }
.head { display: flex; align-items: flex-end; justify-content: space-between; gap: 16px; margin-bottom: 16px; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 0; color: var(--muted); font-size: 0.82rem; max-width: 60ch; }

.tier-badge { font-size: 0.72rem; padding: 3px 9px; border-radius: 999px; white-space: nowrap;
  border: 1px solid var(--border); color: var(--muted); }
.tier-badge.member { border-color: #3d8b5f; color: #3d8b5f; }
.tier-badge.perpetual { border-color: #3d7b8b; color: #3d7b8b; }
.tier-badge.admin { border-color: #8b5f3d; color: #8b5f3d; }

.gate { border: 1px solid var(--border); border-radius: 10px; padding: 20px; background: var(--surface);
  text-align: center; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.gate p { margin: 0; color: var(--muted); font-size: 0.88rem; }

.panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
  background: var(--surface); margin-bottom: 18px; }
.panel h3 { margin: 0 0 12px; font-size: 0.95rem; }
.panel-head { display: flex; align-items: baseline; justify-content: space-between; }
.panel-head h3 { margin-bottom: 12px; }

.row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.row > label:first-child { flex: 0 0 70px; font-size: 0.82rem; color: var(--muted); }
.row input[type="text"] { flex: 1; }
.row.two { gap: 16px; }
.stack { display: flex; flex-direction: column; gap: 3px; font-size: 0.82rem; color: var(--muted); }

input[type="text"], input[type="number"], input[type="date"] {
  background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 8px; font: inherit; font-size: 0.84rem;
}
.bbox { display: flex; gap: 8px; flex-wrap: wrap; }
.mini { display: flex; align-items: center; gap: 4px; font-size: 0.78rem; color: var(--muted); }
.mini input { width: 82px; }

.stages { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; flex: 1; }
.stage { display: flex; gap: 7px; align-items: flex-start; font-size: 0.8rem; }
.stage strong { display: block; font-weight: 600; }
.stage em { display: block; font-style: normal; color: var(--muted); font-size: 0.75rem; }

.actions { display: flex; align-items: center; gap: 12px; margin-top: 12px; }
.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.82rem; cursor: pointer; }
.btn:hover:not(:disabled) { border-color: var(--muted); }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
.btn.small { padding: 4px 9px; font-size: 0.78rem; }
.linkish { background: none; border: none; color: var(--accent, #3d8b5f); font: inherit;
  font-size: 0.8rem; cursor: pointer; padding: 0; text-decoration: underline; }

.hint { margin: 0; font-size: 0.78rem; color: var(--muted); }
.msg { margin: 8px 0 0; font-size: 0.82rem; color: var(--muted); }
.msg.error { color: #b3492f; }
.msg.ok { color: #3d8b5f; }
.msg.warn { color: #8b6b3d; }

.job-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.job { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.job-top { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.job-name { font-size: 0.88rem; }
.when { margin-left: auto; color: var(--muted); font-size: 0.74rem; }
.job-meta { margin: 4px 0 0; font-size: 0.76rem; color: var(--muted); }

.status { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em;
  padding: 2px 7px; border-radius: 999px; border: 1px solid var(--border); color: var(--muted); }
.status.running, .status.queued { border-color: #3d6b8b; color: #3d6b8b; }
.status.succeeded { border-color: #3d8b5f; color: #3d8b5f; }
.status.failed { border-color: #b3492f; color: #b3492f; }

.bar { position: relative; height: 20px; margin-top: 8px; border-radius: 5px;
  background: var(--bg); border: 1px solid var(--border); overflow: hidden; }
.fill { height: 100%; background: var(--accent, #3d8b5f); opacity: 0.28; transition: width 0.4s ease; }
.bar-text { position: absolute; inset: 0; display: flex; align-items: center;
  padding: 0 8px; font-size: 0.72rem; color: var(--text); }

.job-actions { display: flex; gap: 8px; margin-top: 8px; align-items: center; flex-wrap: wrap; }
.job-actions:empty { display: none; }
.btn.danger { color: #b3492f; }
.btn.danger:hover:not(:disabled) { border-color: #b3492f; }
.saved-as { font-size: 0.78rem; color: var(--muted); }

/* Choosing what a job runs over. */
.sources { display: flex; gap: 14px; flex-wrap: wrap; flex: 1; }
.pick { display: flex; align-items: center; gap: 6px; font-size: 0.82rem; }
.pick.disabled { opacity: 0.5; }

.save-form { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border);
  display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.save-form .actions { grid-column: 1 / -1; margin-top: 0; }
.save-form .msg { grid-column: 1 / -1; }

.ds-list { list-style: none; margin: 10px 0 0; padding: 0; display: grid; gap: 10px; }
.ds { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.ds-top { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.ds-meta { margin: 3px 0 0; font-size: 0.76rem; color: var(--muted); }
.ds-meta code { font-size: 0.74rem; }

.vis-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); }
/* Private is the default and the quiet one; sharing is what stands out, since
   that is the state worth noticing at a glance. */
.vis-badge.members { border-color: #3d8b5f; color: #3d8b5f; }
.vis-badge.public { border-color: #8b5f3d; color: #8b5f3d; }

@media (prefers-reduced-motion: reduce) { .fill { transition: none; } }
</style>
