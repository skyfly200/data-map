<template>
  <div class="admin">
    <div class="head">
      <div>
        <h2>Administration</h2>
        <p class="sub">Membership tiers, Earth Engine quotas and FRMS saved datasets.</p>
      </div>
    </div>

    <ClientOnly>
      <div v-if="!membership.isAdmin.value" class="gate">
        <p>This screen is for FRMS administrators.</p>
        <NuxtLink v-if="!membership.isAuthed.value" to="/login" class="btn primary">Sign in</NuxtLink>
      </div>

      <template v-else>
        <div class="tabs" role="tablist">
          <button :class="{ on: tab === 'members' }" role="tab" @click="show('members')">Members</button>
          <button :class="{ on: tab === 'datasets' }" role="tab" @click="show('datasets')">Datasets</button>
          <button :class="{ on: tab === 'layers' }" role="tab" @click="show('layers')">Map layers</button>
        </div>

        <p v-if="error" class="msg error">{{ error }}</p>
        <p v-if="note" class="msg ok">{{ note }}</p>
        <p v-if="loading" class="msg">Loading…</p>

        <!-- ── Members ────────────────────────────────────────────────── -->
        <section v-if="tab === 'members' && !loading" class="panel">
          <p v-if="!members.length" class="msg">No accounts yet.</p>

          <template v-else>
            <!-- The question an admin opens this screen with is whether the
                 Earth Engine budget is being spent and by whom, and a list of
                 per-member rows does not answer it. -->
            <div class="totals">
              <div class="total">
                <span class="n">{{ totals.unitsThisMonth.toLocaleString() }}</span>
                <span class="l">units this month</span>
                <div class="bar" :class="barClass(totals.share)">
                  <span class="fill" :style="{ width: barWidth(totals.share) }"></span>
                </div>
                <span class="sub">of {{ totals.quotaTotal.toLocaleString() }} allocated</span>
              </div>
              <div class="total">
                <span class="n">{{ totals.active }}</span>
                <span class="l">active members</span>
                <span class="sub">{{ totals.members }} accounts in total</span>
              </div>
              <div class="total">
                <span class="n">{{ totals.jobsToday }}</span>
                <span class="l">jobs today</span>
                <span class="sub">{{ totals.running }} running now</span>
              </div>
              <div class="total">
                <span class="n" :class="{ warn: totals.overQuota || totals.nearQuota }">
                  {{ totals.overQuota + totals.nearQuota }}
                </span>
                <span class="l">at or near quota</span>
                <span class="sub">{{ totals.idle }} used none of theirs</span>
              </div>
            </div>

            <div class="sorter">
              <label for="member-sort">Sort by</label>
              <select id="member-sort" v-model="sortBy">
                <option value="usage">Quota used</option>
                <option value="tier">Tier</option>
                <option value="name">Name</option>
                <option value="recent">Last active</option>
              </select>
            </div>

          <ul class="rows">
            <li v-for="m in sortedMembers" :key="m.user_id" class="row-item">
              <div class="row-top">
                <strong>{{ m.display_name || m.user_id.slice(0, 8) }}</strong>
                <span class="tier-badge" :class="m.tier">{{ m.tier }}</span>
                <span v-if="lapsedFor(m)" class="tier-badge lapsed">lapsed</span>
                <button class="linkish" @click="edit(m)">
                  {{ editing === m.user_id ? 'Close' : 'Edit' }}
                </button>
              </div>

              <!-- A limit with no usage beside it is a number nobody can judge,
                   and a number with no bar is one nobody reads at a glance. -->
              <div class="meter">
                <div class="bar" :class="barClass(shareFor(m))">
                  <span class="fill" :style="{ width: barWidth(shareFor(m)) }"></span>
                </div>
                <span class="meter-text">
                  <strong>{{ m.usage.unitsThisMonth.toLocaleString() }}</strong>
                  / {{ m.ee_quota_monthly.toLocaleString() }} units
                  <span class="pct" :class="barClass(shareFor(m))">{{ pctFor(m) }}</span>
                </span>
                <span class="meter-more">
                  {{ m.usage.jobsToday }}/{{ m.ee_jobs_per_day }} jobs today
                  <template v-if="m.usage.running"> · <strong>{{ m.usage.running }} running</strong></template>
                  <template v-if="m.usage.failed"> · {{ m.usage.failed }} failed</template>
                  · {{ lastActive(m) }}
                </span>
              </div>

              <div v-if="editing === m.user_id" class="editor">
                <div class="grid">
                  <label class="stack">
                    <span>Tier</span>
                    <select v-model="draft.tier">
                      <option v-for="t in TIERS" :key="t" :value="t">{{ t }}</option>
                    </select>
                  </label>
                  <label class="stack">
                    <span>Member until</span>
                    <input v-model="draft.member_until" type="date" />
                    <!-- Left editable rather than disabled: the date is still
                         a record of dues paid, it just stops governing access.
                         Saying so beats an admin wondering why it had no
                         effect. -->
                    <small v-if="neverExpires(draft.tier)" class="field-note">
                      Not used: this tier does not expire. The date is kept as a
                      record of dues paid.
                    </small>
                  </label>
                  <label v-for="f in LIMIT_FIELDS" :key="f" class="stack">
                    <span>{{ limitLabel(f) }}</span>
                    <input v-model.number="draft[f]" type="number" min="0" />
                  </label>
                </div>
                <label class="stack wide">
                  <span>Notes</span>
                  <textarea v-model="draft.notes" rows="2"></textarea>
                </label>
                <div class="actions">
                  <button class="btn primary" :disabled="saving" @click="saveMember(m)">
                    {{ saving ? 'Saving…' : 'Save' }}
                  </button>
                  <button class="btn" @click="editing = ''">Cancel</button>
                </div>
              </div>
            </li>
          </ul>
          </template>
        </section>

        <!-- ── Custom Earth Engine layers ─────────────────────────────── -->
        <section v-if="tab === 'layers' && !loading" class="panel">
          <p class="hint lead">
            Compute a layer in the Earth Engine Code Editor, export it with
            <code>Export.image.toAsset()</code>, then register the asset here. The app
            renders it like any built-in layer. An asset ID is stored, never a
            script: what you register names something already computed under your
            own Earth Engine project.
          </p>

          <!-- The rendering is the part that is easy to get wrong and hard to
               notice: a percent layer with no mask paints the whole world the
               bottom of its ramp, and ten classes across a five-colour ramp
               read as a quantity that is not there. Neither is an error — the
               layer draws, badly, and the asset gets blamed. -->
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

          <p v-if="!layers.length" class="msg">No custom layers yet.</p>
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

        <!-- ── Datasets ───────────────────────────────────────────────── -->
        <section v-if="tab === 'datasets' && !loading" class="panel">
          <p v-if="!datasets.length" class="msg">No saved datasets yet.</p>

          <ul v-else class="rows">
            <li v-for="d in datasets" :key="d.id" class="row-item">
              <div class="row-top">
                <strong>{{ d.title }}</strong>
                <span class="tier-badge" :class="d.visibility">{{ d.visibility }}</span>
                <span class="usage">
                  {{ (d.feature_count || 0).toLocaleString() }} features
                  · {{ fmtWhen(d.created_at) }}
                </span>
              </div>
              <div class="actions">
                <label class="stack">
                  <span>Visibility</span>
                  <select :value="d.visibility" @change="setVisibility(d, $event.target.value)">
                    <option value="private">private</option>
                    <option value="members">members</option>
                    <option value="public">public</option>
                  </select>
                </label>
                <button class="btn danger" @click="removeDataset(d)">Remove</button>
              </div>
            </li>
          </ul>
          <p class="hint">
            Removing a dataset takes it out of the list. The stored file is left in place,
            so a shared dataset cannot be lost with one click.
          </p>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup>
import { computed, ref, reactive, onMounted } from 'vue'
import { TIERS, neverExpires } from '~/netlify/lib/tiers.mjs'
import { DEFAULT_LIMITS, quotaFraction, rollUpUsage } from '~/netlify/lib/quotas.mjs'
import { LAYER_PRESETS, applyPreset } from '~/netlify/lib/ee-custom-layers.mjs'

const membership = useMembership()
const { accessToken } = useAuth()

const LIMIT_FIELDS = Object.keys(DEFAULT_LIMITS)

const tab = ref('members')
const sortBy = ref('usage')

// ── Reading the quota ────────────────────────────────────────────────────────

const totals = computed(() => rollUpUsage(members.value))

const shareFor = (m) => quotaFraction(m.usage?.unitsThisMonth || 0, m.ee_quota_monthly || 0)
const pctFor = (m) => `${Math.round(shareFor(m) * 100)}%`

// The bar is capped at full width; the percentage beside it is not, so someone
// at 240% of their quota reads as over rather than as exactly full.
const barWidth = (share) => `${Math.min(100, Math.max(0, share * 100))}%`
const barClass = (share) => (share >= 1 ? 'over' : share >= 0.8 ? 'near' : '')

function lastActive(m) {
  const at = m.usage?.lastJobAt
  if (!at) return 'never run a job'
  const days = Math.floor((Date.now() - new Date(at).getTime()) / 86400000)
  if (days <= 0) return 'active today'
  if (days === 1) return 'active yesterday'
  if (days < 30) return `active ${days} days ago`
  return `last active ${new Date(at).toLocaleDateString(undefined, { month: 'short', year: 'numeric' })}`
}

// Heaviest users first by default: that is the list an admin is scanning for.
const sortedMembers = computed(() => {
  const list = [...members.value]
  const byName = (a, b) => (a.display_name || a.user_id).localeCompare(b.display_name || b.user_id)
  if (sortBy.value === 'name') return list.sort(byName)
  if (sortBy.value === 'tier') {
    const rank = { admin: 0, perpetual: 1, member: 2, free: 3 }
    return list.sort((a, b) => (rank[a.tier] ?? 4) - (rank[b.tier] ?? 4) || byName(a, b))
  }
  if (sortBy.value === 'recent') {
    return list.sort((a, b) =>
      new Date(b.usage?.lastJobAt || 0) - new Date(a.usage?.lastJobAt || 0) || byName(a, b))
  }
  return list.sort((a, b) => shareFor(b) - shareFor(a) || byName(a, b))
})
const members = ref([])
const datasets = ref([])
const layers = ref([])

/** A blank registration form. Defaults that render something rather than nothing. */
const BLANK_LAYER = {
  id: null, name: '', slug: '', group: 'Custom', tier: 'member',
  asset_id: '', asset_type: 'image', band: '', reducer: 'mosaic',
  date_from: '', date_to: '',
  vis_min: '', vis_max: '', mask_below: '', opacity: 0.8,
  palette: '#f7fcb9, #addd8e, #31a354', attribution: '', note: '',
}
const draftLayer = reactive({ ...BLANK_LAYER })

// Which preset the rendering came from. Only a starting point — everything it
// fills stays editable, and changing a field afterwards does not un-choose it.
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
    // Stored as an array and edited as text; the server takes either.
    palette: (l.palette || []).join(', '),
    // <input type="date"> wants exactly YYYY-MM-DD and shows nothing for an
    // ISO timestamp.
    date_from: l.date_from ? String(l.date_from).slice(0, 10) : '',
    date_to: l.date_to ? String(l.date_to).slice(0, 10) : '',
    vis_min: l.vis_min ?? '', vis_max: l.vis_max ?? '', mask_below: l.mask_below ?? '',
  })
}

async function saveLayer() {
  saving.value = true
  error.value = ''
  note.value = ''
  try {
    await call('/.netlify/functions/admin-members', {
      method: 'POST',
      body: JSON.stringify({ action: 'save-layer', ...draftLayer }),
    })
    note.value = draftLayer.id ? 'Layer saved.' : 'Layer added. It appears in the map’s layer picker.'
    resetLayer()
    await show('layers')
  } catch (e) {
    error.value = e.message
  } finally {
    saving.value = false
  }
}

async function removeLayer(l) {
  if (!confirm(`Remove the layer “${l.name}”?`)) return
  error.value = ''
  try {
    await call('/.netlify/functions/admin-members', {
      method: 'POST',
      body: JSON.stringify({ action: 'delete-layer', id: l.id }),
    })
    await show('layers')
  } catch (e) {
    error.value = e.message
  }
}
const loading = ref(false)
const saving = ref(false)
const error = ref('')
const note = ref('')
const editing = ref('')
const draft = reactive({})

const LABELS = {
  ee_quota_monthly: 'Units / month',
  ee_jobs_per_day: 'Jobs / day',
  ee_max_points: 'Max points / job',
  ee_max_concurrent: 'Concurrent jobs',
}
const limitLabel = (f) => LABELS[f] || f

const fmtWhen = (iso) => {
  const d = new Date(iso)
  return Number.isFinite(d.getTime()) ? d.toLocaleDateString() : ''
}

// A date in the past on a tier that ignores dates is not a lapse. Badging one
// would send an admin chasing a renewal from somebody who does not owe one.
const lapsedFor = (m) => !neverExpires(m.tier)
  && m.member_until && new Date(m.member_until) < new Date()

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
  const data = await res.json().catch(() => ({}))
  if (!res.ok || !data.ok) throw new Error(messageFrom(data, res.status))
  return data
}

async function show(which) {
  tab.value = which
  error.value = ''
  note.value = ''
  loading.value = true
  try {
    const data = await call(`/.netlify/functions/admin-members?what=${which}`)
    if (which === 'members') members.value = data.members || []
    else if (which === 'layers') layers.value = data.layers || []
    else datasets.value = data.datasets || []
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

function edit(m) {
  if (editing.value === m.user_id) { editing.value = ''; return }
  editing.value = m.user_id
  Object.keys(draft).forEach((k) => delete draft[k])
  Object.assign(draft, {
    tier: m.tier,
    // <input type="date"> wants exactly YYYY-MM-DD and silently shows nothing
    // for anything else, including the ISO timestamp the column holds.
    member_until: m.member_until ? String(m.member_until).slice(0, 10) : '',
    notes: m.notes || '',
    ...Object.fromEntries(LIMIT_FIELDS.map((f) => [f, m[f]])),
  })
}

async function saveMember(m) {
  saving.value = true
  error.value = ''
  note.value = ''
  try {
    const data = await call('/.netlify/functions/admin-members', {
      method: 'POST',
      body: JSON.stringify({
        action: 'set-member',
        user_id: m.user_id,
        ...draft,
        member_until: draft.member_until || null,
      }),
    })
    note.value = data.note || 'Saved.'
    editing.value = ''
    await show('members')
  } catch (e) {
    error.value = e.message
  } finally {
    saving.value = false
  }
}

async function setVisibility(d, visibility) {
  error.value = ''
  try {
    await call('/.netlify/functions/admin-members', {
      method: 'POST',
      body: JSON.stringify({ action: 'set-dataset', id: d.id, visibility }),
    })
    await show('datasets')
  } catch (e) {
    error.value = e.message
  }
}

async function removeDataset(d) {
  if (!confirm(`Remove “${d.title}” from the dataset list?`)) return
  error.value = ''
  try {
    await call('/.netlify/functions/admin-members', {
      method: 'POST',
      body: JSON.stringify({ action: 'delete-dataset', id: d.id }),
    })
    await show('datasets')
  } catch (e) {
    error.value = e.message
  }
}

onMounted(async () => {
  await membership.refresh()
  if (membership.isAdmin.value) show('members')
})
</script>

<style scoped>
.admin { padding: 16px 18px; max-width: 900px; margin: 0 auto; }
.head h2 { margin: 0; font-size: 1.1rem; }
.sub { margin: 2px 0 16px; color: var(--muted); font-size: 0.82rem; }

.gate { border: 1px solid var(--border); border-radius: 10px; padding: 20px; background: var(--surface);
  text-align: center; display: flex; flex-direction: column; align-items: center; gap: 10px; }
.gate p { margin: 0; color: var(--muted); font-size: 0.88rem; }

.tabs { display: flex; gap: 6px; margin-bottom: 14px; }
.tabs button { background: none; border: 1px solid var(--border); border-radius: 6px;
  padding: 5px 12px; font: inherit; font-size: 0.82rem; color: var(--muted); cursor: pointer; }
.tabs button.on { color: var(--text); border-color: var(--muted); }

.panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--surface); }
.rows { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 10px; }
.row-item { border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
.row-top { display: flex; align-items: baseline; gap: 9px; flex-wrap: wrap; }
.usage { margin-left: auto; color: var(--muted); font-size: 0.75rem; }

/* ── Quota at a glance ──────────────────────────────────────────────────── */
.totals {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px;
  margin-bottom: 16px;
}
.total {
  border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
  background: var(--surface-2); display: flex; flex-direction: column; gap: 2px;
}
.total .n { font-size: 1.5rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.total .n.warn { color: #c9772f; }
.total .l { font-size: 0.78rem; color: var(--muted); }
.total .sub { font-size: 0.72rem; color: var(--muted); opacity: 0.85; margin-top: 2px; }

.bar { height: 6px; background: var(--surface-3); border-radius: 3px; overflow: hidden; margin: 6px 0 2px; }
.bar .fill { display: block; height: 100%; background: var(--accent); }
.bar.near .fill { background: #c9772f; }
.bar.over .fill { background: #b3492f; }

.sorter { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 0.8rem; color: var(--muted); }
.sorter select {
  background: var(--surface-2); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 4px 8px; font: inherit; font-size: 0.8rem;
}

.meter { margin: 6px 0 2px; }
.meter .bar { margin: 0 0 4px; }
.meter-text { font-size: 0.78rem; color: var(--muted); font-variant-numeric: tabular-nums; }
.meter-text strong { color: var(--text); }
.meter .pct { margin-left: 6px; font-weight: 700; color: var(--muted); }
.meter .pct.near { color: #c9772f; }
.meter .pct.over { color: #b3492f; }
.meter-more { display: block; font-size: 0.72rem; color: var(--muted); margin-top: 2px; }

.tier-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); }
.tier-badge.member, .tier-badge.members { border-color: #3d8b5f; color: #3d8b5f; }
/* Deliberately close to member rather than to admin: a perpetual account has a
   member's powers, and a colour borrowed from admin would read as authority. */
.tier-badge.perpetual { border-color: #3d7b8b; color: #3d7b8b; }
.tier-badge.admin { border-color: #8b5f3d; color: #8b5f3d; }
.tier-badge.lapsed { border-color: #b3492f; color: #b3492f; }

.field-note { font-size: 0.72rem; color: var(--muted); margin-top: 3px; }

.editor { margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--border); }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.stack { display: flex; flex-direction: column; gap: 3px; font-size: 0.78rem; color: var(--muted); }
.stack.wide { margin-top: 10px; }

input, select, textarea { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 5px 8px; font: inherit; font-size: 0.82rem; width: 100%; }

.actions { display: flex; align-items: flex-end; gap: 10px; margin-top: 12px; }
.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 12px; font: inherit; font-size: 0.82rem; cursor: pointer; }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
.btn.danger { color: #b3492f; }
.linkish { background: none; border: none; color: var(--accent, #3d8b5f); font: inherit;
  font-size: 0.8rem; cursor: pointer; padding: 0; text-decoration: underline; }

.msg { margin: 8px 0; font-size: 0.82rem; color: var(--muted); }
.msg.error { color: #b3492f; }
.msg.ok { color: #3d8b5f; }
.hint { margin: 10px 0 0; font-size: 0.76rem; color: var(--muted); }
.hint.lead { margin: 0 0 14px; font-size: 0.82rem; line-height: 1.5; max-width: 68ch; }
.hint code, .row-top code {
  font: 0.9em ui-monospace, SFMono-Regular, Menlo, monospace;
  background: var(--surface-2); border-radius: 4px; padding: 1px 5px;
}
.linkish.danger { color: #b3492f; margin-left: 10px; }
.row-top code { font-size: 0.72rem; }
</style>
