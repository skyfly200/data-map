<template>
  <div class="admin">
    <div class="head">
      <div>
        <h2>Administration</h2>
        <p class="sub">Membership tiers, Earth Engine quotas and the society's saved datasets.</p>
      </div>
    </div>

    <ClientOnly>
      <div v-if="!membership.isAdmin.value" class="gate">
        <p>This screen is for society administrators.</p>
        <NuxtLink v-if="!membership.isAuthed.value" to="/login" class="btn primary">Sign in</NuxtLink>
      </div>

      <template v-else>
        <div class="tabs" role="tablist">
          <button :class="{ on: tab === 'members' }" role="tab" @click="show('members')">Members</button>
          <button :class="{ on: tab === 'datasets' }" role="tab" @click="show('datasets')">Datasets</button>
        </div>

        <p v-if="error" class="msg error">{{ error }}</p>
        <p v-if="note" class="msg ok">{{ note }}</p>
        <p v-if="loading" class="msg">Loading…</p>

        <!-- ── Members ────────────────────────────────────────────────── -->
        <section v-if="tab === 'members' && !loading" class="panel">
          <p v-if="!members.length" class="msg">No accounts yet.</p>

          <ul v-else class="rows">
            <li v-for="m in members" :key="m.user_id" class="row-item">
              <div class="row-top">
                <strong>{{ m.display_name || m.user_id.slice(0, 8) }}</strong>
                <span class="tier-badge" :class="m.tier">{{ m.tier }}</span>
                <span v-if="lapsedFor(m)" class="tier-badge lapsed">lapsed</span>
                <!-- The reading that makes a quota actionable: a limit with no
                     usage next to it is a number nobody can judge. -->
                <span class="usage">
                  {{ m.usage.unitsThisMonth }} / {{ m.ee_quota_monthly }} units this month
                  · {{ m.usage.jobsToday }} jobs today
                </span>
                <button class="linkish" @click="edit(m)">
                  {{ editing === m.user_id ? 'Close' : 'Edit' }}
                </button>
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
import { ref, reactive, onMounted } from 'vue'
import { TIERS } from '~/netlify/lib/tiers.mjs'
import { DEFAULT_LIMITS } from '~/netlify/lib/quotas.mjs'

const membership = useMembership()
const { accessToken } = useAuth()

const LIMIT_FIELDS = Object.keys(DEFAULT_LIMITS)

const tab = ref('members')
const members = ref([])
const datasets = ref([])
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

const lapsedFor = (m) => m.member_until && new Date(m.member_until) < new Date()

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

.tier-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 999px;
  border: 1px solid var(--border); color: var(--muted); }
.tier-badge.member, .tier-badge.members { border-color: #3d8b5f; color: #3d8b5f; }
.tier-badge.admin { border-color: #8b5f3d; color: #8b5f3d; }
.tier-badge.lapsed { border-color: #b3492f; color: #b3492f; }

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
</style>
