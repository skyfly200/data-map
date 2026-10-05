<template>
  <div class="areas">
    <h2>My areas</h2>
    <p class="sub">
      Named sets of places you may collect, private to you or shared with a club. You assert what
      is in them: they are shown on your map as <strong>Your area</strong> or <strong>Club area</strong>
      and are not verified by Nexstrata.
    </p>

    <ClientOnly>
      <div v-if="membership.configured && !membership.isAuthed.value" class="gate">
        <p>Managing allowed areas is a membership benefit.</p>
        <NuxtLink to="/login" class="btn primary">Sign in</NuxtLink>
      </div>
      <div v-else-if="membership.configured && !membership.isMember.value" class="gate">
        <p>Allowed areas are a benefit of Front Range Mycological Society membership.</p>
      </div>
      <div v-else-if="api.unavailable.value" class="gate" role="status">
        <p>Allowed areas are not available on this server yet. Check back soon.</p>
      </div>

      <template v-else>
        <p v-if="api.error.value" class="msg error" role="alert">{{ api.error.value }}</p>

        <!-- ── Sets ───────────────────────────────────────────────── -->
        <section class="panel" aria-labelledby="sets-h">
          <h3 id="sets-h">Your sets</h3>
          <p v-if="api.loading.value" class="muted">Loading…</p>
          <p v-else-if="!api.sets.value.length" class="muted">No sets yet. Create one below.</p>
          <ul v-else class="set-list">
            <li v-for="s in api.sets.value" :key="s.id" :class="{ active: s.id === activeId }">
              <button type="button" class="set-btn" :aria-pressed="s.id === activeId" @click="openSet(s.id)">
                <span class="set-name">{{ s.name }}</span>
                <span class="badge" :class="s.scope">{{ s.scope === 'club' ? (s.club_name || 'Club') : 'Private' }}</span>
                <span class="badge role">{{ s.role }}</span>
                <span class="muted">{{ s.area_count }} area{{ s.area_count === 1 ? '' : 's' }}</span>
              </button>
            </li>
          </ul>

          <form class="row" @submit.prevent="onCreateSet">
            <label for="new-set">New set</label>
            <input id="new-set" v-model="newSet.name" maxlength="80" placeholder="e.g. Pike NF spots" required>
            <label for="new-scope" class="sr-only">Shared with</label>
            <select id="new-scope" v-model="newSet.clubId">
              <option value="">Private to me</option>
              <option v-for="c in clubs" :key="c.id" :value="c.id">Club: {{ c.name }}</option>
            </select>
            <button class="btn primary" :disabled="busy">Create set</button>
          </form>
        </section>

        <!-- ── Clubs ──────────────────────────────────────────────── -->
        <section class="panel" aria-labelledby="clubs-h">
          <h3 id="clubs-h">Clubs</h3>
          <p class="muted">A private club lets several members share one set of areas.</p>
          <form class="row" @submit.prevent="onCreateClub">
            <label for="new-club">New club</label>
            <input id="new-club" v-model="newClub" maxlength="80" placeholder="Club name" required>
            <button class="btn" :disabled="busy">Create club</button>
          </form>
          <form v-if="manageableClubs.length" class="row" @submit.prevent="onAddMember">
            <label for="mem-club">Add member to</label>
            <select id="mem-club" v-model="member.clubId" required>
              <option value="" disabled>Choose club</option>
              <option v-for="c in manageableClubs" :key="c.id" :value="c.id">{{ c.name }}</option>
            </select>
            <label for="mem-email" class="sr-only">Member email</label>
            <input id="mem-email" v-model="member.email" type="email" placeholder="member@example.org" required>
            <button class="btn" :disabled="busy">Add member</button>
          </form>
          <p v-if="notice" class="msg ok" role="status">{{ notice }}</p>
          <ul v-if="members.length" class="members" aria-label="Club members">
            <li v-for="m in members" :key="m.user_id">
              {{ m.email || m.display_name || m.user_id }} <span class="badge role">{{ m.role }}</span>
              <button v-if="canManage(set?.role) && m.role !== 'owner'" type="button" class="link danger"
                      :aria-label="`Remove ${m.email || m.user_id}`" @click="onRemoveMember(m)">Remove</button>
            </li>
          </ul>
        </section>

        <!-- ── Selected set ───────────────────────────────────────── -->
        <section v-if="set" class="panel" aria-labelledby="set-h">
          <div class="panel-head">
            <h3 id="set-h">{{ set.name }}</h3>
            <span class="badge" :class="set.scope">{{ scopeLabel(set.scope) }}</span>
          </div>

          <form v-if="writable" class="row" @submit.prevent="onRename">
            <label for="rename">Rename</label>
            <input id="rename" v-model="renameTo" maxlength="80" required>
            <button class="btn" :disabled="busy">Save name</button>
            <button type="button" class="btn danger" :disabled="busy" @click="onDeleteSet">Delete set</button>
          </form>
          <p v-else class="muted">You can view this set but not change it ({{ set.role }}).</p>

          <AreaDrawMap :areas="api.areas.value" :selected-id="selectedId" :drawing="drawing"
                       :label="scopeLabel(set.scope)" @select="selectArea" @drawn="onDrawn" @cancel="drawing = false" />

          <div v-if="writable && !drawing" class="row tools">
            <button type="button" class="btn primary" @click="startDraw">Draw polygon</button>
            <label class="btn file-btn">
              Import GeoJSON
              <input type="file" accept=".geojson,.json,application/geo+json,application/json" class="sr-only" @change="onFile">
            </label>
          </div>
          <p v-if="formError && !form" class="msg error" role="alert">{{ formError }}</p>
          <p v-if="importMsg"class="msg" :class="importMsg.kind" :role="importMsg.kind === 'error' ? 'alert' : 'status'">{{ importMsg.text }}</p>

          <!-- Editor: a new drawn polygon, or an existing area. -->
          <form v-if="form" class="editor" aria-labelledby="ed-h" @submit.prevent="onSaveArea">
            <h4 id="ed-h">{{ form.id ? 'Edit area' : 'New area' }}</h4>
            <label for="a-name">Name</label>
            <input id="a-name" v-model="form.name" maxlength="120" required :disabled="!writable">
            <label for="a-fee">Fee</label>
            <select id="a-fee" v-model="form.fee_status" :disabled="!writable">
              <option v-for="f in FEE_STATUSES" :key="f" :value="f">{{ FEE_LABELS[f] }}</option>
            </select>
            <label for="a-col">Collecting</label>
            <select id="a-col" v-model="form.collecting" :disabled="!writable">
              <option v-for="c in COLLECTING_STATUSES" :key="c" :value="c">{{ COLLECTING_LABELS[c] }}</option>
            </select>
            <label for="a-notes">Notes</label>
            <textarea id="a-notes" v-model="form.notes" rows="3" maxlength="2000" :disabled="!writable"></textarea>
            <p class="muted assert">
              You assert this information. It is shown on your map as “{{ scopeLabel(set.scope) }}”
              and is not checked against land-manager records.
            </p>
            <p v-if="formError" class="msg error" role="alert">{{ formError }}</p>
            <div v-if="writable" class="row">
              <button class="btn primary" :disabled="busy">{{ form.id ? 'Save changes' : 'Add area' }}</button>
              <button v-if="form.id" type="button" class="btn danger" :disabled="busy" @click="onDeleteArea">Delete area</button>
              <button type="button" class="btn" @click="form = null">Close</button>
            </div>
          </form>

          <h4>Areas ({{ api.areas.value.length }})</h4>
          <p v-if="!api.areas.value.length" class="muted">No areas in this set yet.</p>
          <ul v-else class="area-list">
            <li v-for="a in api.areas.value" :key="a.properties.id">
              <button type="button" class="set-btn" :aria-pressed="a.properties.id === selectedId" @click="selectArea(a.properties.id)">
                <span class="set-name">{{ a.properties.name }}</span>
                <span class="badge">{{ COLLECTING_LABELS[a.properties.collecting] }}</span>
                <span class="badge">{{ FEE_LABELS[a.properties.fee_status] }}</span>
              </button>
            </li>
          </ul>
        </section>
      </template>
    </ClientOnly>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import {
  AreaError, COLLECTING_LABELS, COLLECTING_STATUSES, FEE_LABELS, FEE_STATUSES,
  canManage, ringToPolygon, scopeLabel,
} from '~/composables/areaGeometry'
import { useAccessSets } from '~/composables/useAccessSets'

useHead({ title: 'My areas · Nexstrata' })

const membership = useMembership()
const api = useAccessSets()

const set = computed(() => api.current.value)
const activeId = computed(() => set.value?.id || '')
const writable = computed(() => !!set.value && (set.value.scope === 'user' || canManage(set.value.role) || set.value.role === 'editor'))
const members = computed<any[]>(() => (set.value as any)?.members || [])

// Clubs the viewer belongs to, derived from their club-scoped sets.
const clubs = computed(() => {
  const m = new Map<string, { id: string, name: string, role: string }>()
  for (const s of api.sets.value) if (s.club_id) m.set(s.club_id, { id: s.club_id, name: s.club_name || 'Club', role: s.role })
  return [...m.values()]
})
const manageableClubs = computed(() => clubs.value.filter((c) => canManage(c.role)))

const busy = ref(false)
const notice = ref('')
const newSet = reactive({ name: '', clubId: '' })
const newClub = ref('')
const member = reactive({ clubId: '', email: '' })
const renameTo = ref('')
const drawing = ref(false)
const selectedId = ref('')
const form = ref<any>(null)
const formError = ref('')
const importMsg = ref<{ kind: string, text: string } | null>(null)

async function run(fn: () => Promise<any>, ok = '') {
  busy.value = true
  notice.value = ''
  try { const r = await fn(); notice.value = ok; return r } catch { return undefined } finally { busy.value = false }
}

async function openSet(id: string) {
  form.value = null; selectedId.value = ''; drawing.value = false; importMsg.value = null
  await run(() => api.open(id))
  renameTo.value = set.value?.name || ''
}

async function onCreateSet() {
  const clubId = newSet.clubId
  const r = await run(() => api.createSet(newSet.name, clubId ? 'club' : 'user', clubId || undefined), 'Set created.')
  if (r) {
    newSet.name = ''
    const created = r.set?.id ? r.set.id : api.sets.value.find((s) => s.name === r.set?.name)?.id
    if (created) await openSet(created)
  }
}
async function onCreateClub() {
  if (await run(() => api.createClub(newClub.value), 'Club created.')) newClub.value = ''
}
async function onAddMember() {
  if (await run(() => api.addMember(member.clubId, member.email), 'Member added.')) {
    member.email = ''
    if (set.value) await api.open(set.value.id).catch(() => {})
  }
}
async function onRemoveMember(m: any) {
  if (!set.value?.club_id || !confirm(`Remove ${m.email || 'this member'} from the club?`)) return
  await run(() => api.removeMember(set.value!.club_id!, m.user_id), 'Member removed.')
  await api.open(set.value!.id).catch(() => {})
}
async function onRename() { await run(() => api.renameSet(set.value!.id, renameTo.value), 'Renamed.'); await api.open(set.value!.id).catch(() => {}) }
async function onDeleteSet() {
  if (!confirm(`Delete “${set.value!.name}” and all its areas? This cannot be undone.`)) return
  if (await run(() => api.deleteSet(set.value!.id))) { api.current.value = null; api.areas.value = []; form.value = null }
}

function startDraw() { form.value = null; selectedId.value = ''; formError.value = ''; drawing.value = true }
function onDrawn(ring: Array<[number, number]>) {
  drawing.value = false
  try {
    form.value = { id: '', name: '', fee_status: 'unknown', collecting: 'unknown', notes: '', geometry: ringToPolygon(ring) }
    formError.value = ''
  } catch (e: any) { formError.value = e.message }
}
function selectArea(id: string) {
  const a = api.areas.value.find((x) => x.properties.id === id)
  if (!a) return
  selectedId.value = id
  formError.value = ''
  form.value = { id, ...a.properties, geometry: a.geometry }
}
async function onSaveArea() {
  formError.value = ''
  const f = form.value
  try {
    const r = await run(() => (f.id ? api.updateArea(set.value!.id, f.id, f) : api.addArea(set.value!.id, f)), 'Saved.')
    if (r) form.value = null
  } catch (e: any) { formError.value = e instanceof AreaError ? e.message : api.error.value }
  if (api.error.value && form.value) formError.value = api.error.value
}
async function onDeleteArea() {
  if (!confirm(`Delete area “${form.value.name}”?`)) return
  if (await run(() => api.deleteArea(set.value!.id, form.value.id))) { form.value = null; selectedId.value = '' }
}

async function onFile(ev: Event) {
  const input = ev.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  importMsg.value = null
  if (!file || !set.value) return
  try {
    const r = await run(async () => api.importGeojson(set.value!.id, await file.text()))
    if (r) importMsg.value = { kind: 'ok', text: `Imported ${r.count} area${r.count === 1 ? '' : 's'}${r.skipped ? `; skipped ${r.skipped} that were not valid polygons` : ''}.` }
    else if (api.error.value) importMsg.value = { kind: 'error', text: api.error.value }
  } catch (e: any) { importMsg.value = { kind: 'error', text: e.message } }
}

onMounted(() => { api.refresh() })
</script>

<style scoped>
.areas { padding: 16px 18px; max-width: 860px; margin: 0 auto; }
.sub { margin: 2px 0 14px; color: var(--muted); font-size: 0.82rem; max-width: 62ch; }
.muted { color: var(--muted); font-size: 0.84rem; margin: 4px 0; }
.sr-only { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.gate { border: 1px solid var(--border); border-radius: 10px; padding: 20px; background: var(--surface);
  display: flex; flex-direction: column; gap: 10px; align-items: flex-start; }
.gate p { margin: 0; color: var(--muted); font-size: 0.88rem; }
.panel { border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; background: var(--surface); margin-bottom: 18px; }
.panel h3 { margin: 0 0 10px; font-size: 0.95rem; }
.panel h4 { margin: 14px 0 6px; font-size: 0.88rem; }
.panel-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
.row > label:not(.btn) { font-size: 0.82rem; color: var(--muted); }
.row input, .row select { flex: 1 1 160px; min-width: 0; min-height: 40px; padding: 4px 8px;
  border: 1px solid var(--border); border-radius: 6px; font: inherit; font-size: 0.88rem; }
.set-list, .area-list, .members { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }
.members { margin-top: 10px; font-size: 0.86rem; }
.set-btn { width: 100%; text-align: left; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; min-height: 44px;
  background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; font: inherit; cursor: pointer; }
.set-btn[aria-pressed="true"], .active .set-btn { border-color: var(--accent, #3d8b5f); box-shadow: 0 0 0 1px var(--accent, #3d8b5f); }
.set-name { font-weight: 600; flex: 1 1 auto; }
.badge { font-size: 0.72rem; border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px; color: var(--muted); }
.badge.club { color: var(--accent, #3d8b5f); border-color: var(--accent, #3d8b5f); }
.btn { background: var(--bg); color: var(--text); border: 1px solid var(--border); border-radius: 6px;
  padding: 6px 12px; font: inherit; font-size: 0.84rem; cursor: pointer; min-height: 40px; text-decoration: none;
  display: inline-flex; align-items: center; }
.btn:disabled { opacity: 0.55; cursor: default; }
.btn.primary { background: var(--accent, #3d8b5f); color: #fff; border-color: transparent; }
.btn.danger, .link.danger { color: #b3492f; }
.link { background: none; border: 0; font: inherit; font-size: 0.8rem; cursor: pointer; text-decoration: underline; margin-left: 8px; padding: 8px 4px; }
.file-btn { cursor: pointer; }
.file-btn:focus-within { outline: 2px solid var(--accent, #3d8b5f); outline-offset: 2px; }
.btn:focus-visible, .set-btn:focus-visible, .link:focus-visible { outline: 2px solid var(--accent, #3d8b5f); outline-offset: 2px; }
.editor { display: grid; grid-template-columns: 90px 1fr; gap: 8px 10px; align-items: center; margin-top: 14px;
  padding-top: 10px; border-top: 1px solid var(--border); }
.editor h4, .editor .assert, .editor .msg, .editor .row { grid-column: 1 / -1; }
.editor label { font-size: 0.82rem; color: var(--muted); }
.editor input, .editor select, .editor textarea { min-width: 0; padding: 6px 8px; border: 1px solid var(--border); border-radius: 6px; font: inherit; font-size: 0.88rem; }
.msg.error { color: #b3492f; }
.msg.ok { color: #3d8b5f; }
.msg { font-size: 0.84rem; margin: 8px 0 0; }
@media (max-width: 600px) {
  .areas { padding: 12px 16px; }
  .editor { grid-template-columns: 1fr; }
}
</style>
