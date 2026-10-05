// Member-defined "allowed area" sets (WANT-17), layered over public PAD-US access data.
// Auth required: Bearer token, member tier re-checked against the database (requireMemberFresh).
// Everything is scoped to the caller; a set the caller cannot see answers 404 (never 403, no leak).
//
//   GET  /.netlify/functions/access-sets
//     -> { ok:true,
//          sets:[{ id, name, scope:'user'|'club', club_id|null, club_name|null, area_count, role }],
//          clubs:[{ id, name, role }] }
//        role = 'owner' for the caller's own user-scope sets, else the caller's club role
//        ('owner'|'admin'|'member'). Writable when scope='user', or role is owner/admin
//        (a club set's creator may also keep editing while still a member).
//
//   POST application/json { action, ... }  -> { ok:true, ... } | { ok:false, error } (400 bad input,
//        401 no auth, 403 visible but not permitted, 404 not found/not visible, 503 no database)
//     create_set    { name, scope:'user'|'club', club_id? }        club scope: club owner/admin only -> { set }
//     update_set    { set_id, name }                               -> { set }
//     delete_set    { set_id }                                     deletes its areas too
//     add_area      { set_id, name, geometry, fee_status?, collecting?, notes? } -> { area:{id,...} }
//     update_area   { area_id, name?, geometry?, fee_status?, collecting?, notes? }
//     delete_area   { area_id }
//     import_geojson{ set_id, geojson }  FeatureCollection of Polygon/MultiPolygon features; name/notes
//                    (and optional fee_status/collecting) from properties -> { imported: n }; all-or-nothing
//     create_club   { name }                                       caller becomes owner -> { club }
//     add_club_member    { club_id, email | user_id, role?:'member'|'admin' }  owner/admin; only owner grants 'admin'
//     remove_club_member { club_id, email | user_id }               owner/admin; admin cannot remove admin/owner
//   geometry: GeoJSON Polygon|MultiPolygon, WGS84 lon/lat, closed rings.
//   fee_status 'free'|'fee'|'unknown' (default unknown); collecting 'allowed'|'likely_allowed'|
//   'restricted'|'prohibited'|'unknown' (default unknown). 'allowed' is owner-asserted here.
//   Caps: name 120 chars, notes 2000, 5000 vertices/area, 400 KB geometry JSON, 500 areas/set,
//   50 sets/user, 10 clubs owned/user, 100 members/club, import <= 200 features / 50000 vertices / 2 MB.
//
// Read side: GET access?include_sets=1 (see access.mjs) merges the caller's visible set areas into `areas`.

import { adminClient, requireMemberFresh } from '../lib/auth.mjs'

export const LIMITS = {
  NAME: 120, NOTES: 2000, VERTICES: 5000, GEOM_BYTES: 400_000, AREAS_PER_SET: 500, SETS_PER_USER: 50,
  CLUBS_OWNED: 10, CLUB_MEMBERS: 100, IMPORT_FEATURES: 200, IMPORT_VERTICES: 50_000, IMPORT_BYTES: 2_000_000,
  MAX_POLYGONS: 50, MAX_RINGS: 100,
}
export const FEE = ['free', 'fee', 'unknown']
export const COLLECTING = ['allowed', 'likely_allowed', 'restricted', 'prohibited', 'unknown']

class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status }
}
const bad = (m) => new HttpError(400, m)
const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})
const dbCheck = ({ error }) => { if (error) throw new HttpError(500, error.message) }

const isObj = (v) => v && typeof v === 'object' && !Array.isArray(v)
const id = (v, label) => {
  const n = Number(v)
  if (!Number.isInteger(n) || n <= 0) throw bad(`${label} must be a positive integer.`)
  return n
}
const text = (v, label, max, { required = false } = {}) => {
  if (v == null || v === '') {
    if (required) throw bad(`${label} is required.`)
    return null
  }
  if (typeof v !== 'string') throw bad(`${label} must be a string.`)
  const t = v.trim()
  if (required && !t) throw bad(`${label} is required.`)
  if (t.length > max) throw bad(`${label} may be at most ${max} characters.`)
  return t || null
}
const oneOf = (v, list, label) => {
  if (!list.includes(v)) throw bad(`${label} must be one of ${list.join(', ')}.`)
  return v
}

function checkRing(ring) {
  if (!Array.isArray(ring) || ring.length < 4) throw bad('Each ring needs at least 4 positions.')
  for (const p of ring) {
    if (!Array.isArray(p) || p.length < 2 || typeof p[0] !== 'number' || typeof p[1] !== 'number'
      || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) throw bad('Positions must be [lon, lat] numbers.')
    if (p[0] < -180 || p[0] > 180 || p[1] < -90 || p[1] > 90) throw bad('Coordinates out of range (lon -180..180, lat -90..90).')
  }
  const a = ring[0], b = ring[ring.length - 1]
  if (a[0] !== b[0] || a[1] !== b[1]) throw bad('Rings must be closed (first position equals last).')
  return ring.length
}

/** Validate a Polygon/MultiPolygon; returns { geometry, vertices } with only type+coordinates kept. */
export function validateGeometry(g) {
  if (!isObj(g) || (g.type !== 'Polygon' && g.type !== 'MultiPolygon')) throw bad('geometry must be a GeoJSON Polygon or MultiPolygon.')
  const polys = g.type === 'Polygon' ? [g.coordinates] : g.coordinates
  if (!Array.isArray(polys) || !polys.length) throw bad('geometry has no coordinates.')
  if (polys.length > LIMITS.MAX_POLYGONS) throw bad(`geometry may have at most ${LIMITS.MAX_POLYGONS} polygons.`)
  let vertices = 0
  for (const poly of polys) {
    if (!Array.isArray(poly) || !poly.length) throw bad('Polygon needs at least one ring.')
    if (poly.length > LIMITS.MAX_RINGS) throw bad(`Polygon may have at most ${LIMITS.MAX_RINGS} rings.`)
    for (const ring of poly) {
      vertices += checkRing(ring)
      if (vertices > LIMITS.VERTICES) throw bad(`geometry may have at most ${LIMITS.VERTICES} vertices.`)
    }
  }
  const geometry = { type: g.type, coordinates: g.coordinates }
  if (JSON.stringify(geometry).length > LIMITS.GEOM_BYTES) throw bad('geometry is too large.')
  return { geometry, vertices }
}

function areaFields(b, { partial = false, nameOptional = false } = {}) {
  const out = {}
  if (!partial || 'name' in b) out.name = text(b.name, 'name', LIMITS.NAME, { required: !partial && !nameOptional })
  if (!partial || 'notes' in b) out.notes = text(b.notes, 'notes', LIMITS.NOTES)
  if (!partial || 'fee_status' in b) out.fee_status = oneOf(b.fee_status ?? 'unknown', FEE, 'fee_status')
  if (!partial || 'collecting' in b) out.collecting = oneOf(b.collecting ?? 'unknown', COLLECTING, 'collecting')
  return out
}

// ---- data access -------------------------------------------------------------------------------

async function clubRole(client, clubId, userId) {
  const { data, error } = await client.from('club_members').select('role')
    .eq('club_id', clubId).eq('user_id', userId).maybeSingle()
  dbCheck({ error })
  return data?.role || null
}
const isAdminRole = (r) => r === 'owner' || r === 'admin'

/** Load a set the caller can SEE, with their role; 404 otherwise. */
async function visibleSet(client, setId, userId) {
  const { data: set, error } = await client.from('access_area_sets').select('*').eq('id', setId).maybeSingle()
  dbCheck({ error })
  if (!set) throw new HttpError(404, 'Set not found.')
  if (set.scope === 'user') {
    if (set.owner_id !== userId) throw new HttpError(404, 'Set not found.')
    return { set, role: 'owner' }
  }
  const role = await clubRole(client, set.club_id, userId)
  if (!role) throw new HttpError(404, 'Set not found.')
  return { set, role }
}
const canWriteSet = (set, role, userId) =>
  set.scope === 'user' ? role === 'owner' : isAdminRole(role) || set.owner_id === userId
const writableSet = async (client, setId, userId) => {
  const v = await visibleSet(client, setId, userId)
  if (!canWriteSet(v.set, v.role, userId)) throw new HttpError(403, 'You cannot edit this set.')
  return v
}
async function writableArea(client, areaId, userId) {
  const { data: area, error } = await client.from('access_set_areas').select('*').eq('id', areaId).maybeSingle()
  dbCheck({ error })
  if (!area) throw new HttpError(404, 'Area not found.')
  const { set, role } = await visibleSet(client, area.set_id, userId)
  const ok = canWriteSet(set, role, userId) || (set.scope === 'club' && area.created_by === userId)
  if (!ok) throw new HttpError(403, 'You cannot edit this area.')
  return { area, set }
}

async function areaCount(client, setId) {
  const { data, error } = await client.from('access_set_areas').select('id').eq('set_id', setId)
  dbCheck({ error })
  return (data || []).length
}

async function insertArea(client, setId, fields, geometry, userId) {
  const { data, error } = await client.rpc('access_set_area_insert', {
    p_set_id: setId, p_name: fields.name, p_geom: geometry, p_fee: fields.fee_status,
    p_collecting: fields.collecting, p_notes: fields.notes, p_user: userId,
  })
  dbCheck({ error })
  return data
}

async function findUserId(client, b) {
  if (b.user_id) {
    if (typeof b.user_id !== 'string') throw bad('user_id must be a string.')
    return b.user_id
  }
  const email = text(b.email, 'email', 320, { required: true }).toLowerCase()
  if (!client.auth?.admin?.listUsers) throw new HttpError(503, 'User lookup unavailable.')
  for (let page = 1; page <= 20; page++) {
    const { data, error } = await client.auth.admin.listUsers({ page, perPage: 1000 })
    dbCheck({ error })
    const users = data?.users || []
    const hit = users.find((u) => String(u.email || '').toLowerCase() === email)
    if (hit) return hit.id
    if (users.length < 1000) break
  }
  throw new HttpError(404, 'No account with that email.')
}

// ---- GET ---------------------------------------------------------------------------------------

async function listSets(client, userId) {
  const mine = await client.from('access_area_sets').select('*').eq('owner_id', userId).eq('scope', 'user')
  dbCheck(mine)
  const mem = await client.from('club_members').select('club_id, role').eq('user_id', userId)
  dbCheck(mem)
  const roles = new Map((mem.data || []).map((m) => [m.club_id, m.role]))
  let clubSets = [], clubs = []
  if (roles.size) {
    const ids = [...roles.keys()]
    const cs = await client.from('access_area_sets').select('*').in('club_id', ids).eq('scope', 'club')
    dbCheck(cs); clubSets = cs.data || []
    const cl = await client.from('clubs').select('id, name').in('id', ids)
    dbCheck(cl); clubs = cl.data || []
  }
  const clubName = new Map(clubs.map((c) => [c.id, c.name]))
  const all = [...(mine.data || []), ...clubSets]
  const counts = new Map()
  if (all.length) {
    const ar = await client.from('access_set_areas').select('set_id').in('set_id', all.map((s) => s.id))
    dbCheck(ar)
    for (const r of ar.data || []) counts.set(r.set_id, (counts.get(r.set_id) || 0) + 1)
  }
  return {
    sets: all.map((s) => ({
      id: s.id, name: s.name, scope: s.scope, club_id: s.club_id ?? null,
      club_name: s.club_id != null ? clubName.get(s.club_id) ?? null : null,
      area_count: counts.get(s.id) || 0, role: s.scope === 'user' ? 'owner' : roles.get(s.club_id),
    })),
    clubs: clubs.map((c) => ({ id: c.id, name: c.name, role: roles.get(c.id) })),
  }
}

// ---- actions -----------------------------------------------------------------------------------

const ACTIONS = {
  async create_set(client, u, b) {
    const name = text(b.name, 'name', LIMITS.NAME, { required: true })
    const scope = oneOf(b.scope, ['user', 'club'], 'scope')
    let clubId = null
    if (scope === 'club') {
      clubId = id(b.club_id, 'club_id')
      const role = await clubRole(client, clubId, u)
      if (!role) throw new HttpError(404, 'Club not found.')
      if (!isAdminRole(role)) throw new HttpError(403, 'Only club owners and admins can create club sets.')
    } else if (b.club_id != null) throw bad('club_id is only for club scope.')
    const owned = await client.from('access_area_sets').select('id').eq('owner_id', u)
    dbCheck(owned)
    if ((owned.data || []).length >= LIMITS.SETS_PER_USER) throw bad(`You can have at most ${LIMITS.SETS_PER_USER} sets.`)
    const { data, error } = await client.from('access_area_sets')
      .insert({ name, scope, owner_id: u, club_id: clubId }).select().single()
    dbCheck({ error })
    return { set: data }
  },
  async update_set(client, u, b) {
    const { set } = await writableSet(client, id(b.set_id, 'set_id'), u)
    const name = text(b.name, 'name', LIMITS.NAME, { required: true })
    const { data, error } = await client.from('access_area_sets').update({ name }).eq('id', set.id).select().single()
    dbCheck({ error })
    return { set: data }
  },
  async delete_set(client, u, b) {
    const { set } = await writableSet(client, id(b.set_id, 'set_id'), u)
    dbCheck(await client.from('access_set_areas').delete().eq('set_id', set.id))
    dbCheck(await client.from('access_area_sets').delete().eq('id', set.id))
    return {}
  },
  async add_area(client, u, b) {
    const { set } = await writableSet(client, id(b.set_id, 'set_id'), u)
    const fields = areaFields(b)
    const { geometry } = validateGeometry(b.geometry)
    if (await areaCount(client, set.id) >= LIMITS.AREAS_PER_SET) throw bad(`A set may hold at most ${LIMITS.AREAS_PER_SET} areas.`)
    const newId = await insertArea(client, set.id, fields, geometry, u)
    return { area: { id: newId, set_id: set.id, ...fields } }
  },
  async update_area(client, u, b) {
    const { area } = await writableArea(client, id(b.area_id, 'area_id'), u)
    const fields = areaFields(b, { partial: true })
    const geom = 'geometry' in b ? validateGeometry(b.geometry).geometry : null
    if (!Object.keys(fields).length && !geom) throw bad('Nothing to update.')
    if (Object.keys(fields).length) {
      dbCheck(await client.from('access_set_areas').update(fields).eq('id', area.id))
    }
    if (geom) dbCheck(await client.rpc('access_set_area_update_geom', { p_id: area.id, p_geom: geom }))
    return { area: { id: area.id, set_id: area.set_id, ...fields } }
  },
  async delete_area(client, u, b) {
    const { area } = await writableArea(client, id(b.area_id, 'area_id'), u)
    dbCheck(await client.from('access_set_areas').delete().eq('id', area.id))
    return {}
  },
  async import_geojson(client, u, b) {
    const { set } = await writableSet(client, id(b.set_id, 'set_id'), u)
    let gj = b.geojson
    if (typeof gj === 'string') {
      if (gj.length > LIMITS.IMPORT_BYTES) throw bad('geojson is too large.')
      try { gj = JSON.parse(gj) } catch { throw bad('geojson is not valid JSON.') }
    }
    if (!isObj(gj) || gj.type !== 'FeatureCollection' || !Array.isArray(gj.features)) throw bad('geojson must be a FeatureCollection.')
    if (!gj.features.length) throw bad('geojson has no features.')
    if (gj.features.length > LIMITS.IMPORT_FEATURES) throw bad(`Import may have at most ${LIMITS.IMPORT_FEATURES} features.`)
    if (await areaCount(client, set.id) + gj.features.length > LIMITS.AREAS_PER_SET) {
      throw bad(`A set may hold at most ${LIMITS.AREAS_PER_SET} areas.`)
    }
    let total = 0
    const rows = gj.features.map((f, i) => {
      try {
        if (!isObj(f) || f.type !== 'Feature') throw bad('not a Feature.')
        const p = isObj(f.properties) ? f.properties : {}
        const { geometry, vertices } = validateGeometry(f.geometry)
        total += vertices
        if (total > LIMITS.IMPORT_VERTICES) throw bad(`Import may have at most ${LIMITS.IMPORT_VERTICES} vertices.`)
        const fields = areaFields({
          name: p.name != null ? String(p.name) : null,
          notes: typeof p.notes === 'string' ? p.notes : null,
          fee_status: p.fee_status, collecting: p.collecting,
        }, { nameOptional: true })
        return { fields, geometry }
      } catch (e) {
        if (e instanceof HttpError) throw bad(`Feature ${i}: ${e.message}`)
        throw e
      }
    })
    for (const r of rows) await insertArea(client, set.id, r.fields, r.geometry, u)
    return { imported: rows.length }
  },
  async create_club(client, u, b) {
    const name = text(b.name, 'name', LIMITS.NAME, { required: true })
    const owned = await client.from('club_members').select('club_id').eq('user_id', u).eq('role', 'owner')
    dbCheck(owned)
    if ((owned.data || []).length >= LIMITS.CLUBS_OWNED) throw bad(`You can own at most ${LIMITS.CLUBS_OWNED} clubs.`)
    const { data, error } = await client.from('clubs').insert({ name, created_by: u }).select().single()
    dbCheck({ error })
    dbCheck(await client.from('club_members').insert({ club_id: data.id, user_id: u, role: 'owner' }))
    return { club: { id: data.id, name: data.name, role: 'owner' } }
  },
  async add_club_member(client, u, b) {
    const clubId = id(b.club_id, 'club_id')
    const role = await clubRole(client, clubId, u)
    if (!role) throw new HttpError(404, 'Club not found.')
    if (!isAdminRole(role)) throw new HttpError(403, 'Only club owners and admins can add members.')
    const newRole = oneOf(b.role ?? 'member', ['member', 'admin'], 'role')
    if (newRole === 'admin' && role !== 'owner') throw new HttpError(403, 'Only the club owner can add admins.')
    const target = await findUserId(client, b)
    const members = await client.from('club_members').select('user_id').eq('club_id', clubId)
    dbCheck(members)
    const list = members.data || []
    if (list.some((m) => m.user_id === target)) throw bad('Already a member.')
    if (list.length >= LIMITS.CLUB_MEMBERS) throw bad(`A club may have at most ${LIMITS.CLUB_MEMBERS} members.`)
    dbCheck(await client.from('club_members').insert({ club_id: clubId, user_id: target, role: newRole }))
    return { member: { club_id: clubId, user_id: target, role: newRole } }
  },
  async remove_club_member(client, u, b) {
    const clubId = id(b.club_id, 'club_id')
    const role = await clubRole(client, clubId, u)
    if (!role) throw new HttpError(404, 'Club not found.')
    if (!isAdminRole(role)) throw new HttpError(403, 'Only club owners and admins can remove members.')
    const target = await findUserId(client, b)
    const tRole = await clubRole(client, clubId, target)
    if (!tRole) throw new HttpError(404, 'Not a member.')
    if (tRole === 'owner') throw new HttpError(403, 'The club owner cannot be removed.')
    if (tRole === 'admin' && role !== 'owner') throw new HttpError(403, 'Only the club owner can remove admins.')
    dbCheck(await client.from('club_members').delete().eq('club_id', clubId).eq('user_id', target))
    return {}
  },
}

// ---- handler -----------------------------------------------------------------------------------

/** `auth` is the result of requireMemberFresh (injected in tests). */
export async function handleAccessSets(request, client, auth) {
  if (!auth?.ok) return auth?.response || json({ ok: false, error: 'Sign in required.' }, 401)
  const userId = auth.user?.id
  if (!userId) return json({ ok: false, error: 'Sign in required.' }, 401)
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)
  try {
    if (request.method === 'GET') return json({ ok: true, ...await listSets(client, userId) })
    if (request.method !== 'POST') return json({ ok: false, error: 'Use GET or POST.' }, 405)
    const raw = await request.text()
    if (raw.length > LIMITS.IMPORT_BYTES + 10_000) return json({ ok: false, error: 'Request too large.' }, 413)
    let body
    try { body = JSON.parse(raw) } catch { throw bad('Send JSON.') }
    if (!isObj(body)) throw bad('Send a JSON object.')
    const fn = Object.hasOwn(ACTIONS, body.action) ? ACTIONS[body.action] : null
    if (!fn) throw bad(`action must be one of ${Object.keys(ACTIONS).join(', ')}.`)
    return json({ ok: true, ...await fn(client, userId, body) })
  } catch (e) {
    if (e instanceof HttpError) return json({ ok: false, error: e.message }, e.status)
    return json({ ok: false, error: 'Unexpected error.' }, 500)
  }
}

export default async (request) => {
  const auth = await requireMemberFresh(request)
  return handleAccessSets(request, adminClient(), auth)
}
