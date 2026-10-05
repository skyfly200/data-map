import test from 'node:test'
import assert from 'node:assert/strict'
import { handleAccessSets, validateGeometry, LIMITS } from '../netlify/functions/access-sets.mjs'
import { handleAccess } from '../netlify/functions/access.mjs'

// Minimal in-memory Supabase fake: tables + chained eq/in filters, rpc stubs, auth.admin.listUsers.
function fake({ users = [] } = {}) {
  const t = { access_regions: [], access_area_sets: [], access_set_areas: [], clubs: [], club_members: [] }
  const seq = { access_area_sets: 0, access_set_areas: 0, clubs: 0 }
  const rpcCalls = []
  const query = (table) => {
    let op = 'select', payload, filters = []
    const run = () => {
      const rows = t[table]
      const match = (r) => filters.every(([k, v, isIn]) => (isIn ? v.includes(r[k]) : r[k] === v))
      if (op === 'insert') {
        const r = { ...payload }
        if (table in seq) r.id = ++seq[table]
        rows.push(r); return { data: [r], error: null }
      }
      if (op === 'update') { const hit = rows.filter(match); hit.forEach((r) => Object.assign(r, payload)); return { data: hit, error: null } }
      if (op === 'delete') { t[table] = rows.filter((r) => !match(r)); return { data: null, error: null } }
      return { data: rows.filter(match), error: null }
    }
    const b = {
      select: () => b, eq: (k, v) => (filters.push([k, v]), b), in: (k, v) => (filters.push([k, v, true]), b),
      insert: (p) => (op = 'insert', payload = p, b), update: (p) => (op = 'update', payload = p, b),
      delete: () => (op = 'delete', b),
      maybeSingle: async () => { const r = run(); return { data: r.data?.[0] ?? null, error: r.error } },
      single: async () => { const r = run(); return { data: r.data?.[0] ?? null, error: r.error } },
      then: (res, rej) => Promise.resolve(run()).then(res, rej),
    }
    return b
  }
  const client = {
    t, rpcCalls,
    from: query,
    rpc: async (name, args) => {
      rpcCalls.push([name, args])
      if (name === 'access_set_area_insert') {
        const id = ++seq.access_set_areas
        t.access_set_areas.push({ id, set_id: args.p_set_id, name: args.p_name, fee_status: args.p_fee,
          collecting: args.p_collecting, notes: args.p_notes, created_by: args.p_user })
        return { data: id, error: null }
      }
      if (name === 'access_set_areas_in_bbox') return { data: client.setRows?.[args.p_user] || [], error: null }
      if (name === 'access_areas_in_bbox') return { data: client.padus || [], error: null }
      return { data: null, error: null }
    },
    auth: { admin: { listUsers: async () => ({ data: { users }, error: null }) } },
  }
  return client
}
const as = (id) => ({ ok: true, user: { id } })
const sq = (x = 0, y = 0, d = 1) => ({ type: 'Polygon', coordinates: [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]] })
const post = (client, auth, body) => handleAccessSets(new Request('http://x/a', { method: 'POST', body: JSON.stringify(body) }), client, auth)
  .then(async (r) => ({ status: r.status, ...(await r.json()) }))
const get = (client, auth) => handleAccessSets(new Request('http://x/a'), client, auth).then(async (r) => ({ status: r.status, ...(await r.json()) }))

test('requires auth', async () => {
  const r = await handleAccessSets(new Request('http://x/a'), fake(), { ok: false, response: new Response('{}', { status: 401 }) })
  assert.equal(r.status, 401)
  assert.equal((await handleAccessSets(new Request('http://x/a'), fake(), { ok: true, user: null })).status, 401)
})

test('geometry validation', () => {
  assert.equal(validateGeometry(sq()).vertices, 5)
  for (const g of [null, { type: 'Point', coordinates: [0, 0] }, { type: 'Polygon', coordinates: [] },
    { type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1]]] },
    { type: 'Polygon', coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1]]] }, // open
    { type: 'Polygon', coordinates: [[[0, 0], [200, 0], [1, 1], [0, 0]]] },
    { type: 'Polygon', coordinates: [[[0, 0], ['a', 0], [1, 1], [0, 0]]] }]) {
    assert.throws(() => validateGeometry(g), /./)
  }
  const ring = Array.from({ length: LIMITS.VERTICES }, (_, i) => [Math.cos(i), Math.sin(i)])
  ring.push(ring[0])
  assert.throws(() => validateGeometry({ type: 'Polygon', coordinates: [ring] }), /vertices/)
})

test('create/add/list/update/delete round trip for a user set', async () => {
  const c = fake()
  const s = await post(c, as('u1'), { action: 'create_set', name: ' Mine ', scope: 'user' })
  assert.equal(s.set.name, 'Mine')
  const a = await post(c, as('u1'), { action: 'add_area', set_id: s.set.id, name: 'Woods', geometry: sq(), collecting: 'allowed', fee_status: 'free', notes: 'n' })
  assert.equal(a.status, 200)
  assert.equal(c.t.access_set_areas[0].collecting, 'allowed')
  const l = await get(c, as('u1'))
  assert.deepEqual(l.sets, [{ id: s.set.id, name: 'Mine', scope: 'user', club_id: null, club_name: null, area_count: 1, role: 'owner' }])
  assert.equal((await post(c, as('u1'), { action: 'update_area', area_id: a.area.id, name: 'W2' })).status, 200)
  assert.equal(c.t.access_set_areas[0].name, 'W2')
  assert.equal((await post(c, as('u1'), { action: 'update_area', area_id: a.area.id, geometry: sq(2, 2) })).status, 200)
  assert.equal(c.rpcCalls.at(-1)[0], 'access_set_area_update_geom')
  assert.equal((await post(c, as('u1'), { action: 'delete_area', area_id: a.area.id })).status, 200)
  assert.equal((await post(c, as('u1'), { action: 'delete_set', set_id: s.set.id })).status, 200)
  assert.equal(c.t.access_area_sets.length, 0)
})

test('input validation errors', async () => {
  const c = fake()
  const s = (await post(c, as('u1'), { action: 'create_set', name: 'S', scope: 'user' })).set
  const base = { action: 'add_area', set_id: s.id, name: 'A', geometry: sq() }
  for (const patch of [{ name: '' }, { name: 'x'.repeat(121) }, { notes: 'x'.repeat(2001) }, { collecting: 'yes' },
    { fee_status: 'cheap' }, { geometry: { type: 'Point', coordinates: [0, 0] } }, { set_id: 'abc' }]) {
    assert.equal((await post(c, as('u1'), { ...base, ...patch })).status, 400, JSON.stringify(patch).slice(0, 40))
  }
  assert.equal((await post(c, as('u1'), { action: 'nope' })).status, 400)
  assert.equal((await post(c, as('u1'), { action: 'create_set', name: 'x', scope: 'club' })).status, 400)
  assert.equal((await post(c, as('u1'), { action: 'create_set', name: 'x', scope: 'user', club_id: 3 })).status, 400)
  const r = await handleAccessSets(new Request('http://x/a', { method: 'POST', body: '{nope' }), c, as('u1'))
  assert.equal(r.status, 400)
  assert.equal((await handleAccessSets(new Request('http://x/a', { method: 'PUT' }), c, as('u1'))).status, 405)
})

test('cross-user denial: other users see 404 and cannot touch a private set', async () => {
  const c = fake()
  const s = (await post(c, as('u1'), { action: 'create_set', name: 'S', scope: 'user' })).set
  const a = (await post(c, as('u1'), { action: 'add_area', set_id: s.id, name: 'A', geometry: sq() })).area
  for (const body of [{ action: 'add_area', set_id: s.id, name: 'x', geometry: sq() }, { action: 'update_set', set_id: s.id, name: 'h' },
    { action: 'delete_set', set_id: s.id }, { action: 'update_area', area_id: a.id, name: 'h' }, { action: 'delete_area', area_id: a.id },
    { action: 'import_geojson', set_id: s.id, geojson: { type: 'FeatureCollection', features: [] } }]) {
    assert.equal((await post(c, as('u2'), body)).status, 404, body.action)
  }
  assert.deepEqual((await get(c, as('u2'))).sets, [])
  assert.equal(c.t.access_area_sets.length, 1)
  assert.equal(c.t.access_set_areas.length, 1)
})

test('clubs: owner creates, adds by email; members read but cannot write; outsiders denied', async () => {
  const c = fake({ users: [{ id: 'm1', email: 'Mem@x.org' }, { id: 'a1', email: 'adm@x.org' }, { id: 'o2', email: 'out@x.org' }] })
  const club = (await post(c, as('own'), { action: 'create_club', name: 'FRMS' })).club
  assert.equal(club.role, 'owner')
  assert.equal((await post(c, as('own'), { action: 'add_club_member', club_id: club.id, email: 'mem@x.org' })).member.user_id, 'm1')
  assert.equal((await post(c, as('own'), { action: 'add_club_member', club_id: club.id, user_id: 'a1', role: 'admin' })).status, 200)
  assert.equal((await post(c, as('own'), { action: 'add_club_member', club_id: club.id, email: 'mem@x.org' })).status, 400)
  assert.equal((await post(c, as('own'), { action: 'add_club_member', club_id: club.id, email: 'ghost@x.org' })).status, 404)
  assert.equal((await post(c, as('m1'), { action: 'add_club_member', club_id: club.id, user_id: 'o2' })).status, 403)
  assert.equal((await post(c, as('o2'), { action: 'add_club_member', club_id: club.id, user_id: 'o2' })).status, 404)
  assert.equal((await post(c, as('m1'), { action: 'create_set', name: 'X', scope: 'club', club_id: club.id })).status, 403)
  assert.equal((await post(c, as('o2'), { action: 'create_set', name: 'X', scope: 'club', club_id: club.id })).status, 404)
  assert.equal((await post(c, as('a1'), { action: 'add_club_member', club_id: club.id, user_id: 'o2', role: 'admin' })).status, 403)

  const set = (await post(c, as('a1'), { action: 'create_set', name: 'Club land', scope: 'club', club_id: club.id })).set
  assert.equal(set.club_id, club.id)
  const area = (await post(c, as('own'), { action: 'add_area', set_id: set.id, name: 'Z', geometry: sq() })).area
  const ml = await get(c, as('m1'))
  assert.deepEqual(ml.sets.map((x) => [x.name, x.club_name, x.role, x.area_count]), [['Club land', 'FRMS', 'member', 1]])
  for (const body of [{ action: 'add_area', set_id: set.id, name: 'x', geometry: sq() }, { action: 'update_set', set_id: set.id, name: 'h' },
    { action: 'delete_set', set_id: set.id }, { action: 'update_area', area_id: area.id, name: 'h' }, { action: 'delete_area', area_id: area.id }]) {
    assert.equal((await post(c, as('m1'), body)).status, 403, body.action)
  }
  for (const body of [{ action: 'add_area', set_id: set.id, name: 'x', geometry: sq() }, { action: 'delete_set', set_id: set.id }, { action: 'delete_area', area_id: area.id }]) {
    assert.equal((await post(c, as('o2'), body)).status, 404, body.action)
  }
  assert.deepEqual((await get(c, as('o2'))).sets, [])
  assert.equal((await post(c, as('a1'), { action: 'remove_club_member', club_id: club.id, user_id: 'own' })).status, 403)
  assert.equal((await post(c, as('own'), { action: 'remove_club_member', club_id: club.id, user_id: 'own' })).status, 403)
  assert.equal((await post(c, as('m1'), { action: 'remove_club_member', club_id: club.id, user_id: 'a1' })).status, 403)
  assert.equal((await post(c, as('a1'), { action: 'remove_club_member', club_id: club.id, email: 'mem@x.org' })).status, 200)
  assert.deepEqual((await get(c, as('m1'))).sets, [])
  assert.equal((await post(c, as('m1'), { action: 'delete_area', area_id: area.id })).status, 404)
})

test('removed club set creator loses access', async () => {
  const c = fake({ users: [{ id: 'm1', email: 'm@x.org' }] })
  const club = (await post(c, as('own'), { action: 'create_club', name: 'C' })).club
  await post(c, as('own'), { action: 'add_club_member', club_id: club.id, user_id: 'm1', role: 'admin' })
  const set = (await post(c, as('m1'), { action: 'create_set', name: 'S', scope: 'club', club_id: club.id })).set
  await post(c, as('own'), { action: 'remove_club_member', club_id: club.id, user_id: 'm1' })
  assert.equal((await post(c, as('m1'), { action: 'update_set', set_id: set.id, name: 'x' })).status, 404)
})

test('caps: areas per set, sets per user, clubs, import size', async () => {
  const c = fake()
  const s = (await post(c, as('u1'), { action: 'create_set', name: 'S', scope: 'user' })).set
  for (let i = 0; i < LIMITS.AREAS_PER_SET; i++) c.t.access_set_areas.push({ id: 1000 + i, set_id: s.id })
  assert.equal((await post(c, as('u1'), { action: 'add_area', set_id: s.id, name: 'x', geometry: sq() })).status, 400)
  const c2 = fake()
  for (let i = 0; i < LIMITS.SETS_PER_USER; i++) c2.t.access_area_sets.push({ id: 500 + i, owner_id: 'u1', scope: 'user' })
  assert.equal((await post(c2, as('u1'), { action: 'create_set', name: 'x', scope: 'user' })).status, 400)
  const c3 = fake()
  for (let i = 0; i < LIMITS.CLUBS_OWNED; i++) c3.t.club_members.push({ club_id: 900 + i, user_id: 'u1', role: 'owner' })
  assert.equal((await post(c3, as('u1'), { action: 'create_club', name: 'x' })).status, 400)
  const c4 = fake()
  const s4 = (await post(c4, as('u1'), { action: 'create_set', name: 'S', scope: 'user' })).set
  const feat = (i) => ({ type: 'Feature', properties: { name: `f${i}` }, geometry: sq(i % 100, 0) })
  const tooMany = { type: 'FeatureCollection', features: Array.from({ length: LIMITS.IMPORT_FEATURES + 1 }, (_, i) => feat(i)) }
  assert.equal((await post(c4, as('u1'), { action: 'import_geojson', set_id: s4.id, geojson: tooMany })).status, 400)
  assert.equal(c4.t.access_set_areas.length, 0)
})

test('import_geojson: maps properties, all-or-nothing', async () => {
  const c = fake()
  const s = (await post(c, as('u1'), { action: 'create_set', name: 'S', scope: 'user' })).set
  const good = { type: 'FeatureCollection', features: [
    { type: 'Feature', properties: { name: 'A', notes: 'hi', collecting: 'allowed' }, geometry: sq() },
    { type: 'Feature', properties: {}, geometry: { type: 'MultiPolygon', coordinates: [sq(5, 5).coordinates] } },
  ] }
  const r = await post(c, as('u1'), { action: 'import_geojson', set_id: s.id, geojson: good })
  assert.equal(r.imported, 2)
  assert.deepEqual(c.t.access_set_areas.map((a) => [a.name, a.notes, a.collecting]), [['A', 'hi', 'allowed'], [null, null, 'unknown']])
  const badFc = { type: 'FeatureCollection', features: [good.features[0], { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: [0, 0] } }] }
  const r2 = await post(c, as('u1'), { action: 'import_geojson', set_id: s.id, geojson: badFc })
  assert.equal(r2.status, 400)
  assert.match(r2.error, /Feature 1/)
  assert.equal(c.t.access_set_areas.length, 2)
  assert.equal((await post(c, as('u1'), { action: 'import_geojson', set_id: s.id, geojson: { type: 'Feature' } })).status, 400)
})

// ---- access.mjs merge ----
const padusRow = { id: 7, name: 'NF', manager_type: 'usfs', public_access: 'open', fee_status: 'free', fee_source: 'estimated', collecting: 'likely_allowed', collecting_source: 'estimated', geometry: sq() }
const setRow = (id, scope, extra = {}) => ({ id, name: `a${id}`, fee_status: 'free', collecting: 'allowed', notes: 'n', set_id: 3, set_name: 'Mine', scope, geometry: sq(), ...extra })
const call = async (client, qs, user) => {
  const r = await handleAccess(new Request(`http://x/access?bbox=0,0,1,1${qs}`), client, async () => user)
  return { r, body: await r.json() }
}

test('access.mjs: no include_sets or no auth leaves PAD-US only', async () => {
  const c = fake()
  c.padus = [padusRow]
  c.setRows = { u1: [setRow(1, 'user')] }
  let { r, body } = await call(c, '', { id: 'u1' })
  assert.equal(body.areas.features.length, 1)
  assert.equal(body.areas.features[0].properties.source, 'padus')
  assert.equal(r.headers.get('cache-control'), 'public, max-age=300')
  assert.equal(c.rpcCalls.some(([n]) => n === 'access_set_areas_in_bbox'), false)
  ;({ r, body } = await call(c, '&include_sets=1', null))
  assert.equal(body.areas.features.length, 1)
  assert.equal(body.sets_included, false)
  assert.equal(r.headers.get('cache-control'), 'public, max-age=300')
  assert.equal(c.rpcCalls.some(([n]) => n === 'access_set_areas_in_bbox'), false)
})

test('access.mjs: include_sets merges caller sets with source and owner_asserted', async () => {
  const c = fake()
  c.padus = [padusRow]
  c.setRows = { u1: [setRow(1, 'user'), setRow(2, 'club', { collecting: 'likely_allowed', fee_status: 'fee' })], u2: [setRow(9, 'user')] }
  const { r, body } = await call(c, '&include_sets=1', { id: 'u1' })
  const f = body.areas.features
  assert.deepEqual(f.map((x) => x.properties.source), ['padus', 'user', 'club'])
  assert.equal(f[1].properties.owner_asserted, true)
  assert.equal(f[1].properties.set_id, 3)
  assert.equal(f[1].properties.set_name, 'Mine')
  assert.equal(f[1].properties.id, 'set:1')
  assert.equal(f[1].properties.collecting, 'allowed')
  assert.equal('owner_asserted' in f[0].properties, false)
  assert.equal(body.sets_included, true)
  assert.equal(body.truncated.sets, false)
  assert.equal(r.headers.get('cache-control'), 'private, no-store')
  assert.equal(c.rpcCalls.find(([n]) => n === 'access_set_areas_in_bbox')[1].p_user, 'u1')
  const o = await call(c, '&include_sets=1', { id: 'u3' })
  assert.deepEqual(o.body.areas.features.map((x) => x.properties.source), ['padus'])
  const fl = await call(c, '&include_sets=1&free=1', { id: 'u1' })
  assert.deepEqual(fl.body.areas.features.map((x) => x.properties.source), ['padus', 'user'])
  const col = await call(c, '&include_sets=1&collecting=likely_allowed', { id: 'u1' })
  assert.deepEqual(col.body.areas.features.map((x) => x.properties.source), ['padus', 'club'])
})

test('access.mjs: set areas capped at 400 with truncated.sets', async () => {
  const c = fake()
  c.padus = []
  c.setRows = { u1: Array.from({ length: 450 }, (_, i) => setRow(i + 1, 'user')) }
  const { body } = await call(c, '&include_sets=1', { id: 'u1' })
  assert.equal(body.areas.features.length, 400)
  assert.equal(body.truncated.sets, true)
})
