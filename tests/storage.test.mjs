import { test } from 'node:test'
import assert from 'node:assert/strict'
import { supabaseStore, backendName } from '../netlify/lib/storage.mjs'

function fakeClient() {
  const objs = new Map()
  const api = {
    async upload(p, body, opts) {
      if (!opts.upsert && objs.has(p)) return { error: { status: 409, message: 'exists' } }
      objs.set(p, { body, contentType: opts.contentType })
      return { error: null }
    },
    async download(p) {
      if (!objs.has(p)) return { data: null, error: { status: 400, message: 'Object not found' } }
      return { data: new Blob([objs.get(p).body]), error: null }
    },
    async remove(paths) { paths.forEach((p) => objs.delete(p)); return { error: null } },
    async list(dir, { limit, offset, search }) {
      const names = [...objs.keys()]
        .filter((k) => k.startsWith(dir + '/') && !k.slice(dir.length + 1).includes('/'))
        .map((k) => k.slice(dir.length + 1))
        .filter((n) => !search || n.includes(search))
        .sort()
      return { data: names.slice(offset, offset + limit).map((name) => ({ name, id: name })), error: null }
    },
  }
  return { objs, storage: { from: () => api } }
}

test('setJSON/get json round-trips with json content type, under <store>/<key>', async () => {
  const c = fakeClient(); const s = supabaseStore('ee-tiles', c)
  await s.setJSON('a', { x: 1 })
  assert.equal(c.objs.get('ee-tiles/a').contentType, 'application/json')
  assert.deepEqual(await s.get('a', { type: 'json' }), { x: 1 })
  assert.equal(await s.get('a'), '{"x":1}')
})

test('set overwrites (upsert) and stores binary', async () => {
  const c = fakeClient(); const s = supabaseStore('m', c)
  await s.set('k', 'one'); await s.set('k', 'two')
  assert.equal(await s.get('k'), 'two')
  await s.set('b', new Uint8Array([1, 2, 3]))
  assert.equal(c.objs.get('m/b').contentType, 'application/octet-stream')
  assert.deepEqual([...new Uint8Array(await s.get('b', { type: 'arrayBuffer' }))], [1, 2, 3])
})

test('missing key -> null; other errors throw', async () => {
  const s = supabaseStore('m', fakeClient())
  assert.equal(await s.get('nope', { type: 'json' }), null)
  const bad = { storage: { from: () => ({ download: async () => ({ data: null, error: { status: 500, message: 'boom' } }) }) } }
  await assert.rejects(supabaseStore('m', bad).get('k'))
})

test('delete removes; deleting a missing key is fine', async () => {
  const s = supabaseStore('m', fakeClient())
  await s.set('k', 'v'); await s.delete('k'); await s.delete('k')
  assert.equal(await s.get('k'), null)
})

test('list paginates, scopes to store, honors prefix', async () => {
  const c = fakeClient(); const s = supabaseStore('m', c)
  for (let i = 0; i < 250; i++) await s.set('k' + String(i).padStart(3, '0'), 'v')
  await s.set('other', 'v')
  await supabaseStore('n', c).set('k999', 'v')
  assert.equal((await s.list()).blobs.length, 251)
  assert.equal((await s.list({ prefix: 'k' })).blobs.length, 250)
})

test('backend selection', () => {
  assert.equal(backendName({ NETLIFY: 'true' }), 'netlify')
  assert.equal(backendName({}), 'supabase')
  assert.equal(backendName({ NETLIFY: 'true', STORAGE_BACKEND: 'supabase' }), 'supabase')
  assert.equal(backendName({ STORAGE_BACKEND: 'netlify' }), 'netlify')
})
