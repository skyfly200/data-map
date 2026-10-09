import test from 'node:test'
import assert from 'node:assert/strict'
import { fetchedPath, isFetchedPath, isNarrowedFetch, registerAndEnrich } from '../netlify/lib/species-fetch.mjs'

test('an unnarrowed fetch writes the shared species file', () => {
  assert.equal(isNarrowedFetch(new URLSearchParams('species=Morchella')), false)
  assert.equal(fetchedPath({ slug: 'morchella', userId: 'u1', narrowed: false }), 'species/morchella.geojson')
})

test('a fetch narrowed by place or date gets its own file under the caller', () => {
  for (const q of ['d1=2024-01-01', 'd2=2024-12-31', 'lat=40&lng=-105&radius=10']) {
    assert.equal(isNarrowedFetch(new URLSearchParams(`species=Morchella&${q}`)), true, q)
  }
  assert.equal(fetchedPath({ slug: 'morchella', userId: 'u1', narrowed: true, now: 5 }),
    'species/u1/morchella-5.geojson')
})

test('only clean species/ paths can be registered', () => {
  assert.equal(isFetchedPath('species/morchella.geojson'), true)
  assert.equal(isFetchedPath('species/u1/morchella-5.geojson'), true)
  assert.equal(isFetchedPath('datasets/u1/x.geojson'), false)
  assert.equal(isFetchedPath('species/../datasets/u2/x.geojson'), false)
  assert.equal(isFetchedPath('species//x.geojson'), false)
})

// A minimal saved_datasets stand-in: enough of the query builder for registerFetched.
function fakeClient(rows = []) {
  return {
    rows,
    from() {
      const self = this
      let filters = []
      let op = 'select'
      let payload = null
      const q = {
        select() { return q },
        eq(k, v) { filters.push((r) => r[k] === v); return q },
        like(k, v) { const p = v.replace(/%$/, ''); filters.push((r) => String(r[k]).startsWith(p)); return q },
        insert(row) { op = 'insert'; payload = row; return q },
        update(patch) { op = 'update'; payload = patch; return q },
        maybeSingle() { return Promise.resolve({ data: self.rows.find((r) => filters.every((f) => f(r))) || null }) },
        single() {
          if (op === 'insert') { const row = { id: `d${self.rows.length + 1}`, ...payload }; self.rows.push(row); return Promise.resolve({ data: row }) }
          const row = self.rows.find((r) => filters.every((f) => f(r))); Object.assign(row, payload); return Promise.resolve({ data: row })
        },
        then(res) { return Promise.resolve({ data: self.rows.filter((r) => filters.every((f) => f(r))), count: self.rows.filter((r) => filters.every((f) => f(r))).length, error: null }).then(res) },
      }
      return q
    },
  }
}

test('auto-enrich queues the registered dataset, not the species file name', async () => {
  // Someone else already owns a dataset whose slug is the species slug.
  const client = fakeClient([{ id: 'x', owner_id: 'other', slug: 'morchella', path: 'datasets/other/m.geojson', visibility: 'public' }])
  let submitted = null
  const res = await registerAndEnrich({
    auth: { user: { id: 'u1' }, tier: 'member' },
    profile: { tier: 'member' },
    client,
    storagePath: 'species/u1/morchella-5.geojson',
    slug: 'morchella',
    title: 'Morchella (iNat, 3)',
    count: 3,
    submitJob: async ({ spec }) => { submitted = spec; return { job: { id: 'j1' } } },
    measureSource: async () => ({ points: 3, dates: 1 }),
    viewerFrom: (a) => ({ userId: a.user.id, tier: a.tier }),
  })
  assert.equal(res.dataset.owner_id, 'u1')
  assert.equal(res.dataset.path, 'species/u1/morchella-5.geojson')
  assert.notEqual(submitted.source.slug, 'morchella')
  assert.equal(submitted.source.slug, res.dataset.slug)
})

test('free accounts are not registered or enriched', async () => {
  const res = await registerAndEnrich({
    auth: { user: { id: 'u1' }, tier: 'free' }, profile: { tier: 'free' }, client: fakeClient(),
    storagePath: 'species/morchella.geojson', slug: 'morchella', title: 't', count: 1,
    submitJob: () => { throw new Error('should not submit') }, measureSource: () => {}, viewerFrom: () => ({}),
  })
  assert.equal(res, null)
})
