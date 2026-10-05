import test from 'node:test'
import assert from 'node:assert/strict'
import {
  parseAccessResponse, normaliseArea, accessForCell, passesSwitches, effectiveSwitches, attachAccess,
  filterByAccess, buildShortlist, shortlistToCsv, shortlistToText, FORAY_MODES, NO_SWITCHES,
  bboxInColorado, DISCLAIMER, UNKNOWN_ACCESS,
} from '../composables/forayPlanner.ts'
import { fetchAccess } from '../composables/useAccess.ts'

const sq = (w, s, e, n) => ({ type: 'Polygon', coordinates: [[[w, s], [e, s], [e, n], [w, n], [w, s]]] })
const area = (id, box, props) => ({
  id, name: `Area ${id}`, manager_type: 'federal', public_access: 'open', fee_status: 'free', fee_source: 'ridb',
  collecting: 'allowed', collecting_source: 'estimated', geom: sq(...box), ...props,
})

const response = {
  region: { loaded: true, name: 'Colorado' },
  areas: [
    area('1', [-105.4, 40.0, -105.2, 40.2]),
    area('2', [-105.2, 40.0, -105.0, 40.2], { fee_status: 'fee', fee_source: 'estimated', collecting: 'restricted' }),
    area('3', [-105.0, 40.0, -104.8, 40.2], { public_access: 'unknown', fee_status: 'unknown', fee_source: null, collecting: 'unknown', collecting_source: null }),
    area('4', [-104.8, 40.0, -104.6, 40.2], { manager_type: 'private', public_access: 'open' }),
  ],
}
const parsed = parseAccessResponse(response)

test('parse: loaded, flat rows and GeoJSON features both accepted', () => {
  assert.equal(parsed.status, 'loaded')
  assert.equal(parsed.areas.length, 4)
  assert.equal(parsed.areas[0].geom.type, 'MultiPolygon')
  const feat = normaliseArea({ type: 'Feature', geometry: sq(0, 0, 1, 1), properties: { id: 9, public_access: 'weird' } })
  assert.equal(feat.public_access, 'unknown')
  assert.equal(normaliseArea({ id: 1 }), null)
})

test('parse: missing, empty and not-loaded never read as loaded', () => {
  assert.equal(parseAccessResponse(null).status, 'unavailable')
  assert.equal(parseAccessResponse('x').status, 'unavailable')
  assert.equal(parseAccessResponse({ region: { loaded: false }, areas: [response.areas[0]] }).status, 'not-loaded')
  assert.equal(parseAccessResponse({ areas: [] }).status, 'not-loaded')
  assert.equal(parseAccessResponse({ regionLoaded: true, areas: [] }).status, 'no-data')
})

test('fetchAccess: fallbacks for 404, network error, non-Colorado bbox', async () => {
  const co = [-105.3, 40.0, -105.1, 40.2]
  assert.equal((await fetchAccess(co, async () => ({ ok: false }))).status, 'unavailable')
  assert.equal((await fetchAccess(co, async () => { throw new Error('offline') })).status, 'unavailable')
  assert.equal((await fetchAccess(co, async () => ({ ok: true, json: async () => { throw new Error('html') } }))).status, 'unavailable')
  assert.equal((await fetchAccess([-120, 35, -119, 36], async () => assert.fail('no request'))).status, 'not-loaded')
  let url
  const ok = await fetchAccess(co, async (u) => { url = u; return { ok: true, json: async () => response } })
  assert.equal(ok.status, 'loaded')
  assert.match(url, /bbox=-105\.3000,40\.0000,-105\.1000,40\.2000/)
  assert.equal(bboxInColorado([-100, 40, -99, 41]), false)
})

test('cell access: containment, no coverage is all unknown, overlap takes most restrictive', () => {
  assert.equal(accessForCell([-105.3, 40.1], parsed.areas).fee_status, 'free')
  assert.deepEqual(accessForCell([-100, 30], parsed.areas), UNKNOWN_ACCESS)
  const overlap = parseAccessResponse({ regionLoaded: true, areas: [area('a', [0, 0, 2, 2]), area('b', [1, 1, 3, 3], { fee_status: 'fee', collecting: 'prohibited', public_access: 'restricted' })] }).areas
  const a = accessForCell([1.5, 1.5], overlap)
  assert.equal(a.fee_status, 'fee')
  assert.equal(a.collecting, 'prohibited')
  assert.equal(a.public_access, 'restricted')
})

const A = (p) => ({ ...UNKNOWN_ACCESS, ...p })

test('switches: each filters on its own attribute; unknown excluded by default', () => {
  const free = { ...NO_SWITCHES, free: true }
  assert.equal(passesSwitches(A({ fee_status: 'free' }), free), true)
  assert.equal(passesSwitches(A({ fee_status: 'fee' }), free), false)
  assert.equal(passesSwitches(A({ fee_status: 'unknown' }), free), false)
  assert.equal(passesSwitches(A({ fee_status: 'unknown' }), { ...free, includeUnknown: true }), true)
  const pub = { ...NO_SWITCHES, public: true }
  assert.equal(passesSwitches(A({ public_access: 'open' }), pub), true)
  assert.equal(passesSwitches(A({ public_access: 'restricted' }), pub), false)
  assert.equal(passesSwitches(A({ public_access: 'open', manager_type: 'private' }), pub), false)
  const col = { ...NO_SWITCHES, collecting: true }
  assert.equal(passesSwitches(A({ collecting: 'allowed' }), col), true)
  assert.equal(passesSwitches(A({ collecting: 'restricted' }), col), false)
  assert.equal(passesSwitches(A({ collecting: 'unknown' }), col), false)
  assert.equal(passesSwitches(A({ collecting: 'likely_allowed' }), col), false)
  assert.equal(passesSwitches(A({ collecting: 'likely_allowed' }), { ...col, includeLikely: true }), true)
  assert.equal(passesSwitches(A({}), NO_SWITCHES), true)
})

test('effective switches are off unless access is loaded', () => {
  const all = { free: true, public: true, collecting: true, includeUnknown: false }
  assert.deepEqual(effectiveSwitches(all, 'loaded'), all)
  for (const s of ['idle', 'loading', 'not-loaded', 'no-data', 'unavailable']) {
    assert.deepEqual(effectiveSwitches(all, s), NO_SWITCHES)
  }
})

const cells = [
  { key: 'c1', lon: -105.3, lat: 40.1, score: 0.9, n: 10 },
  { key: 'c2', lon: -105.1, lat: 40.1, score: 0.8, n: 6 },
  { key: 'c3', lon: -104.9, lat: 40.1, score: 0.7, n: 5 },
  { key: 'c4', lon: -104.7, lat: 40.1, score: 0.6, n: 4 },
  { key: 'c5', lon: -100, lat: 39, score: 0.5, n: 3 },
]

test('filterByAccess over cells, combined switches', () => {
  const withAccess = attachAccess(cells, parsed.areas)
  const keys = (sw) => filterByAccess(withAccess, sw).map((c) => c.key)
  assert.deepEqual(keys(NO_SWITCHES), ['c1', 'c2', 'c3', 'c4', 'c5'])
  assert.deepEqual(keys({ ...NO_SWITCHES, free: true }), ['c1', 'c4'])
  assert.deepEqual(keys({ ...NO_SWITCHES, public: true }), ['c1', 'c2'])
  assert.deepEqual(keys({ ...NO_SWITCHES, collecting: true }), ['c1', 'c4'])
  assert.deepEqual(keys({ free: true, public: true, collecting: true, includeUnknown: false }), ['c1'])
  assert.deepEqual(keys({ free: true, public: true, collecting: true, includeUnknown: true }), ['c1', 'c3', 'c5'])
})

test('mode defaults: leader is strict, others open; score is not part of modes', () => {
  assert.deepEqual(FORAY_MODES.forager.switches, NO_SWITCHES)
  assert.deepEqual(FORAY_MODES.researcher.switches, NO_SWITCHES)
  const l = FORAY_MODES.leader.switches
  assert.ok(l.free && l.public && l.collecting && !l.includeUnknown)
  assert.ok(FORAY_MODES.leader.showShortlist && !FORAY_MODES.forager.showShortlist)
  assert.ok(FORAY_MODES.researcher.showComponents && FORAY_MODES.researcher.showCaveats)
})

const ranked = attachAccess(cells.slice(0, 3).map((c) => ({ ...c, components: [{ species: 'Boletus edulis' }, { species: 'Cantharellus' }] })), parsed.areas)

test('shortlist rows: access class, fee and collecting with source labels, notes', () => {
  const rows = buildShortlist(ranked, { c2: 'Park at the north lot' }, 2)
  assert.equal(rows.length, 2)
  assert.equal(rows[0].site, 'Area 1')
  assert.equal(rows[0].access_class, 'open')
  assert.equal(rows[0].fee_status, 'free (verified)')
  assert.equal(rows[0].collecting_status, 'allowed (estimated)')
  assert.equal(rows[1].fee_status, 'fee (estimated)')
  assert.equal(rows[1].collecting_status, 'restricted (estimated)')
  assert.equal(rows[1].notes, 'Park at the north lot')
  assert.equal(rows[0].top_species, 'Boletus edulis; Cantharellus')
  assert.equal(buildShortlist(ranked)[2].fee_status, 'unknown')
})

test('shortlist CSV: header, quoting, disclaimer row', () => {
  const rows = buildShortlist(ranked, { c1: 'Bring "boots", maps\nand water' })
  const csv = shortlistToCsv(rows)
  const lines = csv.split('\n')
  assert.equal(lines[0], 'rank,site,lat,lon,score,access_class,manager,fee_status,collecting_status,finds,top_species,notes')
  assert.ok(csv.includes('"Bring ""boots"", maps\nand water"'))
  assert.ok(csv.includes(DISCLAIMER))
  assert.ok(lines[1].startsWith('1,Area 1,40.1,-105.3,0.9,open,federal,free (verified),allowed (estimated),10,'))
})

test('shortlist text and empty export', () => {
  const txt = shortlistToText(buildShortlist(ranked, { c1: 'Meet 8am' }))
  assert.match(txt, /1\. Area 1/)
  assert.match(txt, /notes: Meet 8am/)
  assert.match(txt, /regulations/)
  assert.equal(shortlistToCsv([]).split('\n').length, 3)
})

test('unnamed unknown site falls back to coordinates', () => {
  const [r] = buildShortlist(attachAccess([{ key: 'z', lon: -100.1234, lat: 39.5, score: 0.3, n: 3 }], parsed.areas))
  assert.equal(r.site, '39.500, -100.123')
  assert.equal(r.fee_status, 'unknown')
})
