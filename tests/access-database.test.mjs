import test from 'node:test'
import assert from 'node:assert/strict'
import {
  applyRidbFees, classifyArea, estimateCollecting, estimateFee, managerType, namesMatch, overpassQuery,
  parsePadUs, parseRidbFacilities, tileBbox,
} from '../netlify/lib/access-ingest.mjs'
import { REGIONS, findCoveringRegion, normaliseAccessSpec, runAccessIngest } from '../netlify/lib/access-regions.mjs'
import { ACCESS_JOBS_PER_MONTH, ACCESS_MAX_AREA_KM2, bboxAreaKm2, checkAccessQuota } from '../netlify/lib/quotas.mjs'
import { handleAccess } from '../netlify/functions/access.mjs'

const sq = (x, y, d = 0.1) => ({ type: 'Polygon', coordinates: [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]] })
const padus = (props, geometry = sq(-105.5, 40)) => ({ type: 'Feature', properties: props, geometry })

test('manager type + estimates are conservative and labelled', () => {
  assert.equal(managerType('BLM'), 'blm')
  assert.equal(managerType('NPS'), 'nps')
  assert.equal(managerType('XYZ', 'STAT'), 'state')
  assert.deepEqual(estimateFee({ managerType: 'blm' }), { fee_status: 'free', fee_source: 'estimated' })
  assert.deepEqual(estimateFee({ managerType: 'nps' }), { fee_status: 'fee', fee_source: 'estimated' })
  assert.deepEqual(estimateFee({ managerType: 'local' }), { fee_status: 'unknown', fee_source: null })
  assert.equal(estimateCollecting({ managerType: 'nps', access: 'open' }).collecting, 'restricted')
  assert.deepEqual(estimateCollecting({ managerType: 'blm', access: 'open' }), { collecting: 'likely_allowed', collecting_source: 'estimated' })
  assert.equal(estimateCollecting({ managerType: 'usfs', access: 'open' }).collecting, 'likely_allowed')
  assert.equal(estimateCollecting({ managerType: 'usfs', designation: 'WSA', access: 'open' }).collecting, 'restricted')
  assert.equal(estimateCollecting({ managerType: 'state', access: 'open' }).collecting, 'unknown')
  assert.equal(estimateCollecting({ managerType: 'blm', access: 'unknown' }).collecting, 'unknown')
  assert.equal(estimateCollecting({ managerType: 'blm', access: 'closed' }).collecting, 'prohibited')
  for (const m of ['blm', 'usfs', 'nps', 'fws', 'state_park', 'state', 'local', 'private', 'other', 'unknown']) {
    for (const a of ['open', 'restricted', 'closed', 'unknown']) {
      assert.notEqual(estimateCollecting({ managerType: m, designation: 'WA', access: a }).collecting, 'allowed')
    }
  }
  assert.equal(classifyArea({ mangName: 'USFS', designation: 'NF', access: 'open' }).manager_type, 'usfs')
})

test('parsePadUs carries classification columns', () => {
  const [r] = parsePadUs({ features: [padus({ OBJECTID: 1, Unit_Nm: 'X NF', Mang_Name: 'USFS', Pub_Access: 'Open' })] })
  assert.equal(r.public_access, 'open'); assert.equal(r.fee_status, 'free'); assert.equal(r.fee_source, 'estimated')
  assert.equal(r.collecting, 'likely_allowed'); assert.equal(r.collecting_source, 'estimated')
})

test('RIDB parse + match upgrades fee source; needs name AND proximity', () => {
  const facilities = parseRidbFacilities({ RECDATA: [
    { FacilityID: 1, FacilityName: 'Pine Ridge Campground', FacilityLatitude: 40.05, FacilityLongitude: -105.45, FacilityUseFeeDescription: '$20 per night' },
    { FacilityID: 2, FacilityName: 'Elsewhere Camp', FacilityLatitude: 40.05, FacilityLongitude: -105.45, FacilityUseFeeDescription: '$5' },
    { FacilityID: 3, FacilityName: 'Pine Ridge Far', FacilityLatitude: 45, FacilityLongitude: -100, FacilityUseFeeDescription: 'Free' },
    { FacilityID: 4, FacilityName: 'No Fee Text', FacilityLatitude: 40.05, FacilityLongitude: -105.45 },
    { FacilityName: 'No coords' },
  ] })
  assert.deepEqual(facilities.map((f) => f.fee), ['fee', 'fee', 'free', null])
  const areas = parsePadUs({ features: [
    padus({ OBJECTID: 1, Unit_Nm: 'Pine Ridge', Mang_Name: 'USFS', Pub_Access: 'Open' }),
    padus({ OBJECTID: 2, Unit_Nm: 'Unrelated Forest', Mang_Name: 'USFS', Pub_Access: 'Open' }, sq(-105.5, 40, 0.2)),
  ] })
  assert.equal(applyRidbFees(areas, facilities).matched, 1)
  assert.deepEqual([areas[0].fee_status, areas[0].fee_source], ['fee', 'ridb'])
  assert.deepEqual([areas[1].fee_status, areas[1].fee_source], ['free', 'estimated'])
  assert.ok(namesMatch('Pine Ridge', 'Pine Ridge Campground'))
  assert.ok(!namesMatch('Pine Ridge', 'Elsewhere Camp'))
})

test('tiling, query, spec, covering region', () => {
  assert.equal(tileBbox([0, 0, 1, 1], 0.25).length, 16)
  assert.equal(tileBbox([0, 0, 1.1, 1], 1).length, 2)
  assert.match(overpassQuery([-105, 40, -104.9, 40.1]), /\(40,-105,40\.1,-104\.9\)/)
  const co = normaliseAccessSpec({ kind: 'access_ingest', region: 'Colorado' })
  assert.deepEqual(co.bbox, REGIONS.colorado.bbox); assert.equal(co.region, 'colorado'); assert.deepEqual(co.sources, ['padus', 'osm', 'ridb'])
  assert.throws(() => normaliseAccessSpec({ region: 'atlantis' }))
  assert.throws(() => normaliseAccessSpec({ bbox: [1, 1, 0, 0] }))
  assert.throws(() => normaliseAccessSpec({ bbox: '0,0,1,1', sources: ['bogus'] }))
  const regions = [{ name: 'colorado', status: 'loaded', bbox: REGIONS.colorado.bbox }, { name: 'x', status: 'failed', bbox: [-180, -90, 180, 90] }]
  assert.equal(findCoveringRegion(regions, [-106, 39, -105, 40]).name, 'colorado')
  assert.equal(findCoveringRegion(regions, [-120, 39, -119, 40]), null)
})

test('quota: area cap, monthly count, admin bypass, shared checkQuota', () => {
  const member = { tier: 'member', member_until: '2099-01-01', ee_quota_monthly: 500 }
  const small = [-105.3, 40, -105.1, 40.1]
  assert.ok(bboxAreaKm2(small) < ACCESS_MAX_AREA_KM2)
  assert.equal(checkAccessQuota({ profile: member, bbox: small }).ok, true)
  assert.equal(checkAccessQuota({ profile: member, bbox: REGIONS.colorado.bbox }).code, 'area_too_large')
  const jobs = Array.from({ length: ACCESS_JOBS_PER_MONTH }, () => ({ kind: 'access_ingest', status: 'succeeded', created_at: new Date().toISOString(), cost_units: 1 }))
  assert.equal(checkAccessQuota({ profile: member, history: jobs, bbox: small }).code, 'access_monthly_limit')
  const running = [{ kind: 'access_ingest', status: 'running', created_at: new Date(Date.now() - 864e5 * 40).toISOString() }, { kind: 'enrich', status: 'running', created_at: new Date().toISOString() }]
  assert.equal(checkAccessQuota({ profile: member, history: running, bbox: small }).code, 'already_running')
  assert.equal(checkAccessQuota({ profile: { tier: 'free' }, bbox: small }).code, 'not_a_member')
  const admin = { tier: 'admin' }
  assert.equal(checkAccessQuota({ profile: admin, bbox: REGIONS.colorado.bbox }).ok, true)
})

function fakeClient({ regions = [], areaRows = [], lineRows = [] } = {}) {
  const saved = [], rpcs = []
  return {
    saved, rpcs,
    rpc: async (fn, args) => {
      rpcs.push([fn, args])
      if (fn === 'access_upsert_areas' || fn === 'access_upsert_lines') return { data: args.p_rows.length, error: null }
      if (fn === 'access_areas_in_bbox') return { data: areaRows, error: null }
      if (fn === 'access_lines_in_bbox') return { data: lineRows, error: null }
      return { data: null, error: { message: 'unknown rpc' } }
    },
    from: (table) => ({
      upsert: async (row) => { saved.push([table, row]); return { error: null } },
      select: () => ({ order: async () => ({ data: regions, error: null }), then: (res) => res({ data: regions, error: null }) }),
    }),
  }
}

const jsonRes = (body, ok = true, status = 200) => ({ ok, status, json: async () => body })

test('runAccessIngest: loads tiles, upserts, degrades without RIDB key, marks loaded', async () => {
  const client = fakeClient()
  const urls = []
  const fetchImpl = async (url) => {
    urls.push(String(url))
    if (String(url).includes('/query?')) return jsonRes({ features: [padus({ OBJECTID: 7, Unit_Nm: 'A', Mang_Name: 'BLM', Pub_Access: 'Open' })] })
    return jsonRes({ elements: [{ type: 'way', id: 5, tags: { highway: 'path' }, geometry: [{ lon: -105.3, lat: 40.05 }, { lon: -105.3, lat: 40.06 }] }] })
  }
  const spec = normaliseAccessSpec({ bbox: [-105.5, 40, -105.25, 40.25], region: 'test' })
  const out = await runAccessIngest({ client, spec, fetchImpl, env: {}, politeMs: 0 })
  assert.equal(out.status, 'loaded'); assert.equal(out.ridb, 'skipped_no_key')
  assert.equal(out.areas, 1); assert.equal(out.lines, 1)
  assert.ok(!urls.some((u) => u.includes('ridb.recreation.gov')))
  assert.equal(client.rpcs[0][1].p_region, 'test')
  const last = client.saved.at(-1)[1]
  assert.equal(last.status, 'loaded'); assert.ok(last.loaded_at)
})

test('runAccessIngest: retries Overpass 429, fails region on persistent error, partial at deadline', async () => {
  const spec = normaliseAccessSpec({ bbox: [-105.5, 40, -105.25, 40.25], region: 'r', sources: ['osm'] })
  let calls = 0
  const flaky = async () => (++calls === 1 ? jsonRes({}, false, 429) : jsonRes({ elements: [] }))
  const ok = await runAccessIngest({ client: fakeClient(), spec, fetchImpl: flaky, env: {}, politeMs: 0, wait: async () => {} })
  assert.equal(ok.status, 'loaded')
  const client = fakeClient()
  await assert.rejects(runAccessIngest({ client, spec, fetchImpl: async () => jsonRes({}, false, 500), env: {}, politeMs: 0, wait: async () => {} }))
  assert.equal(client.saved.at(-1)[1].status, 'failed')
  const part = await runAccessIngest({ client: fakeClient(), spec, fetchImpl: flaky, env: {}, deadlineMs: 0, politeMs: 0 })
  assert.equal(part.status, 'partial')
})

test('runAccessIngest: RIDB failure keeps estimates', async () => {
  const spec = normaliseAccessSpec({ bbox: [-105.5, 40, -105, 40.5], region: 'rb', sources: ['padus', 'ridb'] })
  const fetchImpl = async (url) => (String(url).includes('ridb') ? jsonRes({}, false, 500)
    : jsonRes({ features: [padus({ OBJECTID: 1, Unit_Nm: 'A', Mang_Name: 'NPS', Pub_Access: 'Open' })] }))
  const out = await runAccessIngest({ client: fakeClient(), spec, fetchImpl, env: { RIDB_API_KEY: 'k' }, politeMs: 0 })
  assert.match(out.ridb, /^failed/); assert.equal(out.areas, 1)
})

test('GET access: contract shape, filters, caps, errors', async () => {
  const areaRows = [{ id: 1, name: 'A', manager_type: 'blm', public_access: 'open', fee_status: 'free', fee_source: 'estimated', collecting: 'unknown', collecting_source: null, geometry: sq(-105.5, 40) }]
  const regions = [{ name: 'colorado', status: 'loaded', bbox: REGIONS.colorado.bbox, loaded_at: 'now' }]
  const client = { ...fakeClient({ areaRows, lineRows: [{ id: 9, kind: 'trail', highway: 'path', name: null, geometry: { type: 'LineString', coordinates: [[0, 0], [1, 1]] } }] }) }
  client.from = () => ({ select: async () => ({ data: regions, error: null }) })
  const get = (qs) => handleAccess(new Request(`https://x/.netlify/functions/access?${qs}`), client)

  const res = await get('bbox=-105.6,39.9,-105.2,40.3&free=1&public=1&collecting=unknown&lines=1')
  const body = await res.json()
  assert.equal(res.status, 200); assert.equal(body.ok, true); assert.equal(body.loaded, true)
  assert.deepEqual(Object.keys(body.areas.features[0].properties).sort(),
    ['collecting', 'collecting_source', 'fee_source', 'fee_status', 'id', 'manager_type', 'name', 'public_access', 'source'])
  assert.equal(body.lines.features[0].properties.kind, 'trail')
  assert.deepEqual(body.truncated, { areas: false, lines: false })
  assert.equal(body.regions[0].name, 'colorado')
  const call = client.rpcs.find(([f]) => f === 'access_areas_in_bbox')[1]
  assert.deepEqual([call.p_free, call.p_public, call.p_collecting], [true, true, 'unknown'])

  const wide = await (await get('bbox=-107,39,-105,40&lines=1')).json()
  assert.equal(wide.lines, null); assert.equal(wide.lines_omitted, 'bbox_too_large')
  assert.equal((await (await get('bbox=-105.6,39.9,-105.2,40.3')).json()).lines_omitted, 'not_requested')
  assert.equal((await get('bbox=1,2')).status, 400)
  assert.equal((await get('bbox=-110,36,-100,41')).status, 400)
  assert.equal((await get('bbox=-105.6,39.9,-105.2,40.3&collecting=maybe')).status, 400)
  assert.equal((await handleAccess(new Request('https://x/?bbox=0,0,1,1'), null)).status, 503)

  const many = { ...client, rpc: async () => ({ data: Array.from({ length: 401 }, (_, i) => ({ ...areaRows[0], id: i })), error: null }) }
  const t = await (await handleAccess(new Request('https://x/?bbox=-105.6,39.9,-105.2,40.3'), many)).json()
  assert.equal(t.areas.features.length, 400); assert.equal(t.truncated.areas, true)
})
