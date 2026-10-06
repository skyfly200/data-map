import test from 'node:test'
import assert from 'node:assert/strict'
import { normaliseArea, NO_SWITCHES } from '../composables/forayPlanner.ts'
import {
  styleForArea, classify, legendFor, filterAreas, popupHtml, areaTag, planView, accessNotice,
  UNKNOWN_COLOR, ESTIMATED_DASH, SOURCE_OUTLINE, DEFAULT_SOURCES, sourceCounts,
} from '../composables/accessLayer.ts'
import { fetchAccess, clearAccessCache } from '../composables/useAccess.ts'

const poly = { type: 'Polygon', coordinates: [[[-105, 40], [-104.9, 40], [-104.9, 40.1], [-105, 40.1], [-105, 40]]] }
const mk = (p) => normaliseArea({ type: 'Feature', geometry: poly, properties: { id: 'a', name: 'A', manager_type: 'BLM', public_access: 'open', fee_status: 'free', fee_source: 'ridb', collecting: 'allowed', collecting_source: null, ...p } })

test('estimated values are dashed, verified solid, unknown neutral', () => {
  assert.equal(styleForArea(mk({ fee_source: 'estimated' }), 'fee').dashArray, ESTIMATED_DASH)
  assert.equal(styleForArea(mk({}), 'fee').dashArray, null)
  const u = styleForArea(mk({ fee_status: 'unknown', fee_source: null }), 'fee')
  assert.equal(u.fillColor, UNKNOWN_COLOR); assert.equal(u.dashArray, null)
  const l = styleForArea(mk({ collecting: 'likely_allowed', collecting_source: 'estimated' }), 'collecting')
  assert.equal(l.dashArray, ESTIMATED_DASH)
})

test('unknown never takes a good colour on any attribute', () => {
  const a = mk({ public_access: 'unknown', fee_status: 'unknown', collecting: 'unknown' })
  for (const attr of ['public', 'fee', 'collecting']) assert.equal(classify(a, attr).color, UNKNOWN_COLOR)
})

test('user/club areas: distinct outline, owner-asserted not estimated; absent fields = PAD-US', () => {
  const u = mk({ source: 'user', fee_source: 'estimated', owner_asserted: true })
  const s = styleForArea(u, 'fee')
  assert.equal(s.color, SOURCE_OUTLINE.user); assert.equal(s.dashArray, null); assert.equal(s.weight, 3)
  assert.equal(areaTag(u), 'Your area')
  assert.equal(areaTag(mk({ source: 'club', set_name: 'Front Range Club' })), 'Club area: Front Range Club')
  assert.equal(areaTag(mk({})), null)
  assert.match(popupHtml(u), /owner-asserted/)
  assert.doesNotMatch(popupHtml(u), /check regulations/i)
})

test('filters: switches, likely, sources', () => {
  const free = mk({ id: '1' }), fee = mk({ id: '2', fee_status: 'fee' })
  const unk = mk({ id: '3', fee_status: 'unknown', collecting: 'unknown' })
  const likely = mk({ id: '4', collecting: 'likely_allowed', collecting_source: 'estimated' })
  const mine = mk({ id: '5', source: 'user' })
  const all = [free, fee, unk, likely, mine]
  assert.equal(filterAreas(all, NO_SWITCHES).length, 5)
  assert.deepEqual(filterAreas(all, { ...NO_SWITCHES, free: true }).map((a) => a.id), ['1', '4', '5'])
  assert.deepEqual(filterAreas(all, { ...NO_SWITCHES, collecting: true }).map((a) => a.id), ['1', '2', '5'])
  assert.deepEqual(filterAreas(all, { ...NO_SWITCHES, collecting: true, includeLikely: true }).map((a) => a.id), ['1', '2', '4', '5'])
  assert.deepEqual(filterAreas(all, NO_SWITCHES, { ...DEFAULT_SOURCES, user: false }).map((a) => a.id), ['1', '2', '3', '4'])
  assert.deepEqual(sourceCounts(all), { padus: 4, user: 1, club: 0 })
})

test('popup: likely wording, disclaimer, escaping', () => {
  const h = popupHtml(mk({ name: '<b>x</b>', collecting: 'likely_allowed', collecting_source: 'estimated', fee_source: 'estimated' }))
  assert.match(h, /Likely allowed \(estimated\)/); assert.match(h, /estimate/); assert.doesNotMatch(h, /<b>x/)
  assert.match(popupHtml(mk({ fee_status: 'unknown', fee_source: null })), /Fee<\/dt><dd>Unknown/)
})

test('legend', () => {
  assert.ok(legendFor('fee').some((i) => /not assumed free/.test(i.label)))
  assert.ok(legendFor('collecting').some((i) => i.dashed))
  assert.ok(legendFor('public').every((i) => !/estimated/i.test(i.label)))
})

test('planView: zoom floor, Colorado, lines cap, tile cap', () => {
  assert.equal(planView([-105.2, 39.9, -105, 40], 7).kind, 'zoom-in')
  assert.equal(planView([-80, 30, -79, 31], 10).kind, 'outside')
  const p = planView([-105.1, 39.9, -104.95, 40.0], 13)
  assert.equal(p.kind, 'ok'); assert.equal(p.lines, true)
  const wide = planView([-107, 38, -104, 40], 9)
  assert.equal(wide.kind, 'ok'); assert.equal(wide.lines, false)
  assert.equal(planView([-109, 37, -102, 41], 8).kind, 'ok')
  assert.match(accessNotice({ kind: 'ok' }, 'partial'), /incomplete/)
  assert.match(accessNotice({ kind: 'ok' }, 'not-loaded'), /not loaded/)
  assert.equal(accessNotice({ kind: 'ok' }, 'loaded'), '')
})

test('fetchAccess: include_sets + bearer only with a token; lines on request', async () => {
  clearAccessCache()
  const calls = []
  const body = {
    ok: true, loaded: true,
    areas: { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: poly, properties: { id: 'q', source: 'club', set_id: 's', set_name: 'N', owner_asserted: true } }] },
    lines: { type: 'FeatureCollection', features: [{ type: 'Feature', geometry: { type: 'LineString', coordinates: [[-105, 40], [-104.9, 40]] }, properties: { id: 'l', kind: 'trail' } }] },
    truncated: { areas: false, lines: false },
  }
  const f = async (u, init) => { calls.push([u, init]); return { ok: true, json: async () => body } }
  const anon = await fetchAccess([-105.1, 39.9, -104.9, 40.1], f)
  assert.doesNotMatch(calls[0][0], /include_sets/); assert.equal(calls[0][1], undefined)
  const authed = await fetchAccess([-105.1, 39.9, -104.9, 40.1], f, { token: 'tok-abcdefghijklmnop', lines: true })
  assert.match(calls[1][0], /include_sets=1/); assert.match(calls[1][0], /lines=1/)
  assert.equal(calls[1][1].headers.Authorization, 'Bearer tok-abcdefghijklmnop')
  assert.equal(authed.areas[0].source, 'club'); assert.equal(authed.areas[0].set_name, 'N')
  assert.equal(authed.lines.length, 1); assert.equal(anon.status, 'loaded')
})
