/* Local collecting rules over the manager-type estimates (WANT-17). */

import test from 'node:test'
import assert from 'node:assert/strict'

import { COLLECTING_RULES, collectingRuleById, matchCollectingRule } from '../netlify/lib/collecting-rules.mjs'
import { classifyArea, parsePadUs } from '../netlify/lib/access-ingest.mjs'

test('every rule cites a page and a date, and never loosens collecting', () => {
  for (const r of COLLECTING_RULES) {
    assert.match(r.url, /^https:\/\//, r.id)
    assert.match(r.checkedOn, /^\d{4}-\d{2}-\d{2}$/, r.id)
    assert.ok(['restricted', 'prohibited'].includes(r.collecting), r.id)
    assert.equal(collectingRuleById(r.id), r)
  }
  assert.equal(new Set(COLLECTING_RULES.map((r) => r.id)).size, COLLECTING_RULES.length)
})

test('a rule matches only its manager type and agency', () => {
  assert.equal(matchCollectingRule({ managerType: 'local', text: 'CITY | City of Boulder | Flagstaff Mountain' }).id, 'co-boulder-osmp')
  assert.equal(matchCollectingRule({ managerType: 'local', text: 'CNTY | Boulder County | Walker Ranch' }).id, 'co-boulder-county-pos')
  assert.equal(matchCollectingRule({ managerType: 'local', text: 'CNTY | Jefferson County Open Space | Matthews/Winters' }).id, 'co-jeffco-open-space')
  // A USFS unit that happens to carry an agency's name is not that agency's land.
  assert.equal(matchCollectingRule({ managerType: 'usfs', text: 'USFS | Boulder Ranger District' }), null)
  assert.equal(matchCollectingRule({ managerType: 'local', text: 'CITY | City of Golden | Lookout' }), null)
  assert.equal(matchCollectingRule({ managerType: 'state_park', text: 'SPR | Golden Gate Canyon' }).id, 'co-cpw-state-parks')
})

test('classifyArea puts a matched rule over the estimate, with its id', () => {
  const area = classifyArea({ mangName: 'CNTY', mangType: 'LOC', designation: 'LP', access: 'open', locMang: 'Boulder County', unitName: 'Hall Ranch' })
  assert.deepEqual(
    { c: area.collecting, s: area.collecting_source, r: area.collecting_rule },
    { c: 'prohibited', s: 'rule', r: 'co-boulder-county-pos' },
  )
  const plain = classifyArea({ mangName: 'CITY', mangType: 'LOC', designation: 'LP', access: 'open', locMang: 'City of Golden' })
  assert.deepEqual({ c: plain.collecting, s: plain.collecting_source, r: plain.collecting_rule }, { c: 'unknown', s: null, r: null })
})

test('closed land stays prohibited by estimate even where a rule matches', () => {
  const area = classifyArea({ mangName: 'CNTY', mangType: 'LOC', designation: 'LP', access: 'closed', locMang: 'Jefferson County Open Space' })
  assert.deepEqual({ c: area.collecting, s: area.collecting_source, r: area.collecting_rule }, { c: 'prohibited', s: 'estimated', r: null })
})

test('parsePadUs reads Loc_Mang for the rule match', () => {
  const [row] = parsePadUs({ features: [{
    type: 'Feature',
    geometry: { type: 'Polygon', coordinates: [[[-105.3, 40], [-105.2, 40], [-105.2, 40.1], [-105.3, 40]]] },
    properties: { OBJECTID: 7, Unit_Nm: 'Chautauqua', Mang_Name: 'CITY', Mang_Type: 'LOC', Loc_Mang: 'City of Boulder', Des_Tp: 'LP', Pub_Access: 'Open' },
  }] })
  assert.equal(row.collecting_rule, 'co-boulder-osmp')
  assert.equal(row.collecting, 'prohibited')
})

test('the UI keeps a known rule id, drops an unknown one, and links the rule in the popup', async () => {
  const { normaliseArea, collectingLabel, accessForCell } = await import('../composables/forayPlanner.ts')
  const { popupHtml } = await import('../composables/accessLayer.ts')
  const geometry = { type: 'Polygon', coordinates: [[[-105.3, 40], [-105.2, 40], [-105.2, 40.1], [-105.3, 40.1], [-105.3, 40]]] }
  const props = { id: 1, name: 'Hall Ranch', manager_type: 'local', public_access: 'open', fee_status: 'unknown', collecting: 'prohibited', collecting_source: 'rule' }
  const area = normaliseArea({ type: 'Feature', geometry, properties: { ...props, collecting_rule: 'co-boulder-county-pos' } })
  assert.equal(area.collecting_rule, 'co-boulder-county-pos')
  assert.equal(normaliseArea({ type: 'Feature', geometry, properties: { ...props, collecting_rule: 'made-up' } }).collecting_rule, null)
  const cell = accessForCell([-105.25, 40.05], [area])
  assert.equal(collectingLabel(cell), 'prohibited (local rule)')
  const html = popupHtml(area)
  assert.match(html, /Boulder County Parks and Open Space/)
  assert.match(html, /href="https:\/\/bouldercounty\.gov\/open-space\/parks-and-trails\/rules-and-regulations\/"/)
})
