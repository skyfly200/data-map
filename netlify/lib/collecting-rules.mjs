// County, city and state collecting rules for the foray planner (WANT-17).
//
// There is no statewide dataset of local collecting rules, so this is a small
// curated table, keyed by managing agency, applied over the manager-type
// estimates at ingest. Each rule cites the agency's own rules page and the date
// it was read. A rule is still not legal advice: rules change, parks post their
// own closures, and a matched polygon may be mislabelled in PAD-US. The UI
// shows the rule's source link next to the value.
//
// Rules only ever set 'restricted' or 'prohibited'. Nothing here makes land
// look more open than the estimate did; plain 'allowed' stays owner-asserted.
//
// Shared by the ingest (netlify/lib/access-ingest.mjs) and the UI
// (composables/forayPlanner.ts, composables/accessLayer.ts) so both name a rule
// the same way.

/**
 * @typedef {object} CollectingRule
 * @property {string} id               stable key stored in access_areas.collecting_rule
 * @property {string} label            the agency, as shown to people
 * @property {string[]} managerTypes   manager types the rule can apply to
 * @property {RegExp|null} match       tested against PAD-US manager + local manager + unit name; null = every area of those types
 * @property {'restricted'|'prohibited'} collecting
 * @property {string} note             what the rule says, in one line
 * @property {string} url              the agency's rules page
 * @property {string} checkedOn        when the page was read (YYYY-MM-DD)
 */

/** @type {CollectingRule[]} Most specific first: the first match wins. */
export const COLLECTING_RULES = [
  {
    id: 'co-boulder-osmp',
    label: 'City of Boulder Open Space and Mountain Parks',
    managerTypes: ['local'],
    match: /\b(city of boulder|open space (and|&) mountain parks|osmp)\b/i,
    collecting: 'prohibited',
    note: 'Collecting or removing any natural object is not permitted, including picking wildflowers and native plants.',
    url: 'https://bouldercolorado.gov/osmp-rules-and-regulations',
    checkedOn: '2026-10-09',
  },
  {
    id: 'co-boulder-county-pos',
    label: 'Boulder County Parks and Open Space',
    managerTypes: ['local'],
    match: /\bboulder county\b/i,
    collecting: 'prohibited',
    note: 'Do not collect, remove, destroy or deface any natural or manmade object.',
    url: 'https://bouldercounty.gov/open-space/parks-and-trails/rules-and-regulations/',
    checkedOn: '2026-10-09',
  },
  {
    id: 'co-jeffco-open-space',
    label: 'Jefferson County Open Space',
    managerTypes: ['local'],
    match: /\b(jefferson county|jeffco)\b/i,
    collecting: 'restricted',
    note: 'Removing natural features is prohibited; scientific collecting needs a Research and Collections permit.',
    url: 'https://www.jeffco.us/1583/Regulations',
    checkedOn: '2026-10-05',
  },
  {
    id: 'co-cpw-state-parks',
    label: 'Colorado state parks (CPW)',
    managerTypes: ['state_park'],
    match: null,
    collecting: 'restricted',
    note: 'State rules bar removing vegetation on park lands; some parks are reported to issue a free permit. Call the park office.',
    url: 'https://www.sos.state.co.us/CCR/GenerateRulePdf.do?ruleVersionId=2387&fileName=2+CCR+405-1',
    checkedOn: '2026-10-05',
  },
]

const BY_ID = new Map(COLLECTING_RULES.map((r) => [r.id, r]))

/** The rule with this id, or null. */
export function collectingRuleById(id) {
  return (id && BY_ID.get(id)) || null
}

/**
 * The first rule that applies to an area, or null.
 * `text` is everything that names the area's manager: PAD-US Mang_Name,
 * Loc_Mang and the unit name, joined.
 */
export function matchCollectingRule({ managerType, text }, rules = COLLECTING_RULES) {
  const t = String(text || '')
  for (const r of rules) {
    if (!r.managerTypes.includes(managerType)) continue
    if (r.match && !r.match.test(t)) continue
    return r
  }
  return null
}
