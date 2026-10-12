// Foray planner logic that is not scoring (WANT-17 phase 1): access contract
// parsing, per-cell access classification, the three switches, mode defaults and
// the shortlist export. Pure, no framework imports, so tests run on fixtures.
//
// The access endpoint (netlify/functions/access.mjs) is built separately. The
// contract assumed here: GET ?bbox=w,s,e,n returns area polygons with
// {id, name, manager_type, public_access, fee_status, fee_source, collecting,
// collecting_source, collecting_rule} and a region-loaded flag. Nothing here ever treats an
// unknown value as free, public or collectable.

import { pointInArea } from '../netlify/lib/access-ingest.mjs'
import { collectingRuleById } from '../netlify/lib/collecting-rules.mjs'

export type PublicAccess = 'open' | 'restricted' | 'closed' | 'unknown'
export type FeeStatus = 'free' | 'fee' | 'unknown'
export type Collecting = 'allowed' | 'likely_allowed' | 'restricted' | 'prohibited' | 'unknown'
export type CollectingSource = 'estimated' | 'rule' | null

export interface AccessArea {
  id: string
  name: string | null
  manager_type: string | null
  public_access: PublicAccess
  fee_status: FeeStatus
  fee_source: 'estimated' | 'ridb' | null
  collecting: Collecting
  collecting_source: CollectingSource
  /** Id of the local rule behind a 'rule' value (netlify/lib/collecting-rules.mjs). */
  collecting_rule: string | null
  /** Optional (backend may not send them yet): where the area came from. Absent = PAD-US. */
  source?: 'padus' | 'user' | 'club'
  set_id?: string | null
  set_name?: string | null
  owner_asserted?: boolean
  geom: { type: 'MultiPolygon', coordinates: number[][][][] }
}

/** Colorado is the only loaded region for now: [west, south, east, north]. */
export const COLORADO_BBOX: [number, number, number, number] = [-109.06, 36.99, -102.04, 41.0]

export function bboxInColorado([w, s, e, n]: number[]): boolean {
  const [cw, cs, ce, cn] = COLORADO_BBOX
  const cx = (w + e) / 2, cy = (s + n) / 2
  return cx >= cw && cx <= ce && cy >= cs && cy <= cn
}

export const LIKELY_LABEL = 'Likely allowed (estimated)'
export const LIKELY_DISCLAIMER = 'Likely allowed is an estimate for BLM and US Forest Service land only. Regulations and permits (quantity limits, permits, closures) must be checked with the local district office.'
export const DISCLAIMER = 'Fee and collecting information is an estimate. Check regulations and permits with the land manager before you go.'

const PUBLIC_ACCESS = ['open', 'restricted', 'closed', 'unknown']
const FEE = ['free', 'fee', 'unknown']
const COLLECTING = ['allowed', 'likely_allowed', 'restricted', 'prohibited', 'unknown']
const oneOf = <T extends string>(v: any, set: string[], fallback: T): T => (set.includes(v) ? v : fallback)

function toMulti(g: any) {
  if (g?.type === 'Polygon') return { type: 'MultiPolygon' as const, coordinates: [g.coordinates] }
  if (g?.type === 'MultiPolygon') return g
  return null
}

/** One raw endpoint row (flat or GeoJSON Feature) -> AccessArea, or null without a usable polygon. */
export function normaliseArea(raw: any): AccessArea | null {
  if (!raw) return null
  const p = raw.properties ? { ...raw.properties, ...raw } : raw
  const geom = toMulti(raw.geom ?? raw.geometry ?? raw.polygon)
  if (!geom) return null
  return {
    id: String(p.id ?? p.source_id ?? ''),
    name: p.name ?? null,
    manager_type: p.manager_type ?? null,
    public_access: oneOf(p.public_access, PUBLIC_ACCESS, 'unknown'),
    fee_status: oneOf(p.fee_status, FEE, 'unknown'),
    fee_source: p.fee_source === 'ridb' || p.fee_source === 'estimated' ? p.fee_source : null,
    collecting: oneOf(p.collecting, COLLECTING, 'unknown'),
    collecting_source: p.collecting_source === 'estimated' || p.collecting_source === 'rule' ? p.collecting_source : null,
    collecting_rule: p.collecting_source === 'rule' && collectingRuleById(p.collecting_rule) ? String(p.collecting_rule) : null,
    ...(p.source === 'user' || p.source === 'club' || p.source === 'padus' ? { source: p.source } : {}),
    ...(p.set_id != null ? { set_id: String(p.set_id) } : {}),
    ...(p.set_name != null ? { set_name: String(p.set_name) } : {}),
    ...(typeof p.owner_asserted === 'boolean' ? { owner_asserted: p.owner_asserted } : {}),
    geom,
  }
}

export type AccessStatus = 'idle' | 'loading' | 'loaded' | 'partial' | 'not-loaded' | 'no-data' | 'unavailable'

export interface ParsedAccess { status: AccessStatus, areas: AccessArea[], region: string | null, truncated?: boolean }

/**
 * Endpoint JSON -> status + areas. Accepts {areas|features, region:{loaded,name}}
 * or {regionLoaded|region_loaded}; a bare array or FeatureCollection counts as
 * loaded only if it has rows. A missing flag is not assumed to mean loaded.
 */
export function parseAccessResponse(json: any): ParsedAccess {
  if (!json || typeof json !== 'object') return { status: 'unavailable', areas: [], region: null }
  if (json.ok === false) return { status: 'unavailable', areas: [], region: null }
  // Endpoint contract: `areas` is a GeoJSON FeatureCollection; plain arrays also accepted.
  const src = Array.isArray(json) ? json : (json.areas ?? json.features ?? [])
  const rawAreas = Array.isArray(src) ? src : (Array.isArray(src?.features) ? src.features : [])
  const truncated = json.truncated?.areas === true
  const areas = (Array.isArray(rawAreas) ? rawAreas : []).map(normaliseArea).filter(Boolean) as AccessArea[]
  const flag = json.region?.loaded ?? json.regionLoaded ?? json.region_loaded ?? json.loaded
  const region = json.region?.name ?? (typeof json.region === 'string' ? json.region : null)
  if (flag === false) return { status: 'not-loaded', areas: [], region }
  if (flag === undefined && !areas.length) return { status: 'not-loaded', areas: [], region }
  if (!areas.length) return { status: 'no-data', areas: [], region, truncated }
  return { status: 'loaded', areas, region, truncated }
}

export const ACCESS_STATUS_MESSAGE: Record<AccessStatus, string> = {
  idle: '',
  loading: 'Loading access data…',
  loaded: '',
  partial: 'Access data incomplete here: some places have unknown access, so Free, public-land and collecting filters are off.',
  'not-loaded': 'Access data not loaded for this area. Free, public-land and collecting filters are off.',
  'no-data': 'Access data not loaded for this area (no land areas returned). Free, public-land and collecting filters are off.',
  unavailable: 'Access data not loaded for this area (service unavailable). Free, public-land and collecting filters are off.',
}

// Severity: a cell covered by several areas takes the most restrictive value
// for each attribute, so an overlap can never make a place look better.
const PA_RANK: Record<PublicAccess, number> = { open: 0, unknown: 1, restricted: 2, closed: 3 }
const FEE_RANK: Record<FeeStatus, number> = { free: 0, unknown: 1, fee: 2 }
const COL_RANK: Record<Collecting, number> = { allowed: 0, likely_allowed: 0.5, unknown: 1, restricted: 2, prohibited: 3 }
const worst = <T extends string>(vals: T[], rank: Record<string, number>, fallback: T): T =>
  vals.length ? vals.reduce((a, b) => (rank[b] > rank[a] ? b : a)) : fallback

export interface CellAccess {
  covered: boolean
  areaName: string | null
  manager_type: string | null
  public_access: PublicAccess
  fee_status: FeeStatus
  fee_source: 'estimated' | 'ridb' | null
  collecting: Collecting
  collecting_source: CollectingSource
  collecting_rule: string | null
}

export const UNKNOWN_ACCESS: CellAccess = {
  covered: false, areaName: null, manager_type: null, public_access: 'unknown',
  fee_status: 'unknown', fee_source: null, collecting: 'unknown', collecting_source: null, collecting_rule: null,
}

/** Access attributes at a cell centre [lon, lat]. No covering area means everything unknown. */
export function accessForCell(point: [number, number], areas: AccessArea[]): CellAccess {
  const hits = areas.filter((a) => pointInArea(point, a))
  if (!hits.length) return { ...UNKNOWN_ACCESS }
  const fee = worst(hits.map((h) => h.fee_status), FEE_RANK, 'unknown' as FeeStatus)
  const col = worst(hits.map((h) => h.collecting), COL_RANK, 'unknown' as Collecting)
  const feeHit = hits.find((h) => h.fee_status === fee)
  const colHit = hits.find((h) => h.collecting === col)
  return {
    covered: true,
    areaName: hits.find((h) => h.name)?.name ?? null,
    manager_type: hits.find((h) => h.manager_type)?.manager_type ?? null,
    public_access: worst(hits.map((h) => h.public_access), PA_RANK, 'unknown' as PublicAccess),
    fee_status: fee, fee_source: feeHit?.fee_source ?? null,
    collecting: col, collecting_source: colHit?.collecting_source ?? null,
    collecting_rule: colHit?.collecting_rule ?? null,
  }
}

export interface Switches {
  free: boolean
  public: boolean
  collecting: boolean
  /** With collecting on, also keep 'likely_allowed' (BLM/USFS, estimated) areas. */
  includeLikely: boolean
  /** Keep cells whose attribute is unknown. Off by default: unknown is never assumed good. */
  includeUnknown: boolean
}

export const NO_SWITCHES: Switches = { free: false, public: false, collecting: false, includeLikely: false, includeUnknown: false }

/** Does one cell pass the enabled switches? */
export function passesSwitches(a: CellAccess, sw: Switches): boolean {
  const ok = (good: boolean, unknown: boolean) => good || (sw.includeUnknown && unknown)
  if (sw.free && !ok(a.fee_status === 'free', a.fee_status === 'unknown')) return false
  if (sw.public) {
    // Private managers are not public land even if the polygon says open.
    const priv = String(a.manager_type || '').toLowerCase() === 'private'
    if (priv || !ok(a.public_access === 'open', a.public_access === 'unknown')) return false
  }
  if (sw.collecting && !ok(a.collecting === 'allowed' || (sw.includeLikely && a.collecting === 'likely_allowed'), a.collecting === 'unknown')) return false
  return true
}

/**
 * Switches actually applied: only when access data is loaded. When it is not,
 * every switch is off (and disabled in the UI) rather than guessing.
 */
export function effectiveSwitches(sw: Switches, status: AccessStatus): Switches {
  return status === 'loaded' ? sw : { ...NO_SWITCHES }
}

export interface AccessedCell { access: CellAccess }

/** Attach access to each cell (from its centre) and drop cells failing the switches. */
export function attachAccess<T extends { lon: number, lat: number }>(cells: T[], areas: AccessArea[]): (T & AccessedCell)[] {
  return cells.map((c) => ({ ...c, access: accessForCell([c.lon, c.lat], areas) }))
}

export function filterByAccess<T extends AccessedCell>(cells: T[], sw: Switches): T[] {
  return cells.filter((c) => passesSwitches(c.access, sw))
}

// ── Modes ──────────────────────────────────────────────────────────────────

export type ForayMode = 'forager' | 'researcher' | 'leader'

export interface ModeConfig {
  key: ForayMode
  label: string
  blurb: string
  switches: Switches
  showComponents: boolean
  showSample: boolean
  showCaveats: boolean
  showAccessCols: boolean
  showShortlist: boolean
  listLimit: number
}

/** Mode only changes defaults and which panels show; the score is the same. */
export const FORAY_MODES: Record<ForayMode, ModeConfig> = {
  forager: {
    key: 'forager', label: 'Forager', blurb: 'Top places to look, in plain language.',
    switches: { ...NO_SWITCHES },
    showComponents: false, showSample: false, showCaveats: false, showAccessCols: false, showShortlist: false, listLimit: 8,
  },
  researcher: {
    key: 'researcher', label: 'Researcher', blurb: 'Score components, sample size and caveats.',
    switches: { ...NO_SWITCHES },
    showComponents: true, showSample: true, showCaveats: true, showAccessCols: true, showShortlist: false, listLimit: 25,
  },
  leader: {
    key: 'leader', label: 'Foray leader', blurb: 'Group-ready shortlist with access and collecting status.',
    switches: { free: true, public: true, collecting: true, includeLikely: true, includeUnknown: false },
    showComponents: false, showSample: true, showCaveats: false, showAccessCols: true, showShortlist: true, listLimit: 12,
  },
}

// ── Shortlist ──────────────────────────────────────────────────────────────

export interface ShortlistRow {
  rank: number
  site: string
  lat: number
  lon: number
  score: number
  access_class: string
  manager: string
  fee_status: string
  collecting_status: string
  finds: number
  top_species: string
  notes: string
}

const src = (s: string | null) => (s === 'ridb' ? 'verified' : s === 'estimated' ? 'estimated' : s === 'rule' ? 'local rule' : '')

/** The local rule behind a cell's collecting value, for a source link; null for estimates. */
export function collectingRuleFor(a: { collecting_source: CollectingSource, collecting_rule: string | null }) {
  return a.collecting_source === 'rule' ? collectingRuleById(a.collecting_rule) : null
}

export function feeLabel(a: CellAccess): string {
  if (a.fee_status === 'unknown') return 'unknown'
  const s = src(a.fee_source)
  return s ? `${a.fee_status} (${s})` : a.fee_status
}

export function collectingLabel(a: CellAccess): string {
  if (a.collecting === 'unknown') return 'unknown'
  if (a.collecting === 'likely_allowed') return LIKELY_LABEL
  const s = src(a.collecting_source)
  return s ? `${a.collecting} (${s})` : a.collecting
}

export const cellSiteName = (c: { lat: number, lon: number, access?: CellAccess }) =>
  c.access?.areaName || `${c.lat.toFixed(3)}, ${c.lon.toFixed(3)}`

/** Ranked cells -> shortlist rows. `notes` is keyed by cell key. */
export function buildShortlist(
  ranked: Array<{ key: string, lat: number, lon: number, score: number, n: number, components?: { species: string }[], access: CellAccess }>,
  notes: Record<string, string> = {},
  limit = 12,
): ShortlistRow[] {
  return ranked.slice(0, limit).map((c, i) => ({
    rank: i + 1,
    site: cellSiteName(c),
    lat: Number(c.lat.toFixed(4)),
    lon: Number(c.lon.toFixed(4)),
    score: Math.round(c.score * 100) / 100,
    access_class: c.access.public_access,
    manager: c.access.manager_type || '',
    fee_status: feeLabel(c.access),
    collecting_status: collectingLabel(c.access),
    finds: c.n,
    top_species: (c.components || []).slice(0, 3).map((x) => x.species).join('; '),
    notes: notes[c.key] || '',
  }))
}

export const SHORTLIST_COLUMNS: (keyof ShortlistRow)[] = [
  'rank', 'site', 'lat', 'lon', 'score', 'access_class', 'manager', 'fee_status',
  'collecting_status', 'finds', 'top_species', 'notes',
]

function csvCell(v: unknown): string {
  const s = v === null || v === undefined ? '' : String(v)
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}

/** CSV with a header row and the disclaimer as a trailing comment row. */
export function shortlistToCsv(rows: ShortlistRow[]): string {
  const lines = [SHORTLIST_COLUMNS.join(',')]
  for (const r of rows) lines.push(SHORTLIST_COLUMNS.map((k) => csvCell(r[k])).join(','))
  lines.push(csvCell(`# ${DISCLAIMER}`))
  return lines.join('\n') + '\n'
}

/** Plain-text shortlist for pasting into a message. */
export function shortlistToText(rows: ShortlistRow[], title = 'Foray shortlist'): string {
  const out = [title, '']
  for (const r of rows) {
    out.push(`${r.rank}. ${r.site} (${r.lat}, ${r.lon}) score ${r.score}`)
    out.push(`   access: ${r.access_class}${r.manager ? ` / ${r.manager}` : ''}; fee: ${r.fee_status}; collecting: ${r.collecting_status}`)
    if (r.notes) out.push(`   notes: ${r.notes}`)
  }
  out.push('', DISCLAIMER)
  return out.join('\n') + '\n'
}
