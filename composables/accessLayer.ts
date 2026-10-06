// Pure logic for the map's Access overlay (WANT-17): style mapping, filtering,
// legend, popup content and viewport -> bbox. No Leaflet/Vue imports so it
// tests on fixtures. Unknown is always neutral grey and never reads as free,
// public or collectable.

import { DISCLAIMER, LIKELY_DISCLAIMER, LIKELY_LABEL, COLORADO_BBOX, passesSwitches } from './forayPlanner'
import type { AccessArea, CellAccess, Switches } from './forayPlanner'

export type AccessAttr = 'public' | 'fee' | 'collecting'
export type AccessSource = 'padus' | 'user' | 'club'
export const ACCESS_ATTRS: { key: AccessAttr, label: string }[] = [
  { key: 'public', label: 'Public access' },
  { key: 'fee', label: 'Fee status' },
  { key: 'collecting', label: 'Collecting' },
]
export const ACCESS_SOURCES: { key: AccessSource, label: string }[] = [
  { key: 'padus', label: 'Public lands' },
  { key: 'user', label: 'My areas' },
  { key: 'club', label: 'Club areas' },
]

export const UNKNOWN_COLOR = '#9aa0a6'
const GOOD = '#2e7d32', WARN = '#d99a1e', BAD = '#b3382c'
export const SOURCE_OUTLINE: Record<AccessSource, string | null> = { padus: null, user: '#6a1b9a', club: '#ad1457' }
export const ESTIMATED_DASH = '6 4'
export const MIN_ACCESS_ZOOM = 8
export const LINES_MIN_ZOOM = 12
export const MAX_VIEW_TILES = 9

export const sourceOf = (a: Pick<AccessArea, 'source'>): AccessSource => (a.source === 'user' || a.source === 'club' ? a.source : 'padus')
export const isOwnerAsserted = (a: AccessArea) => sourceOf(a) !== 'padus' && a.owner_asserted !== false

interface Cls { color: string, label: string, estimated: boolean, unknown: boolean }

/** Label, colour and "is this a heuristic" for one attribute of one area. */
export function classify(a: AccessArea, attr: AccessAttr): Cls {
  const owner = isOwnerAsserted(a)
  if (attr === 'public') {
    const m: Record<string, [string, string]> = {
      open: [GOOD, 'Open to the public'], restricted: [WARN, 'Restricted access'],
      closed: [BAD, 'Closed to the public'], unknown: [UNKNOWN_COLOR, 'Unknown'],
    }
    const [color, label] = m[a.public_access] ?? m.unknown
    return { color, label, estimated: false, unknown: !m[a.public_access] || a.public_access === 'unknown' }
  }
  if (attr === 'fee') {
    const m: Record<string, [string, string]> = {
      free: ['#1a8a9a', 'Free'], fee: ['#d9731e', 'Fee required'], unknown: [UNKNOWN_COLOR, 'Unknown'],
    }
    const [color, label] = m[a.fee_status] ?? m.unknown
    const unknown = !m[a.fee_status] || a.fee_status === 'unknown'
    return { color, label, unknown, estimated: !unknown && !owner && a.fee_source === 'estimated' }
  }
  const m: Record<string, [string, string]> = {
    allowed: [GOOD, 'Allowed'], likely_allowed: [GOOD, LIKELY_LABEL], restricted: [WARN, 'Restricted'],
    prohibited: [BAD, 'Prohibited'], unknown: [UNKNOWN_COLOR, 'Unknown'],
  }
  const [color, label] = m[a.collecting] ?? m.unknown
  const unknown = !m[a.collecting] || a.collecting === 'unknown'
  return {
    color, label, unknown,
    estimated: !unknown && !owner && (a.collecting === 'likely_allowed' || a.collecting_source === 'estimated'),
  }
}

export interface PolyStyle { color: string, fillColor: string, weight: number, opacity: number, fillOpacity: number, dashArray: string | null }

export function styleForArea(a: AccessArea, attr: AccessAttr): PolyStyle {
  const c = classify(a, attr)
  const outline = SOURCE_OUTLINE[sourceOf(a)]
  return {
    fillColor: c.color,
    color: outline ?? c.color,
    weight: outline ? 3 : c.unknown ? 1 : 2,
    opacity: c.unknown ? 0.6 : 0.95,
    fillOpacity: c.unknown ? 0.12 : c.estimated ? 0.22 : 0.4,
    dashArray: c.estimated ? ESTIMATED_DASH : null,
  }
}

export interface LegendItem { color: string, label: string, dashed?: boolean, kind?: 'fill' | 'outline' }

export function legendFor(attr: AccessAttr): LegendItem[] {
  const items: Record<AccessAttr, LegendItem[]> = {
    public: [
      { color: GOOD, label: 'Open to the public' }, { color: WARN, label: 'Restricted' },
      { color: BAD, label: 'Closed' }, { color: UNKNOWN_COLOR, label: 'Unknown (not assumed public)' },
    ],
    fee: [
      { color: '#1a8a9a', label: 'Free' }, { color: '#d9731e', label: 'Fee required' },
      { color: UNKNOWN_COLOR, label: 'Unknown (not assumed free)' },
    ],
    collecting: [
      { color: GOOD, label: 'Allowed' }, { color: GOOD, label: LIKELY_LABEL, dashed: true },
      { color: WARN, label: 'Restricted' }, { color: BAD, label: 'Prohibited' },
      { color: UNKNOWN_COLOR, label: 'Unknown (not assumed allowed)' },
    ],
  }
  const out = [...items[attr]]
  if (attr !== 'public') out.push({ color: '#555', label: 'Dashed outline = estimated', dashed: true, kind: 'outline' })
  out.push({ color: SOURCE_OUTLINE.user!, label: 'Thick purple outline = your area', kind: 'outline' })
  out.push({ color: SOURCE_OUTLINE.club!, label: 'Thick magenta outline = club area', kind: 'outline' })
  return out
}

/** CellAccess view of one area, so the foray switch logic is shared. */
export function areaAsAccess(a: AccessArea): CellAccess {
  return {
    covered: true, areaName: a.name, manager_type: a.manager_type, public_access: a.public_access,
    fee_status: a.fee_status, fee_source: a.fee_source, collecting: a.collecting, collecting_source: a.collecting_source,
  }
}

export type SourceToggles = Record<AccessSource, boolean>
export const DEFAULT_SOURCES: SourceToggles = { padus: true, user: true, club: true }

/** Areas that pass the source toggles and the free/public/collecting switches. */
export function filterAreas(areas: AccessArea[], sw: Switches, sources: SourceToggles = DEFAULT_SOURCES): AccessArea[] {
  return areas.filter((a) => sources[sourceOf(a)] !== false && passesSwitches(areaAsAccess(a), sw))
}

export const sourceCounts = (areas: AccessArea[]): Record<AccessSource, number> => {
  const n: Record<AccessSource, number> = { padus: 0, user: 0, club: 0 }
  for (const a of areas) n[sourceOf(a)]++
  return n
}

export function areaTag(a: AccessArea): string | null {
  const s = sourceOf(a)
  if (s === 'user') return 'Your area'
  if (s === 'club') return `Club area: ${a.set_name || 'unnamed set'}`
  return null
}

const ESC: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }
const esc = (v: unknown) => String(v ?? '').replace(/[&<>"']/g, (c) => ESC[c])
const srcWord = (s: string | null) => (s === 'ridb' ? 'verified (RIDB)' : s === 'estimated' ? 'estimated' : '')

export function feeText(a: AccessArea): string {
  if (a.fee_status === 'unknown') return 'Unknown'
  const base = a.fee_status === 'free' ? 'Free' : 'Fee required'
  if (isOwnerAsserted(a)) return `${base} (owner-asserted)`
  const s = srcWord(a.fee_source)
  return s ? `${base} (${s})` : base
}

export function collectingText(a: AccessArea): string {
  if (a.collecting === 'unknown') return 'Unknown'
  const base = classify(a, 'collecting').label
  if (isOwnerAsserted(a)) return `${a.collecting === 'likely_allowed' ? 'Allowed' : base} (owner-asserted)`
  if (a.collecting === 'likely_allowed') return LIKELY_LABEL
  const s = srcWord(a.collecting_source)
  return s ? `${base} (${s})` : base
}

/** Popup HTML (all values escaped). */
export function popupHtml(a: AccessArea): string {
  const tag = areaTag(a)
  const rows: [string, string][] = [
    ['Manager', a.manager_type || 'Unknown'],
    ['Public access', classify(a, 'public').label], ['Fee', feeText(a)], ['Collecting', collectingText(a)],
  ]
  const asserted = isOwnerAsserted(a)
  const est = !asserted && (a.fee_source === 'estimated' || a.collecting_source === 'estimated' || a.collecting === 'likely_allowed')
  return `<div class="acc-pop"><strong>${esc(a.name || 'Unnamed area')}</strong>`
    + (tag ? `<div class="acc-pop-tag">${esc(tag)}</div>` : '')
    + `<dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>`
    + (asserted ? '<p class="acc-pop-note">Fee and collecting were set by the area owner, not estimated.</p>' : '')
    + (est ? `<p class="acc-pop-note">${esc(DISCLAIMER)}</p>` : '')
    + (!asserted && a.collecting === 'likely_allowed' ? `<p class="acc-pop-note">${esc(LIKELY_DISCLAIMER)}</p>` : '')
    + '</div>'
}

type BBox = [number, number, number, number]
/** Visible bbox clamped to Colorado, or null when outside it. */
export function clampToLoaded(b: BBox): BBox | null {
  const [cw, cs, ce, cn] = COLORADO_BBOX
  const w = Math.max(b[0], cw), s = Math.max(b[1], cs), e = Math.min(b[2], ce), n = Math.min(b[3], cn)
  return w < e && s < n ? [w, s, e, n] : null
}

/** Snap outward to a 0.25 deg grid so small pans reuse cached tiles. */
export function snapBbox([w, s, e, n]: BBox, step = 0.25): BBox {
  return [Math.floor(w / step) * step, Math.floor(s / step) * step, Math.ceil(e / step) * step, Math.ceil(n / step) * step]
}

export type ViewPlan =
  | { kind: 'ok', bbox: BBox, lines: boolean }
  | { kind: 'zoom-in' }
  | { kind: 'outside' }

/** What to load for a viewport. Respects zoom floor, Colorado, tile count; lines only when the bbox fits the line cap. */
export function planView(view: BBox, zoom: number): ViewPlan {
  if (zoom < MIN_ACCESS_ZOOM) return { kind: 'zoom-in' }
  const c = clampToLoaded(view)
  if (!c) return { kind: 'outside' }
  const lines = zoom >= LINES_MIN_ZOOM && c[2] - c[0] <= 0.5 && c[3] - c[1] <= 0.5
  const bbox = lines ? c : (clampToLoaded(snapBbox(c)) ?? c)
  const tiles = Math.ceil((bbox[2] - bbox[0]) / 3 - 1e-9) * Math.ceil((bbox[3] - bbox[1]) / 3 - 1e-9)
  return tiles > MAX_VIEW_TILES ? { kind: 'zoom-in' } : { kind: 'ok', bbox, lines }
}

export const ACCESS_NOTICE = {
  zoomIn: `Zoom in to load access areas (zoom ${MIN_ACCESS_ZOOM} or closer).`,
  outside: 'Access data not loaded for this area (only Colorado is loaded).',
  partial: 'Access data is incomplete here: some areas may be missing, so no colored area does not mean no access.',
  notLoaded: 'Access data not loaded for this area.',
  lines: `Roads and trails show at zoom ${LINES_MIN_ZOOM} or closer.`,
}

/** Visible message for a plan + load status, or ''. */
export function accessNotice(plan: ViewPlan, status: string): string {
  if (plan.kind === 'zoom-in') return ACCESS_NOTICE.zoomIn
  if (plan.kind === 'outside') return ACCESS_NOTICE.outside
  if (status === 'partial') return ACCESS_NOTICE.partial
  if (status === 'not-loaded' || status === 'no-data' || status === 'unavailable') return ACCESS_NOTICE.notLoaded
  return ''
}
