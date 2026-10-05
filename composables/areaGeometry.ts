// Pure helpers for "allowed areas": validating and parsing polygons client-side.
// The server re-validates; this only gives fast, readable feedback.

import { MAX_UPLOAD_BYTES } from '../netlify/lib/observation-upload.mjs'

export const FEE_STATUSES = ['free', 'fee', 'unknown'] as const
export const COLLECTING_STATUSES = ['allowed', 'likely_allowed', 'restricted', 'prohibited', 'unknown'] as const
export const MAX_AREAS_PER_IMPORT = 500

export const FEE_LABELS: Record<string, string> = { free: 'Free', fee: 'Fee required', unknown: 'Unknown' }
export const COLLECTING_LABELS: Record<string, string> = {
  allowed: 'Allowed', likely_allowed: 'Likely allowed', restricted: 'Restricted',
  prohibited: 'Prohibited', unknown: 'Unknown',
}

export class AreaError extends Error {}

export interface AreaInput {
  name: string
  geometry: any
  fee_status: string
  collecting: string
  notes: string
}

const validPos = (p: any) =>
  Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1])
  && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90

function validRing(r: any): boolean {
  return Array.isArray(r) && r.length >= 4 && r.every(validPos)
    && r[0][0] === r[r.length - 1][0] && r[0][1] === r[r.length - 1][1]
}
const validPolygon = (c: any) => Array.isArray(c) && c.length > 0 && c.every(validRing)

/** Throws AreaError unless `g` is a well-formed Polygon or MultiPolygon. */
export function assertPolygonGeometry(g: any): void {
  if (g?.type === 'Polygon') {
    if (!validPolygon(g.coordinates)) throw new AreaError('That polygon is not closed or has invalid coordinates.')
  } else if (g?.type === 'MultiPolygon') {
    if (!Array.isArray(g.coordinates) || !g.coordinates.length || !g.coordinates.every(validPolygon)) {
      throw new AreaError('That multipolygon has invalid coordinates.')
    }
  } else {
    throw new AreaError('Only polygon areas are accepted (no points or lines).')
  }
}

/** Leaflet-style [[lat,lng],...] ring -> closed GeoJSON Polygon. */
export function ringToPolygon(latlngs: Array<[number, number]>): any {
  if (latlngs.length < 3) throw new AreaError('Draw at least three points.')
  const ring = latlngs.map(([lat, lng]) => [lng, lat])
  const [a, b] = [ring[0], ring[ring.length - 1]]
  if (a[0] !== b[0] || a[1] !== b[1]) ring.push([a[0], a[1]])
  const g = { type: 'Polygon', coordinates: [ring] }
  assertPolygonGeometry(g)
  return g
}

/** GeoJSON text/object -> [{name, geometry}]. Polygons only; throws if none usable. */
export function parseAreaGeoJSON(input: string | object, maxBytes = MAX_UPLOAD_BYTES) {
  let gj: any = input
  if (typeof input === 'string') {
    if (input.length > maxBytes) throw new AreaError('File is larger than 3 MB. Simplify the boundaries first.')
    try { gj = JSON.parse(input) } catch { throw new AreaError('That file is not valid JSON.') }
  }
  let items: any[]
  if (gj?.type === 'FeatureCollection' && Array.isArray(gj.features)) items = gj.features
  else if (gj?.type === 'Feature') items = [gj]
  else if (gj?.type && gj.coordinates) items = [{ type: 'Feature', geometry: gj, properties: {} }]
  else throw new AreaError('Expected a GeoJSON FeatureCollection of polygons.')
  if (items.length > MAX_AREAS_PER_IMPORT) throw new AreaError(`Too many features (max ${MAX_AREAS_PER_IMPORT}).`)

  const areas: Array<{ name: string, geometry: any }> = []
  let skipped = 0
  for (const f of items) {
    try {
      assertPolygonGeometry(f?.geometry)
      const p = f.properties || {}
      areas.push({ name: String(p.name ?? p.Name ?? p.NAME ?? '').trim() || `Area ${areas.length + 1}`, geometry: f.geometry })
    } catch { skipped++ }
  }
  if (!areas.length) throw new AreaError(`No valid polygons found${skipped ? ` (${skipped} non-polygon or invalid features skipped)` : ''}.`)
  return { areas, skipped }
}

/** Normalise and validate the editor form; throws AreaError with a readable message. */
export function validateAreaInput(a: Partial<AreaInput>): AreaInput {
  const name = String(a.name ?? '').trim()
  if (!name) throw new AreaError('Give the area a name.')
  if (name.length > 120) throw new AreaError('Name is too long (120 characters max).')
  const fee_status = a.fee_status || 'unknown'
  const collecting = a.collecting || 'unknown'
  if (!(FEE_STATUSES as readonly string[]).includes(fee_status)) throw new AreaError('Unknown fee status.')
  if (!(COLLECTING_STATUSES as readonly string[]).includes(collecting)) throw new AreaError('Unknown collecting status.')
  const notes = String(a.notes ?? '').trim()
  if (notes.length > 2000) throw new AreaError('Notes are too long (2000 characters max).')
  assertPolygonGeometry(a.geometry)
  return { name, geometry: a.geometry, fee_status, collecting, notes }
}

/** The label the owner's assertion carries on the map. */
export const scopeLabel = (scope: string) => (scope === 'club' ? 'Club area' : 'Your area')

export const canManage = (role?: string | null) => role === 'owner' || role === 'admin'
