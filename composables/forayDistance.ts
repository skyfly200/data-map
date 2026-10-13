// Distance filter for the foray planner: how far each place is from the user
// (their location, a typed place or the map centre), a radius cut and an
// optional nearest-first sort. Pure, so tests run without a browser.

import { haversineKm } from './useFilters'
import { pointInArea } from '../netlify/lib/access-ingest.mjs'

export const KM_PER_MI = 1.609344

/** Radius choices in miles; 0 means no limit. */
export const RADIUS_OPTIONS_MI = [10, 25, 50, 100, 0]
export const DEFAULT_RADIUS_MI = 25

export type OriginSource = 'gps' | 'typed' | 'map' | 'default'

export interface Origin {
  lat: number
  lon: number
  /** Short label for the summary line, e.g. "your location" or "Boulder". */
  label: string
  source: OriginSource
}

/** Start point before the user picks one: Boulder, with the 25 mi and 45 min drive defaults. */
export const DEFAULT_ORIGIN: Origin = { lat: 40.015, lon: -105.2705, label: 'Boulder', source: 'default' }

/** Stored in place of an origin when the user cleared it, so the default does not come back. */
export const NO_ORIGIN = 'none'

/**
 * Origin to start the page with, from the saved value: a saved place, nothing
 * if the user cleared it, otherwise the Boulder default.
 */
export function initialOrigin(saved: string | null): Origin | null {
  if (saved === NO_ORIGIN) return null
  return parseSavedOrigin(saved) ?? { ...DEFAULT_ORIGIN }
}

export type ForaySort = 'score' | 'near'

export const validLatLon = (lat: unknown, lon: unknown): boolean =>
  typeof lat === 'number' && typeof lon === 'number' && Number.isFinite(lat) && Number.isFinite(lon)
  && Math.abs(lat) <= 90 && Math.abs(lon) <= 180

/** "40.01, -105.27" (or space separated) -> {lat, lon}; anything else -> null. */
export function parseLatLon(text: string): { lat: number, lon: number } | null {
  const m = String(text || '').trim().match(/^(-?\d+(?:\.\d+)?)\s*[,\s]\s*(-?\d+(?:\.\d+)?)$/)
  if (!m) return null
  const lat = Number(m[1]), lon = Number(m[2])
  return validLatLon(lat, lon) ? { lat, lon } : null
}

export function distanceMi(origin: { lat: number, lon: number }, p: { lat: number, lon: number }): number {
  return haversineKm({ lat: origin.lat, lng: origin.lon }, { lat: p.lat, lng: p.lon }) / KM_PER_MI
}

/**
 * Attach `distMi` to each cell and drop those beyond the radius. Without an
 * origin, cells pass through unchanged (distMi null) whatever the radius.
 */
export function filterByDistance<T extends { lat: number, lon: number }>(
  cells: T[], origin: { lat: number, lon: number } | null, radiusMi: number,
): (T & { distMi: number | null })[] {
  if (!origin) return cells.map((c) => ({ ...c, distMi: null }))
  const out: (T & { distMi: number })[] = []
  for (const c of cells) {
    const d = distanceMi(origin, c)
    if (!radiusMi || d <= radiusMi) out.push({ ...c, distMi: d })
  }
  return out
}

/** Nearest first; ties (and cells without a distance) keep their score order. */
export function sortByDistance<T extends { distMi?: number | null }>(cells: T[]): T[] {
  return cells
    .map((c, i) => ({ c, i }))
    .sort((a, b) => (a.c.distMi ?? Infinity) - (b.c.distMi ?? Infinity) || a.i - b.i)
    .map((x) => x.c)
}

export function formatDistance(mi: number | null | undefined): string {
  if (mi == null || !Number.isFinite(mi)) return ''
  if (mi < 1) return '<1 mi'
  return `${mi < 10 ? mi.toFixed(1) : Math.round(mi)} mi`
}

export const radiusLabel = (mi: number) => (!mi ? 'Any distance' : `${mi} mi`)

// ── Drive time ─────────────────────────────────────────────────────────────
// Straight-line distance alone suggests places that are close as the crow
// flies but a long way round by road (over a pass, down a back road). A
// drive-time area (isochrone) from a routing service cuts those. Without one
// (service down, offline) only the straight-line radius applies.

/** Drive-time choices in minutes; 0 means no drive-time limit. */
export const DRIVE_OPTIONS_MIN = [30, 45, 60, 90, 120, 0]
export const DEFAULT_DRIVE_MIN = 45

export const driveLabel = (min: number) => (!min ? 'Any drive time' : min < 60 ? `${min} min drive` : `${String(min / 60).replace('.5', '½')} hr drive`)

export interface DriveArea { type: 'MultiPolygon', coordinates: number[][][][] }

/** Is [lon, lat] inside the drive-time area? */
export const inDriveArea = (lon: number, lat: number, geom: DriveArea) => pointInArea([lon, lat], { geom })

/** Keep cells inside the drive-time area; without one, everything passes. */
export function filterByDrive<T extends { lat: number, lon: number }>(cells: T[], geom: DriveArea | null): T[] {
  if (!geom?.coordinates?.length) return cells
  return cells.filter((c) => inDriveArea(c.lon, c.lat, geom))
}

/** Saved origin from localStorage JSON, or null if missing or malformed. */
export function parseSavedOrigin(json: string | null): Origin | null {
  try {
    const o = JSON.parse(json || 'null')
    if (!o || !validLatLon(o.lat, o.lon)) return null
    const source: OriginSource = o.source === 'gps' || o.source === 'map' ? o.source : 'typed'
    return { lat: o.lat, lon: o.lon, label: String(o.label || 'saved place'), source }
  } catch { return null }
}
