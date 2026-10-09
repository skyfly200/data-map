// Foray planner model layers and the under-sampled ranking (WANT-17 phases 2,
// 3 and 5), client side. Pure, no framework imports, so tests run on fixtures.
//
// The server (netlify/functions/foray-layers.mjs) builds the layers; this
// decides which saved models feed them for the species in season, lays out the
// candidate cells for "where few have looked", and ranks them.

import { cellAt, cellKeyAt } from './gridCells'

export const MAX_LAYER_MODELS = 6
/** At most this many candidate cells per sample request (server cap 1500). */
export const MAX_CANDIDATES = 1200
/** A candidate needs at least this layer value to be called promising. */
export const MIN_PROMISE = 0.5
/** "Few finds": a candidate with more than this many finds is not under-sampled. */
export const MAX_FINDS = 2

export interface ForayModel {
  id: string
  title: string
  taxon: string | null
  auc: number | null
  own: boolean
  registered: boolean
  ranges: boolean
  contributions: boolean
  usable: boolean
}

export type ForayLayer = 'ensemble' | 'habitat'

const norm = (s: string | null | undefined) => String(s || '').trim().toLowerCase()

/**
 * Does a model trained on `taxon` (a species, or a genus) cover `species`?
 * "Cantharellus" covers "Cantharellus cibarius"; "Boletus" does not cover
 * "Boletinellus merulioides".
 */
export function taxonCovers(taxon: string | null | undefined, species: string): boolean {
  const t = norm(taxon), s = norm(species)
  if (!t || !s) return false
  return s === t || s.startsWith(`${t} `)
}

/** The in-season species a model covers: by its taxon, else by a species named in its title. */
export function modelSpecies(m: Pick<ForayModel, 'taxon' | 'title'>, species: string[]): string[] {
  if (m.taxon) return species.filter((s) => taxonCovers(m.taxon, s))
  const title = norm(m.title)
  return species.filter((s) => norm(s) && title.includes(norm(s)))
}

export interface PickedModel { id: string, weight: number, species: string[] }

/**
 * Default picks for a layer: models that cover a species in season, weighted by
 * that species' phenology weight (the best one when a model covers several).
 * The habitat layer only takes models with stored ranges and contributions.
 * Strongest first, at most MAX_LAYER_MODELS.
 */
export function defaultPicks(models: ForayModel[], weights: Map<string, number>, layer: ForayLayer): PickedModel[] {
  const names = [...weights.keys()]
  const out: PickedModel[] = []
  for (const m of models) {
    if (layer === 'habitat' ? !(m.ranges && m.contributions) : !m.usable) continue
    const covered = modelSpecies(m, names)
    if (!covered.length) continue
    const weight = Math.max(...covered.map((s) => weights.get(s) || 0))
    if (weight > 0) out.push({ id: m.id, weight, species: covered })
  }
  return out.sort((a, b) => b.weight - a.weight || a.id.localeCompare(b.id)).slice(0, MAX_LAYER_MODELS)
}

/** `id:weight,...` for the endpoint. */
export const modelsParam = (picks: PickedModel[]) => picks.map((p) => `${p.id}:${p.weight.toFixed(3)}`).join(',')

export interface ViewBounds { west: number, south: number, east: number, north: number }
export interface Candidate { key: string, lat: number, lon: number, polygon: [number, number][] }

/**
 * Every grid cell in `bounds`, deduplicated by key. When there would be more
 * than `max`, the view is too large: returns the cap so the caller can ask
 * for a closer zoom rather than sample a thinned, misleading grid.
 */
export function candidateCells(bounds: ViewBounds, size: number, shape = 'hex', max = MAX_CANDIDATES): { cells: Candidate[], tooMany: boolean } {
  const { west, south, east, north } = bounds
  if (!(size > 0) || !(east > west) || !(north > south)) return { cells: [], tooMany: false }
  const step = size * 0.5
  const estimate = ((east - west) / size) * ((north - south) / size)
  if (estimate > max) return { cells: [], tooMany: true }
  const seen = new Map<string, Candidate>()
  for (let lat = south + step / 2; lat < north; lat += step) {
    for (let lon = west + step / 2; lon < east; lon += step) {
      const c = cellAt(lon, lat, size, shape)
      if (seen.has(c.key)) continue
      if (c.lat < south || c.lat > north || c.lon < west || c.lon > east) continue
      seen.set(c.key, { key: c.key, lat: c.lat, lon: c.lon, polygon: c.polygon as [number, number][] })
      if (seen.size > max) return { cells: [], tooMany: true }
    }
  }
  return { cells: [...seen.values()], tooMany: false }
}

/** Finds per cell key, for every cell with any finds (no minimum). */
export function findsPerCell(features: any[], size: number, shape = 'hex', landCover?: string): Map<string, number> {
  const n = new Map<string, number>()
  for (const f of features || []) {
    const co = f?.geometry?.coordinates
    const lon = Number(co?.[0]), lat = Number(co?.[1])
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue
    if (landCover && f.properties?.land_cover_label !== landCover) continue
    const k = cellKeyAt(lon, lat, size, shape)
    n.set(k, (n.get(k) || 0) + 1)
  }
  return n
}

export interface Opportunity<T> { cell: T, promise: number, n: number, opportunity: number }

/**
 * Phase 5 ranking: promising cells with few finds, by promise ÷ (1 + finds).
 * Cells outside the models' training envelope are left out (their promise is
 * an extrapolation) and counted, so the page can say how many were dropped.
 */
export function rankUnderSampled<T extends { n: number, promise: number | null, outside: boolean | null }>(
  cells: T[],
  { minPromise = MIN_PROMISE, maxFinds = MAX_FINDS, limit = 10 } = {},
): { ranked: Opportunity<T>[], outside: number, considered: number } {
  let outside = 0
  let considered = 0
  const keep: Opportunity<T>[] = []
  for (const c of cells) {
    if (c.promise == null || c.n > maxFinds) continue
    considered++
    if (c.outside) { outside++; continue }
    if (c.promise < minPromise) continue
    keep.push({ cell: c, promise: c.promise, n: c.n, opportunity: c.promise / (1 + c.n) })
  }
  keep.sort((a, b) => b.opportunity - a.opportunity || b.promise - a.promise)
  return { ranked: keep.slice(0, limit), outside, considered }
}
