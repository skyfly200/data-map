// Observation coordinates for a layer that is computed FROM the finds.
//
// Most Earth Engine layers are a fixed recipe over a public dataset. The
// habitat-similarity layer is not: it asks "where else looks like the places
// this taxon was found", so its input is our own observations. Those live on
// this side, not in Earth Engine, so they are picked here and sent along as
// plain coordinates — never as anything Earth Engine would execute.
//
// Thinned to a fixed cap. The points travel inside the request that builds the
// map id, and a few thousand Morchella finds would make that request large for
// no gain: the mean of five hundred well-spread sites is the same habitat
// signature as the mean of five thousand.

import { matchesTaxon } from './dataset-taxa.mjs'
import { loadBaseline } from './baseline.mjs'

export const REFERENCE_POINT_LIMIT = 500

/**
 * The [lon, lat] pairs of every record of `taxon`, de-duplicated and thinned.
 *
 * Coordinates are rounded to ~10 m (the embedding's own pixel) before
 * de-duplicating, so twenty photos of one log count once rather than weighting
 * the signature toward whoever photographs most. Thinning takes every nth
 * point in a stable order, so the same dataset always gives the same sites and
 * the same rendered layer.
 */
export function pickReferencePoints(features = [], taxon = '', limit = REFERENCE_POINT_LIMIT) {
  const seen = new Set()
  const points = []
  for (const f of features) {
    const props = f?.properties || {}
    if (!taxon || !matchesTaxon(props, taxon)) continue
    const coords = f?.geometry?.coordinates
    const lon = Number(Array.isArray(coords) ? coords[0] : props.lon)
    const lat = Number(Array.isArray(coords) ? coords[1] : props.lat)
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue
    if (Math.abs(lon) > 180 || Math.abs(lat) > 90) continue
    const key = `${lon.toFixed(4)},${lat.toFixed(4)}`
    if (seen.has(key)) continue
    seen.add(key)
    points.push([Number(lon.toFixed(5)), Number(lat.toFixed(5))])
  }
  if (points.length <= limit) return points
  points.sort((a, b) => a[0] - b[0] || a[1] - b[1])
  const stride = points.length / limit
  return Array.from({ length: limit }, (_, i) => points[Math.floor(i * stride)])
}

// The baseline is tens of megabytes of GeoJSON; parsed once per warm process.
// It changes at most once per refresh run, and the tile cache's TTL is of the
// same order, so a process-scoped copy loses nothing.
let baseline = null

export async function referencePoints(taxon, limit = REFERENCE_POINT_LIMIT) {
  if (!baseline) baseline = await loadBaseline()
  return pickReferencePoints(baseline?.features || [], taxon, limit)
}
