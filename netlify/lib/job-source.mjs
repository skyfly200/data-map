// Which observations a job will run over.
//
// A member picks either an area with some filters, or a dataset they already
// have. Both resolve to a list of GeoJSON features here, so the pipeline and
// the cost estimate see one shape.
//
// Selecting is pure and separately testable; loading is the part that touches
// storage. They are split because the selection rules — what counts as inside
// the box, how a date range is applied — are what a member will argue with, and
// those should be checkable without a Supabase project.

import { loadBaseline } from './baseline.mjs'
import { readJson } from './datasets-store.mjs'

/** Is a feature inside the box? Handles a box that wraps the antimeridian. */
export function withinBounds(feature, bounds) {
  const co = feature?.geometry?.coordinates
  if (!co || co.length < 2) return false
  const lon = Number(co[0])
  const lat = Number(co[1])
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return false
  if (lat < bounds.south || lat > bounds.north) return false
  // normaliseBounds pushes east past 180 for a wrapped box, so a point at -179
  // has to be tried at +181 as well to be found inside it.
  if (lon >= bounds.west && lon <= bounds.east) return true
  return bounds.east > 180 && lon + 360 >= bounds.west && lon + 360 <= bounds.east
}

/**
 * The features a bbox source selects.
 *
 * `taxon` matches at any rank, so "Agaricales" selects an order and "Morchella"
 * a genus without the member having to say which they meant.
 */
export function selectFeatures(features, source) {
  const { bounds, dateFrom, dateTo, taxon } = source
  const needle = (taxon || '').trim().toLowerCase()
  return features.filter((f) => {
    if (!withinBounds(f, bounds)) return false
    const props = f.properties || {}
    const date = (props.date || '').slice(0, 10)
    if (dateFrom && (!date || date < dateFrom)) return false
    if (dateTo && (!date || date > dateTo)) return false
    if (needle) {
      const ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
      const hit = ranks.some((r) => String(props[r] || '').toLowerCase() === needle)
      if (!hit) return false
    }
    return true
  })
}

/** Distinct observation dates, which is what the dated stages cost per. */
export function countDates(features) {
  const dates = new Set()
  for (const f of features) {
    const d = (f?.properties?.date || '').slice(0, 10)
    if (d) dates.add(d)
  }
  return dates.size || 1
}

/** Resolve a normalised spec's source to features. */
export async function loadSource(source) {
  if (source.type === 'dataset') {
    const data = await readJson(`${source.slug}.geojson`)
    if (!data) throw new Error(`Dataset “${source.slug}” could not be read.`)
    return data.features || []
  }
  const baseline = await loadBaseline()
  return selectFeatures(baseline?.features || [], source)
}

/** Points and dates a spec covers, for pricing it before it runs. */
export async function measureSource(spec) {
  const features = await loadSource(spec.source)
  return { points: features.length, dates: countDates(features) }
}
