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

// The taxon-name rules live in dataset-taxa, which is pure: the map imports
// them too, and this module reaches node:fs and Supabase.
import { matchesTaxon } from './dataset-taxa.mjs'
import { loadBaseline } from './baseline.mjs'
import { readJson } from './datasets-store.mjs'
import { DatasetAccessError, resolveDataset } from './dataset-access.mjs'

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

/** Is a feature inside the date range? An undated record is outside any range. */
export function withinDates(feature, { dateFrom, dateTo } = {}) {
  if (!dateFrom && !dateTo) return true
  const date = (feature?.properties?.date || '').slice(0, 10)
  if (!date) return false
  if (dateFrom && date < dateFrom) return false
  if (dateTo && date > dateTo) return false
  return true
}

/**
 * The features a bbox source selects, and what each filter removed.
 *
 * The counts are the point. "No observations to enrich" is true of an area with
 * nothing in it, of a date range before the data starts, and of a taxon the
 * dataset does not carry — and those are three different mistakes with three
 * different fixes. Counting each stage is what lets the refusal say which.
 */
export function explainSelection(features, source) {
  const inBounds = features.filter((f) => withinBounds(f, source.bounds))
  const inDates = inBounds.filter((f) => withinDates(f, source))
  const selected = inDates.filter((f) => matchesTaxon(f.properties, source.taxon))
  return {
    total: features.length,
    inBounds: inBounds.length,
    inDates: inDates.length,
    selected,
  }
}

/** The features a bbox source selects. */
export function selectFeatures(features, source) {
  return explainSelection(features, source).selected
}

export { matchesTaxon }

/** Distinct observation dates, which is what the dated stages cost per. */
export function countDates(features) {
  const dates = new Set()
  for (const f of features) {
    const d = (f?.properties?.date || '').slice(0, 10)
    if (d) dates.add(d)
  }
  return dates.size || 1
}

/**
 * Resolve a normalised spec's source to features.
 *
 * A dataset source is resolved through `resolveDataset`, which answers for the
 * viewer rather than for the server. That matters because this runs with the
 * service role, which row-level security does not apply to: without the check
 * here, a spec naming any slug at all would read that dataset, and slugs are
 * short and guessable. The path comes off the resolved row rather than being
 * built from the slug, so a row is the only way to name a stored file.
 *
 * `viewer` is required for a dataset source and ignored for a bbox, which
 * reads the public baseline.
 */
export async function loadSource(source, { client = null, viewer = null, read = readJson } = {}) {
  if (source.type === 'dataset') {
    const row = await resolveDataset({ client, slug: source.slug, viewer })
    const data = await read(row.path)
    if (!data) {
      // The row exists and they may read it, but the file behind it is gone.
      // Distinct from "no such dataset" because the fix is different and this
      // one is ours, not theirs.
      throw new DatasetAccessError(
        `Dataset "${row.slug}" is registered but its file could not be read.`,
        { status: 500, code: 'no_file' })
    }
    return data.features || []
  }
  const baseline = await loadBaseline()
  return selectFeatures(baseline?.features || [], source)
}

/** Points and dates a spec covers, for pricing it before it runs. */
export async function measureSource(spec, access = {}) {
  if (spec.source?.type === 'dataset') {
    const features = await loadSource(spec.source, access)
    return { points: features.length, dates: countDates(features) }
  }
  // A bbox, which is the case that can come back empty for three unrelated
  // reasons. Carry the breakdown so the refusal can say which one.
  const baseline = await loadBaseline()
  const seen = explainSelection(baseline?.features || [], spec.source)
  return {
    points: seen.selected.length,
    dates: countDates(seen.selected),
    breakdown: { total: seen.total, inBounds: seen.inBounds, inDates: seen.inDates },
  }
}

/**
 * Why a bbox selected nothing, in words the member can act on.
 *
 * Each branch names the filter that emptied the set and the count that survived
 * the one before it, because that number is the whole difference between "look
 * somewhere else" and "ask for a different taxon".
 */
export function explainEmpty(source = {}, breakdown = null) {
  const n = (v) => Number(v || 0).toLocaleString()
  if (!breakdown) return 'That area and date range contain no observations to enrich.'
  if (!breakdown.total) {
    // Not the member's fault: the dataset the box is matched against is the one
    // bundled with the deployment, and it did not load.
    return 'The observation dataset could not be read, so there is nothing to match that area '
      + 'against. That is a fault on our side rather than in what you asked for.'
  }
  if (!breakdown.inBounds) {
    return 'No observations fall inside that area. Try "Use the current map view" with the map '
      + 'over somewhere the points are.'
  }
  if (!breakdown.inDates) {
    const from = source.dateFrom || 'the start'
    const to = source.dateTo || 'now'
    return `${n(breakdown.inBounds)} observations are in that area, but none of them are dated `
      + `between ${from} and ${to}. Widen the date range.`
  }
  return `${n(breakdown.inDates)} observations are in that area and date range, but none of them `
    + `match the taxon "${source.taxon}". Clear the taxon to enrich all of them, or pick one the `
    + 'dataset actually carries.'
}
