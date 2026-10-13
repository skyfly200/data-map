// Bulk GBIF fungi for the main observations dataset (scripts/load-gbif-fungi.mjs).
//
// The app's observations come from iNaturalist, and GBIF already mirrors
// iNaturalist research-grade records, so the useful part of GBIF is everything
// else: herbarium specimens, Mushroom Observer and other survey datasets. Pure
// helpers here; the script does the network and storage work.

/** GBIF backbone key for kingdom Fungi. */
export const FUNGI_TAXON_KEY = 5
/** GBIF's copy of iNaturalist research-grade observations; we already have these from iNat. */
export const INAT_DATASET_KEY = '50c9509d-22c7-4a22-a47d-8c48425ef4a7'
export const GBIF_SEARCH = 'https://api.gbif.org/v1/occurrence/search'
export const PER_PAGE = 300
/** The search API refuses offset + limit beyond this; larger queries are split by year. */
export const SEARCH_WINDOW = 100_000
export const DEFAULT_MAX_UNCERTAINTY_M = 5000

/** Search params for one query over a bbox [w, s, e, n] and an inclusive year range. */
export function searchParams({ bbox, years, taxonKey = FUNGI_TAXON_KEY }) {
  const [w, s, e, n] = bbox
  const p = new URLSearchParams({
    taxonKey: String(taxonKey),
    hasCoordinate: 'true',
    hasGeospatialIssue: 'false',
    occurrenceStatus: 'PRESENT',
    decimalLatitude: `${s},${n}`,
    decimalLongitude: `${w},${e}`,
  })
  if (years) p.set('year', `${years[0]},${years[1]}`)
  return p
}

/**
 * Year ranges each small enough for the search window. `count(years)` returns
 * the record count for a range; ranges are halved until they fit (a single
 * year over the window is kept and truncated, which GBIF data never hits for
 * one state).
 */
export async function partitionYears(count, [from, to], window = SEARCH_WINDOW) {
  const n = await count([from, to])
  if (n === 0) return []
  if (n <= window || from === to) return [{ years: [from, to], count: n }]
  const mid = Math.floor((from + to) / 2)
  return [...await partitionYears(count, [from, mid], window), ...await partitionYears(count, [mid + 1, to], window)]
}

function dayOfYear(dateStr) {
  if (!dateStr) return null
  const d = new Date(`${dateStr}T00:00:00Z`)
  if (Number.isNaN(d.getTime())) return null
  return Math.floor((d.getTime() - Date.UTC(d.getUTCFullYear(), 0, 0)) / 86400000)
}

/** iNaturalist observation id named by an occurrenceID URL, if any. */
export function inatIdFrom(occurrenceID) {
  const m = /inaturalist\.org\/observations\/(\d+)/.exec(String(occurrenceID || ''))
  return m ? Number(m[1]) : null
}

/** Why a record is dropped, or null to keep it. */
export function dropReason(rec, { maxUncertaintyM = DEFAULT_MAX_UNCERTAINTY_M, minYear = null } = {}) {
  if (rec.datasetKey === INAT_DATASET_KEY || inatIdFrom(rec.occurrenceID)) return 'inat'
  if (rec.decimalLatitude == null || rec.decimalLongitude == null) return 'no-coords'
  if (!(rec.species || rec.genus)) return 'no-name'
  const u = rec.coordinateUncertaintyInMeters
  if (u != null && maxUncertaintyM != null && u > maxUncertaintyM) return 'uncertainty'
  if (minYear != null && !(rec.year >= minYear)) return 'year'
  // Records placed on a country or state centroid say nothing about habitat.
  const issues = rec.issues || []
  if (issues.includes('COUNTRY_COORDINATE_MISMATCH') || issues.includes('PRESUMED_SWAPPED_COORDINATE')) return 'coords-suspect'
  return null
}

/** A kept GBIF record as a feature in the observations dataset's shape. */
export function gbifRecordToFeature(rec) {
  const date = rec.eventDate ? String(rec.eventDate).slice(0, 10) : null
  const iso = date && /^\d{4}-\d{2}-\d{2}$/.test(date) ? date : null
  return {
    type: 'Feature',
    geometry: { type: 'Point', coordinates: [rec.decimalLongitude, rec.decimalLatitude] },
    properties: {
      uuid: `gbif:${rec.key}`,
      gbif_id: rec.key,
      source: 'gbif',
      basis_of_record: rec.basisOfRecord || null,
      species: rec.species || rec.genus,
      date: iso,
      day_of_year: dayOfYear(iso),
      location: rec.locality || null,
      coordinate_uncertainty_m: rec.coordinateUncertaintyInMeters ?? null,
      elevation: rec.elevation ?? null,
    },
  }
}

/**
 * Baseline plus the GBIF features it does not already hold. Earlier GBIF
 * features are replaced wholesale so a re-run with tighter filters shrinks the
 * set instead of only growing it.
 */
export function mergeGbif(baseline, gbifFeatures) {
  const kept = (baseline?.features || []).filter((f) => f.properties?.source !== 'gbif')
  const seen = new Set(kept.map((f) => f.properties?.uuid).filter(Boolean))
  const add = []
  for (const f of gbifFeatures) {
    const p = f.properties
    if (seen.has(p.uuid)) continue
    seen.add(p.uuid)
    add.push(f)
  }
  return {
    collection: { ...(baseline || { type: 'FeatureCollection' }), type: 'FeatureCollection', features: [...kept, ...add] },
    added: add.length,
    replaced: (baseline?.features?.length || 0) - kept.length,
  }
}
