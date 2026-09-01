// Field registry shared by the Explore builder and ChartRenderer.
// `unit` marks fields whose display follows the ft/m or °F/°C toggle.
//
// `bounds: [min, max]` is the range a value can physically take. Charts pad
// their axes past the data so marks are not flush against the frame, and
// without a limit that padding invents impossible readings — a compass aspect
// of 0–360° drew an axis to 378°, day-of-year ran past 365, and anything
// non-negative picked up a negative tail. `null` on either end means unbounded
// in that direction.

export const ALL_NUMERIC = [
  { key: 'elevation', label: 'Elevation', unit: 'elev', bounds: [0, null] },
  { key: 'day_of_year', label: 'Day of year', bounds: [1, 366] },
  { key: 'year', label: 'Year', bounds: [0, null] },
  { key: 'month', label: 'Month (1–12)', bounds: [1, 12] },
  { key: 'tmax', label: 'High temp', unit: 'temp' },
  { key: 'tmin', label: 'Low temp', unit: 'temp' },
  { key: 'tavg', label: 'Avg temp', unit: 'temp' },
  { key: 'rain7', label: '7-day rain total (mm)', bounds: [0, null] },
  { key: 'ndvi', label: 'NDVI', bounds: [-1, 1] },
  { key: 'ndmi', label: 'NDMI', bounds: [-1, 1] },
  { key: 'soil_moisture', label: 'Soil moisture', bounds: [0, 1] },
  { key: 'solar_exposure', label: 'Solar exposure', bounds: [0, 1] },
  { key: 'wind_exposure', label: 'Wind exposure', bounds: [0, 1] },
  { key: 'water_retention', label: 'Wetness index (TWI)', bounds: [0, 1] },
  { key: 'slope', label: 'Slope (°)', bounds: [0, 90] },
  { key: 'aspect', label: 'Aspect (°)', bounds: [0, 360] },
]

const BOUNDS = new Map(ALL_NUMERIC.filter((f) => f.bounds).map((f) => [f.key, f.bounds]))

/** Physical bounds for a field key, or null when it has none. */
export function boundsFor(key) {
  return BOUNDS.get(key) || null
}

/**
 * Clamp a padded axis domain to what the quantity can actually be.
 *
 * Falls back to the data's own sign when the field is unknown: a series with no
 * negative values should not get a negative axis just from padding.
 */
export function clampDomain([lo, hi], bounds, values) {
  let min = bounds?.[0]
  let max = bounds?.[1]
  if (min === undefined && Array.isArray(values) && values.length && values.every((v) => v >= 0)) {
    min = 0
  }
  let outLo = min === null || min === undefined ? lo : Math.max(lo, min)
  let outHi = max === null || max === undefined ? hi : Math.min(hi, max)

  // Clamping can collapse the range to zero width — an empty chart falls back to
  // [0, 1], and a field whose floor is 1 (day of year) pins both ends to 1. A
  // zero-width domain divides by zero in every scale function and paints the SVG
  // with NaN coordinates, so widen it back out, staying inside the bounds.
  if (!(outHi > outLo)) {
    const span = Math.max(1e-6, Math.abs(outLo) * 0.01 || 1)
    if (max === null || max === undefined || outLo + span <= max) outHi = outLo + span
    else outLo = outHi - span
  }
  return [outLo, outHi]
}

export const ALL_CATEGORY = [
  { key: 'species', label: 'Species' },
  { key: 'genus', label: 'Genus' },
  { key: 'land_cover_label', label: 'Land cover' },
  { key: 'cluster', label: 'Cluster' },
  { key: 'live_cluster', label: 'Live cluster' },
  { key: 'year', label: 'Year' },
  { key: 'month_name', label: 'Month' },
  { key: 'enrichment_level', label: 'Enrichment level' },
]


// ─── Category ordering ───────────────────────────────────────────────────────
// How the categories of a grouped chart are laid out. Sorting by value answers
// "which is biggest"; sorting by label answers "what is X" — different
// questions, and only the reader knows which one they are asking.

export const SORT_MODES = [
  { key: 'value-desc', label: 'Largest first' },
  { key: 'value-asc', label: 'Smallest first' },
  { key: 'label-asc', label: 'Label A–Z' },
  { key: 'label-desc', label: 'Label Z–A' },
]

/**
 * Order grouped chart entries. Returns a new array; the input is left alone.
 *
 * `valueOf` reads the number a "by value" sort should use — the measure for a
 * bar, the sample count for a box plot.
 */
export function sortEntries(entries, mode, valueOf = (e) => e.value) {
  const list = [...(entries || [])]
  // `numeric` so "Cluster 10" sorts after "Cluster 9" rather than before it.
  const byLabel = (a, b) => String(a.label).localeCompare(String(b.label), undefined, { numeric: true })
  switch (mode) {
    case 'value-asc': return list.sort((a, b) => valueOf(a) - valueOf(b))
    case 'label-asc': return list.sort(byLabel)
    case 'label-desc': return list.sort((a, b) => byLabel(b, a))
    default: return list.sort((a, b) => valueOf(b) - valueOf(a))
  }
}
