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
  return [
    min === null || min === undefined ? lo : Math.max(lo, min),
    max === null || max === undefined ? hi : Math.min(hi, max),
  ]
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
