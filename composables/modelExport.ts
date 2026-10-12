// Choices for exporting a model's suitability surface as a GeoTIFF.
//
// Earth Engine refuses a direct download over about 48 MB. The surface is one
// float band (4 bytes a pixel), so the ceiling is roughly twelve million
// pixels; a resolution that would pass it is offered disabled, with the reason,
// rather than left to fail after a wait.

export const MODEL_EXPORT_SCALES = [
  { metres: 90, label: '90 m', note: 'Fine detail, small areas only' },
  { metres: 250, label: '250 m', note: 'Good for a county or a range' },
  { metres: 1000, label: '1 km', note: 'Regional overview' },
  { metres: 5000, label: '5 km', note: 'Whole-state sketch' },
]

export const MAX_EXPORT_PIXELS = 12_000_000

export interface Bounds { west: number; south: number; east: number; north: number }

/** Pixels a region covers at a scale, or null when the region is unknown. */
export function exportPixels(region: Bounds | null | undefined, metres: number): number | null {
  if (!region || ![region.west, region.south, region.east, region.north].every(Number.isFinite)) return null
  const midLat = ((region.north + region.south) / 2) * Math.PI / 180
  const width = Math.abs(region.east - region.west) * 111_320 * Math.cos(midLat)
  const height = Math.abs(region.north - region.south) * 110_574
  return Math.ceil(width / metres) * Math.ceil(height / metres)
}

/** Smallest scale that fits under the pixel ceiling (the coarsest if none do). */
export function defaultExportScale(region: Bounds | null | undefined): number {
  const fits = MODEL_EXPORT_SCALES.find((s) => {
    const px = exportPixels(region, s.metres)
    return px == null ? s.metres >= 1000 : px <= MAX_EXPORT_PIXELS
  })
  return (fits || MODEL_EXPORT_SCALES[MODEL_EXPORT_SCALES.length - 1]).metres
}
