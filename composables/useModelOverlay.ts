// A suitability surface handed from the jobs page to the map.
//
// A model job's result is a tile template, not a dataset of points, so it cannot
// ride the addInlineDataset path the enrichment results use. This is the parallel
// for a raster: the jobs page drops the minted template here and routes to the
// map, and the map picks it up on mount, draws it as an overlay with its legend,
// and fits to the region it was projected over.
//
// Nuxt useState rather than a store, for the same reason the inline datasets use
// it: it survives the client-side navigation from /jobs to /map and is scoped to
// this browser, nothing more.

export interface ModelOverlay {
  jobId: string
  label: string
  /** The XYZ tile template Earth Engine minted for the fitted surface. */
  template: string
  /** A ramp legend, as the map's other layers carry: { type, unit, min, max, stops }. */
  legend: { type: string; unit?: string; min?: string; max?: string; stops: string[] }
  /** The bounding box the surface was projected over, so the map can fit to it. */
  region: { north: number; south: number; east: number; west: number } | null
  /** When the template was minted — a map id expires, so age is worth showing. */
  mintedAt: string | null
  /** The cross-validation score, or null when the model was not scored. */
  cv: { auc: number; sd: number; folds: number; grade: string } | null
}

export function useModelOverlay() {
  const pending = useState<ModelOverlay | null>('model-overlay', () => null)

  function show(overlay: ModelOverlay) {
    pending.value = overlay
  }

  function clear() {
    pending.value = null
  }

  return { pending, show, clear }
}
