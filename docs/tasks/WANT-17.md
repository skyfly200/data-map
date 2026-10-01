# WANT-17 Foray planner

Extend "Now Fruiting" from "what is fruiting" to "where to go look": score areas by in-season species, model suitability, and shared habitat variables, filtered by access and land type.

## Findings (blockers)

- **Variable contributions are not persisted.** `scoutContributions` / `contributions` in `pages/model.vue` are page state; nothing in `docs/schema.md`, `useMaxEnt.ts`, or `maxent.mjs` stores them on the model row. Cross-model "which variables matter" needs a schema change + worker write-back (+ backfill by re-running models).
- **Accessibility is display-only.** Trails (waymarkedtrails) and land ownership (BLM SMA) in `mapLayers.ts` are third-party raster tiles — drawable, not scorable. Only gHM (`ASSETS.CSP_HM`, 90 m) is an EE image. Scoring access needs a sampleable asset (PAD-US, OSM roads/trails ingest).
- **Bias.** Finds correlate with access. Use accessibility as a filter/mask, never added to the score; use effort-neutral `season`, not `density`. See `WANT-7`.

## Phases

1. **Foray score heatmap mode (client-only).** Move in-season species logic out of `DashboardNowFruiting.vue` into a composable. New mode in `useMapHeatmaps.ts` via `buildCells`: per-cell `season` share for in-season species, weighted by phenology closeness (`dist`, `iqr`), filtered by selected `land_cover_label`. Only covers cells that already have finds.
2. **Model ensemble (EE).** Weighted sum of saved suitability surfaces for in-season species with models, next to `buildSuitabilityImage` in `netlify/lib/maxent.mjs`, served via `model-tiles`. Weights from phenology. Access as mask/threshold. Needs a species-set job spec.
3. **Species-agnostic habitat score.** Persist per-model contributions (see blockers). Normalize each model's contributions over only its own predictors, average across in-season models. Importance has no direction: also store preferred range per predictor (e.g. IQR of the field across in-season observations via `fieldValue`). Score a cell by weighted fraction of top predictors inside range.
4. **Access layer.** Ingest a scorable access asset; apply as mask in phases 1–3.

## Open decisions

- Start phase (1 is shippable now; 3 needs schema work).
- Access source: PAD-US (public land) vs OSM roads/trails vs gHM proxy.
