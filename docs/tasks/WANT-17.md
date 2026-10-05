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
4. **Access layer.** Ingest PAD-US (public land boundaries and access class) and OSM roads/trails as scorable data; apply as a mask in phases 1–3. Distance-to-road/trail and public-land status per cell; PAD-US gives legal access, OSM gives physical access.
5. **Under-sampled opportunity.** Find accessible, promising cells with few or no finds. Promise = phase 2/3 score; sampling effort = observation density plus observer-reach proxy from OSM trail/road distance. Rank by high promise ÷ low effort, among cells passing the access mask. This is the one place access and effort are used as signals, not just a filter: the aim is to counter the observer bias that phase 1 inherits. Guard against "promising" being an extrapolation artifact (cells outside the models' training range, or with no finds because the habitat is truly absent): flag cells whose predictors fall outside the training envelope, and report model AUC next to each suggestion.

## Open decisions

- Start phase (1 is shippable now; 3 needs schema work).
- Access source: decided — PAD-US + OSM roads and trails (replaces the gHM proxy). Open: ingest path (EE asset vs. preprocessed vector tiles) and OSM refresh cadence.
- Under-sampled ranking: how to weight promise vs. effort, and the minimum training-envelope check.

## Access data layer (implemented, unverified live)

Migration 013, `netlify/lib/access-ingest.mjs` (estimates + RIDB overlay), `access-regions.mjs` (`access_ingest` job, region skip), `quotas.mjs` (access cost model), `netlify/functions/access.mjs` (read contract in file header), `scripts/load-access-region.mjs` (Colorado). Not run live: Colorado load, RIDB (needs RIDB_API_KEY), Overpass limits, PAD-US layer URL/field names (PADUS_FEATURE_URL default unverified), migration application. Member jobs are capped at 5000 km2 to fit the 300s worker; Colorado uses the script.
