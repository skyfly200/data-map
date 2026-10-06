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

## Shipped (all unverified live)

Status: implemented and unit-tested on fixtures. **Not run:** migrations 012-015 applied, Colorado load, RIDB, Overpass limits, PAD-US URL/field names (`PADUS_FEATURE_URL` default and the `Mang_Name`/`Des_Tp`/`Pub_Access` pattern mapping), browser verification. Env vars and the load script: `docs/deploying.md` block 7. Sources to check rules against: `docs/access-sources.md`. User docs: `content/guide/foray.md`.

### Planner (`pages/foray.vue`, phase 1)

- Score (`composables/forayScore.ts`): per cell, sum over in-season species of `weight * in-window finds`, divided by all the cell's finds (after the land-cover filter). Weight = `1 / (1 + (dist + iqr/2) / 14)` from the species' median distance and IQR. Min 3 finds. Effort-neutral. Only cells with finds are scored.
- Time: `Now` = today +/- 14 days; a month = mid-month +/- 15. Also a land-cover filter and cell size (0.01-0.1 deg).
- Modes (`forayPlanner.ts`): Forager (8, bands), Researcher (25, components, sample, caveats), Foray leader (12, switches on, shortlist). Mode only changes defaults and panels. Persisted in `localStorage` (`foray-mode`; `?mode=` overrides).
- Switches: free, public (open, non-private), collecting allowed, include likely (BLM/USFS), keep unknown (non-Forager). Unknown fails a switch unless kept. A cell takes the most restrictive value across covering areas. Switches force off when access status is not `loaded`/`partial`.
- Shortlist: notes (`localStorage` `foray-notes`), CSV, copy text, print. Labels carry `estimated`/`verified`.
- Entry points: `Plan a foray` on the dashboard Now Fruiting card, nav `Foray`, `Foray score` heatmap mode on `/map` (`useMapHeatmaps.ts`).
- Limits: phase 1 only (no ensemble or habitat score); access is matched at the cell **centre point**; `/foray` loads Colorado access once as public PAD-US (no member sets); Colorado is the only region.

### Access data

- Ingest (`netlify/lib/access-ingest.mjs`, `access-regions.mjs`): PAD-US polygons (ArcGIS), OSM roads/trails (Overpass, 0.25 deg tiles), RIDB facility fees (needs `RIDB_API_KEY`). Tables: `access_areas`, `access_lines`, `access_regions` (`docs/schema.md`).
- Estimates only: BLM/USFS fee `free` (estimated); NPS/state park `fee` (estimated); collecting `restricted` for NPS/state park/FWS/SP/WA/WSA, `prohibited` for closed land, `likely_allowed` for open BLM/USFS, else `unknown`. **Never plain `allowed`.** `*_source` is `estimated` (or `ridb` for fees). Owner-asserted values live only in `access_set_areas`.
- Endpoint `netlify/functions/access.mjs` (contract in file header): `GET ?bbox=w,s,e,n` (max 3 deg per side, 400 areas), filters `free=1`, `public=1`, `collecting=<v>`, `lines=1` (bbox <= 0.5 deg, 3000 lines, else `lines_omitted`), `include_sets=1` with a bearer token (merges the caller's set areas, `Cache-Control: private`). Returns `loaded`, `regions`, `truncated`. Errors 400/405/503.
- Job `access_ingest`: shares the EE queue; members capped at 5,000 km2 and 3 loads per month (`quotas.mjs`); admins bypass. `scripts/load-access-region.mjs` for Colorado.

### Map layer and My areas

- `components/AccessLayerPanel.vue` (layer window section; mobile layer sheet): enable, color by public/fee/collecting, free/public/collecting/likely filters, sources (Public lands, My areas, Club areas), legend, disclaimer. Zoom >= 8 for areas, >= 12 for lines; view tiled into 3 deg requests (max 9), clamped to Colorado.
- `/areas` (members): sets (user or club), draw polygon or import GeoJSON, per-area fee/collecting/notes (owner-asserted), clubs with owner/admin/member roles. Contract and caps in `netlify/functions/access-sets.mjs` header.
