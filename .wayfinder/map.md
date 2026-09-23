---
label: wayfinder:map
---

# Map: Beta → FRMS Foray Release

## Destination

The app graduates from beta and is ready to demo and promote at the FRMS foray on **27 September 2026**: the EE enrichment pipeline is verified end-to-end against the real API, critical mobile and navigation bugs are fixed, job results survive navigation, and map performance is stable under real usage.

## Notes

Domain: biodiversity / species distribution modelling PWA (Nuxt, Supabase, Google Earth Engine, MaxEnt). FRMS = the society this app serves; members share a pooled EE project. Tens of members expected at launch.

Skills to consult each session: `domain-modeling` (see `CONTEXT.md`), `code-review`.

Standing preferences:
- Critical-path first: VDEBT-1 diagnosis and WANT-11 design gate the most other work.
- DEBT-7/8 VIP for dev ergonomics but post-foray.
- BYOK OAuth deferred to v1.4.
- Execution is in scope for this map (task tickets resolve by doing, not just deciding).

## Decisions so far

- [VDEBT-1: Diagnose EE layer rendering against the real API](.wayfinder/tickets/001-vdebt1-research.md): 4 confirmed code breaks found: bare FIRMS asset ID, WorldCover used as Image not Collection, CANOPY_PCT band name mismatch, fc.toList().getRegion() wrong API. TREEMAP private-path flagged as high-risk.
- [ISSUE-1: Fix LayerManager state persistence on mobile Safari](.wayfinder/tickets/002-issue1-mobile-safari.md): Persisted `activeOverlays`, `overlayOrder`, `layerOpacity` to `localStorage` under `map-overlay-state`; restore on mount for static layers, per-layer for async EE layers.
- [VDEBT-1: Apply EE layer fixes from diagnosis](.wayfinder/tickets/007-vdebt1-fix.md): Fixed 4 confirmed breaks — WorldCover as Image not Collection, CANOPY_PCT→CANOPYPCT band name, getRegion on EE List in ee-assets, bare 'FIRMS' asset ID. TREEMAP private-path access still needs live verification.
- [PERF-4: Debounce map coloring watch](.wayfinder/tickets/003-perf4-coloring-debounce.md): 40ms debounce on the `eachLayer` restyle watch; rapid filter changes now collapse to one pass.
- [PERF-5: Debounce heatmap rebuild](.wayfinder/tickets/004-perf5-heatmap-debounce.md): Converted `heatmapResult` from computed to ref; 50ms debounced watch replaces eager recompute on every filteredData change.
- [ISSUE-3 + PERF-1/2/3: Charts performance improvements](.wayfinder/tickets/008-charts-perf.md): LazyVisible for below-fold GalleryCharts; single-pass bucketing for tempHighLowDist/elevationData; single-pass counter for coverageData; ISSUE-3 already resolved in ScatterChart.
- [WANT-11: Design job→dataset linkage](.wayfinder/tickets/005-want11-design.md): Name at submission; auto-save on completion; extend `useDatasets` with `activate()`/`loadActiveGeojson()`; `?dataset=slug` URL on `/charts` for re-attach; `useObservations.addInlineDataset` as the downstream wire.
- [WANT-11: Implement job→dataset linkage](.wayfinder/tickets/006-want11-implement.md): Auto-save watch in jobs.vue; "Open on charts" on job and dataset rows; `?dataset=slug` mount handler in charts.vue.

## Not yet specified

_All fog cleared. The way to the destination is now fully specified and all tickets are closed._

## Out of scope

- WANT-3 BYOK OAuth — deferred to v1.4
- WANT-5 Navigation overhaul
- WANT-6 Dashboard
- WANT-9 GBIF + iNat unified import
- DEBT-7 TypeScript migration of Netlify backend
- DEBT-8 MushroomMap decomposition
- V12-UI-1/3/4/7/8 UI polish items
- WANT-1 SDM verification against real EE API
- ISSUE-2 GBIF >10 000-record timeout
- ISSUE-4 EE geometry type validation gaps
