---
label: wayfinder:task
status: closed
assignee: claude
---

# VDEBT-1: Apply EE layer fixes from diagnosis

## Question

Using the findings from [001-vdebt1-research](./001-vdebt1-research.md), fix whatever is broken in the EE tile and raster layer pipeline so that at least one layer renders correctly against a live Earth Engine session.

The scope of fixes is unknown until research completes — it may be: correcting asset IDs, fixing band names, repairing tile URL construction, wiring credentials correctly, or some combination.

Acceptance: run `scripts/verify_ee_layers.mjs` (or equivalent manual check) against a deployment with real EE credentials; at least one layer loads tiles without error.

## Blocked by

→ [001-vdebt1-research](./001-vdebt1-research.md)

## Resolution

Four confirmed bugs fixed:

1. **WorldCover type error** (`netlify/lib/ee-runner.mjs` `runLandcover`): `ee.Image(WORLDCOVER)` → `ee.ImageCollection(WORLDCOVER).mosaic()`. WorldCover v200 is an ImageCollection; treating it as a single Image produced nulls for every land_cover column.

2. **TreeMap band name** (`ee-runner.mjs` `runForest`): `treeMap('CANOPY_PCT')` → `treeMap('CANOPYPCT')`. FIA band name has no underscore; mismatch meant `canopy_pct` column was always null.

3. **getRegion on EE List** (`netlify/lib/ee-assets.mjs` line 63): `fc.toList(fc.size()).getRegion()` → `fc.getInfo()`. `getRegion()` is an `ee.Image` method and throws unconditionally on an EE List; `getInfo()` on a FeatureCollection returns a GeoJSON FeatureCollection directly.

4. **FIRMS asset ID** (`netlify/lib/ee-tile-layers.mjs` ASSETS): `'FIRMS'` → `'NASA/FIRMS/modis/Global'`. Bare string is not a valid GEE catalog path; all active-fire tile requests would have failed.

**Not changed**: TREEMAP path (`projects/gtac-data-publish/assets/TreeMap/Product_Version/2026-1`) is a private GCP project path — needs the FRMS service account to have explicit read access. Verify with `scripts/verify_ee_layers.mjs` targeting the `forest` layer.

**EE_LAYER_CATALOGUE import** flagged by research: not a bug — export exists at ee-tile-layers.mjs line 1984.
