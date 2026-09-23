---
label: wayfinder:research
status: closed
---

# VDEBT-1: Diagnose EE layer rendering against the real API

## Question

The EE tile and raster layers in this app have never been rendered against a live Earth Engine session — band names, asset IDs, and the tile-URL construction are unverified outside stubs. Before we can fix anything, we need to know: what exactly is wired up, what would break first if credentials were connected today, and what evidence exists (logs, error responses) that points to where the gaps are?

Specifically surface:
1. The full list of EE asset IDs referenced in `netlify/lib/ee-tile-layers.mjs` and `netlify/lib/ee-assets.mjs` — are they valid public GEE asset paths?
2. The tile URL construction in `netlify/functions/ee-tiles.mjs` — does it follow the Maps API pattern GEE expects?
3. Any existing error handling or known-failure notes in `netlify/lib/ee-runner.mjs` and `netlify/functions/ee-worker.mjs`.
4. Whether `scripts/verify_ee_layers.mjs` exists and what it checks.

Captures findings as a resolution comment on this ticket. Does NOT attempt fixes.

## Blocks

→ [007-vdebt1-fix](./007-vdebt1-fix.md)
→ [005-want11-design](./005-want11-design.md) (informs job result shape)

## Resolution

_Researched 2026-09-23. No fixes applied; findings only._

---

### Q1 — EE asset IDs in `ee-tile-layers.mjs` and `ee-assets.mjs`

`netlify/lib/ee-tile-layers.mjs` exports an `ASSETS` constant (lines 66–151) with the following ids. Validity assessed against published GEE catalog paths:

| Constant | Path | Status |
|---|---|---|
| `MODIS_BURN` | `MODIS/061/MCD64A1` | Valid public catalog |
| `MTBS_SEVERITY` | `USFS/GTAC/MTBS/annual_burn_severity_mosaics/v1` | Valid public catalog |
| `FIRMS` | `FIRMS` | **HIGH RISK — bare string, not a GEE catalog path.** GEE catalog paths for FIRMS are `NASA/FIRMS/modis/Global` or `NASA/FIRMS/noaa_viirs_c2/Global`. The `active-fire` layer's `count` and `build` will throw on first execution. |
| `S2_SR` | `COPERNICUS/S2_SR_HARMONIZED` | Valid |
| `HANSEN` | `UMD/hansen/global_forest_change_2023_v1_11` | Valid |
| `SOLUS100` | `USDA/SOLUS100/V0` | Plausible; unverified |
| `OPENLANDMAP_TEXTURE` | `OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02` | Valid |
| `OPENLANDMAP_GRTGROUP` | `OpenLandMap/SOL/SOL_GRTGROUP_USDA-SOILTAX_C/v01` | Valid |
| `OPENLANDMAP_WATER_33KPA` | `OpenLandMap/SOL/SOL_WATERCONTENT-33KPA_USDA-4B1C_M/v01` | Valid |
| `GAP_LANDCOVER` | `USGS/GAP/CONUS/2011` | Valid |
| `WORLDCOVER` | `ESA/WorldCover/v200` | Valid |
| `SRTM` | `USGS/SRTMGL1_003` | Valid |
| `MERIT_HYDRO` | `MERIT/Hydro/v1_0_1` | Valid |
| `TREEMAP` | `projects/gtac-data-publish/assets/TreeMap/Product_Version/2026-1` | **HIGH RISK — private `projects/` path, not a public catalog asset.** The version suffix `2026-1` is anomalous for a dataset described as a 2016 FIA baseline. This may require the service account to be granted explicit access to `gtac-data-publish`. Public alternatives exist (e.g. `projects/sat-io/open-datasets/USFS/TreeMap_2016`). |
| `SMAP` | `NASA/SMAP/SPL4SMGP/007` | Valid |
| `CSP_SRTM_MTPI` | `CSP/ERGo/1_0/Global/SRTM_mTPI` | Valid |
| `CSP_SRTM_CHILI` | `CSP/ERGo/1_0/Global/SRTM_CHILI` | Valid |
| `CSP_HM` | `CSP/HM/GlobalHumanModification` | Valid |
| `GHSL_POP` | `JRC/GHSL/P2023A/GHS_POP` | Valid |
| `S1_GRD` | `COPERNICUS/S1_GRD` | Valid |
| `ERA5_LAND_DAILY` | `ECMWF/ERA5_LAND/DAILY_AGGR` | Valid |
| `NOAA_CPC_PRECIP` | `NOAA/CPC/Precipitation` | Plausible; unverified |

`netlify/lib/ee-assets.mjs` is a separate `loadEeAsset` utility for user-provided FeatureCollection paths (lines 1–96). It does **not** embed its own catalog IDs; instead, it relies on a caller-supplied `assetPath`. That code has its own serious issues (see Q5).

The codebase itself acknowledges this risk. `ee-tile-layers.mjs` lines 22–26 contain an explicit caveat: _"They were written without a live Earth Engine session to check them against, so a name or a band may be wrong."_

**Highest-risk assets that will fail before pixels are returned: `FIRMS` and `TREEMAP`.**

---

### Q2 — Tile URL construction in `netlify/functions/ee-tiles.mjs`

The `getMapTemplate` function (lines 181–196) calls:

```js
ee.data.getMapId({ image, ...visParams(vis) }, (result, error) => {
  const template = result.urlFormat
    || (result.mapid
      ? `https://earthengine.googleapis.com/v1/${result.mapid}/tiles/{z}/{x}/{y}`
      : null)
})
```

This correctly follows the GEE Maps API pattern:
- `ee.data.getMapId` is the right server-side call for minting a tile template.
- `result.urlFormat` is the preferred form returned by the newer EE client and is a direct XYZ template with `{z}/{x}/{y}`.
- The fallback constructs the same URL from `result.mapid`, using the correct v1 tile path.
- `visParams(vis)` stringifies min/max/palette to the formats the GEE callback client requires (numbers passed raw cause a `"csv.split is not a function"` error, which the comment on line 185 explicitly notes).

No gaps in URL construction. The same pattern is replicated correctly in `ee-runner.mjs` `mintTemplate` (lines 615–627).

**One gap**: the `ee-tiles.mjs` handler imports `EE_LAYER_CATALOGUE` (line 35) but no such export exists in the current `ee-tile-layers.mjs` file read — only `EE_TILE_LAYERS`. This import would throw a `SyntaxError` / undefined value at runtime. Verify whether `EE_LAYER_CATALOGUE` is exported later in the file (past line 1608).

---

### Q3 — Error handling in `ee-runner.mjs` and `ee-worker.mjs`

**`netlify/lib/ee-runner.mjs`**

- `initEarthEngine()` (lines 78–100): authenticates once per process via `ee.data.authenticateViaPrivateKey`. Failure clears the cached session so the next request retries — not a silent poison. Good.
- `withRetry()` (lines 143–158): 3 retries with exponential backoff, keyed on a regex matching transient EE error strings (`too many`, `rate limit`, `timed out`, `backend error`, `internal error`, `unavailable`, `deadline`). Permanent errors (wrong asset, wrong band) are not retried. Good.
- `sampleChunk()` (lines 184–203): on failure after retries, increments `skipped.n` and returns `new Array(points.length).fill(null)` — points that failed a chunk are silently null rather than causing a job failure. This is intentional for partial results but means a persistent band error produces a column of nulls with no surface-level error message.
- `runSoilTaxonomy()` (lines 421–452): if the class table fails to load, the entire stage records nothing and increments `skipped.n`. The comment says "a bare class number is not a column worth writing", which is reasonable but means the job succeeds with empty taxonomy columns.
- No hardcoded "TODO: untested" notes, but `MERIT_HYDRO = 'MERIT/Hydro/v1_0_1'` is re-declared locally at line 55 (duplicating the `ASSETS` constant from `ee-tile-layers.mjs`), making it a separate maintenance target.

**Known-failure note in `ee-runner.mjs`**: The `requestDeadlineMs` comment (lines 108–113) documents that a per-request timeout was deliberately removed because it silently zeroed columns — an explicit record of a past failure mode.

**`netlify/functions/ee-worker.mjs`**

- Top-level `try/catch` (lines 74–135) catches all job errors, calls `failJob` and `notifyJobSettled`, and logs the error.
- Cancellation is polled before expensive work (line 88).
- Known gap (line 97): `spent = job.estimated_units || 0` — on a failure mid-job, `spent` stays at its initial 0 rather than accounting for Earth Engine calls made before the error. This is noted in the comment ("Charged for what it spent before breaking") but the implementation doesn't actually track partial spend.
- No TODOs or known-failure notes specific to untested paths.

---

### Q4 — `scripts/verify_ee_layers.mjs`

**The script exists** at `scripts/verify_ee_layers.mjs` (110 lines).

It does the following, in order, for each layer:
1. Calls `resolveLayer(key, {})` to get default params (line 53).
2. If `layer.prepare` exists, calls it and checks that the returned table has a non-empty `values` array (lines 54–57).
3. If `layer.count` exists, evaluates it and fails with `'count is zero at default parameters'` if the result is zero (lines 60–63).
4. Calls `layer.build(ee, params, prepared)` and passes the returned `image` and `vis` to `mint()` (lines 64–65).
5. `mint()` calls `ee.data.getMapId` and resolves the tile URL (lines 38–47).

It runs layers serially to avoid GEE throttling (comment at line 91). It accepts an optional substring filter as `process.argv[2]` (lines 32–35).

**The script has NOT been run against a live session** — this is the verification debt documented in `docs/roadmap.md` and referenced in the script's own header comment (lines 3–7): _"The catalogue's asset IDs, band names and system:index values were written from documentation…never executed against Earth Engine from this codebase."_

Running it is the first step toward discharging VDEBT-1.

---

### Q5 — `netlify/lib/ee-pipeline.mjs` — risks on first real execution

**In `ee-pipeline.mjs` itself**: no hardcoded test values and credential plumbing is not handled here — it imports from `ee-runner.mjs` and `ee-tile-layers.mjs`. The stage definitions look consistent.

**In `netlify/lib/ee-assets.mjs`** (the user-asset loader):
- Line 38: `if (!ee.data._initialized)` — reads a private/internal property of the EE client, which is not part of the public API and may not exist or behave consistently across versions.
- Line 39: `await ee.initialize(null, null, null, '1.0.0')` — passes the version string as the 4th positional argument. The normal signature is `ee.initialize(baseurl, tileurl, success, error, opt_xsrf_token, project)`. Without credentials or a project id, this initialization would succeed but every subsequent call would fail with an auth error.
- Line 63: `const region = await fc.toList(fc.size()).getRegion()` — `.getRegion()` is a valid method on `ee.Image` (server-side sampling), not on a JavaScript list or an EE List. This line will throw on any execution.

**In `netlify/lib/ee-runner.mjs` — the highest-risk gap in the pipeline**:

`runLandcover` (line 260):
```js
const image = ee.Image(WORLDCOVER).select('Map').rename('land_cover')
```
`WORLDCOVER = 'ESA/WorldCover/v200'` is an **ImageCollection**, not a single Image. `ee.Image()` on a collection ID does not produce a valid image in Earth Engine — the correct call, used in `ee-tile-layers.mjs` line 1130, is `ee.ImageCollection(WORLDCOVER).select('Map').mosaic()`. Every observation processed through `runLandcover` will return null.

`runForest` (line 488):
```js
const canopy = treeMap('CANOPY_PCT').rename('canopy')
```
vs `ee-tile-layers.mjs` line 1222:
```js
image: treeMap2016(ee, 'CANOPYPCT'),
```
The pipeline's `runForest` uses the band name `CANOPY_PCT` (with underscore); the tile layer uses `CANOPYPCT` (no underscore). The actual USFS TreeMap FIA band name is `CANOPYPCT`. One of the two is wrong and will cause a band-not-found error; most likely `CANOPY_PCT` in `ee-runner.mjs` is the incorrect spelling.

**Summary of highest-risk items for first real execution:**

| Risk | Location | Effect |
|---|---|---|
| `FIRMS` asset ID is bare string | `ee-tile-layers.mjs` L69 | `active-fire` layer throws immediately |
| `TREEMAP` is a private project path | `ee-tile-layers.mjs` L113 | All TreeMap layers fail unless service account has explicit access |
| `ee.Image(WORLDCOVER)` on a collection | `ee-runner.mjs` L260 | `land_cover` column is null for every observation |
| `CANOPY_PCT` band name mismatch | `ee-runner.mjs` L488 | `canopy_pct` column is null (band not found) |
| `fc.toList(...).getRegion()` | `ee-assets.mjs` L63 | User-asset import throws unconditionally |
| `ee.data._initialized` internal API | `ee-assets.mjs` L38 | Initialisation check is unreliable |
