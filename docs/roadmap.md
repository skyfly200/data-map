# Roadmap

What is known to be missing, and why. Ordered by what blocks something else first, then by what is most often asked for.

Each entry says what is true today, not just what should be. An item here is a
commitment to an open question; when it is closed the entry added to a commit msg and is deleted here

---

## Blocking

Nothing open.

---

## Wanted

### `WANT-1` Species distribution modeling (the point of the enrichment)

MaxEnt ships end to end: `netlify/lib/maxent.mjs` holds the predictor registry,
background plan, cross-validation, and `buildSuitabilityImage`; the worker runs
`runModel` for a `kind: 'model'` job; `model-tiles` re-mints an expired surface
from the stored spec; the jobs page has an Enrich / Model toggle with predictor
selection; the map legend shows the AUC score, the effort-bias confound note, and
a refresh button when tiles expire.

Open: none of it has run against the real Earth Engine API — tested against a
stub only (see `VDEBT-1`). GeoTIFF export (`V12-MOD-4`) is the heavier follow-up.

Worth exercising once enrichment output is loaded as datasets often enough that
"and then what" is a real question.

### `WANT-2` Member-defined enrichment stages

`STAGES` is six hardcoded entries and `normaliseSpec` refuses any `kind` but
`enrich`, so adding a seventh source is a code change and a deploy. That
allowlist is deliberate — Earth Engine bills the project rather than the caller,
so a member sending EE expressions would be spending FRMS's money on code
nobody read — and any answer here has to keep that property.

It is closer than it looks. `runDated` is already a generic sampler taking
`{scale, reducer, bands, imageFor}`, and soil moisture, rainfall and temperature
are thin wrappers over it. `ee_custom_layers` already validates and stores
exactly the spec such a stage needs: asset id, asset type, band, reducer, date
window. A member-defined stage is that spec pointed at points instead of tiles,
with no EE code execution and the same allowlist posture.

The bespoke runners stay bespoke and should: terrain derives TPI at three radii,
solar and wind exposure and upstream area; NDVI and land cover do cloud and
water masking. Those are not "sample a band".

Worth doing after there is evidence somebody has hit the wall of six stages.
Composing jobs came first because it needed no new Earth Engine surface at all.

### `WANT-3` Member-supplied Earth Engine credentials

Every job runs under one service account against one Cloud project, so all
Earth Engine spend bills FRMS. That single pool is the whole reason `quotas.mjs`
exists: the monthly units, the per-day and concurrency caps, and the admin floor
all divide one budget fairly rather than protect against any one member. A member
who needs more than their share, or who wants to run heavier processing than the
shared budget should carry, has no answer today but "ask an admin to raise the
number", which moves the cost onto FRMS rather than onto them.

The way out is to let a member run their own jobs under their own Earth Engine
project. `initEarthEngine` already takes a service-account key from the
environment; the change is to take the job owner's stored credential instead when
they have one, falling back to the shared account when they do not. A job on a
member's own project spends the member's own Earth Engine budget, so `checkQuota`
skips the shared monthly cap for it — they are metered by Google, not by the
pool — while the point and concurrency caps stay, since those protect the worker
and the throttle, not the budget.

The hard part is not the wiring, it is holding the credential. A Google
service-account key is a long-lived secret that can read and spend against the
member's whole project, so storing one means a service-role-only column
encrypted at rest with a key that is not in the same table, a validation
round-trip that mints one throwaway tile before the key is trusted, and a way to
rotate or revoke it that a member can reach without an admin. The safer shape is
OAuth — the member authorises Nexstrata against their EE project and the app
holds a refresh token rather than a raw key — but that is a consent flow and a
token store rather than a form field, and it is more to build. Either way the
credential never reaches the browser and never appears in a job spec, the same
posture the custom-layer and dataset-access work already took.

There is a licensing edge to name in the UI, not just the code: Earth Engine
distinguishes commercial from noncommercial use per Cloud project, so a member
pointing the app at their own project is asserting their project's terms cover
what they are about to run. The guide already walks through making a Cloud
project and registering it for Earth Engine (`guide/layers`), so the member-facing
half is mostly written.

Worth doing when a member's needs exceed what the shared budget should fund —
the first real "I need my own quota" is the signal, the same way composing jobs
waited for the first real "and then what". Until then the admin floor and a
raised per-member quota cover it.

### `WANT-4` Coverage page, reframed

`/coverage` inventories the **local raster cache** — 25.9 GB of CHIRPS, ERA5 and
NDVI files on disk. As enrichment moves to Earth Engine that cache stops
existing, and the page describes an artifact of the architecture being replaced.

The question that survives the move is field coverage: what fraction of records
actually carry each enrichment column. A mean over the 23% of rows that have
soil moisture is a different claim from a mean over all of them, and nothing in
the app currently says which you are looking at.

Keep the URL, replace the contents, and let the raster inventory go when the
cache does.

### `WANT-5` Navigation and the pages behind it

The nav grew around the tools that existed when each was added, and it shows.
`/jobs` sits in it as a top-level destination, but a job is something you start
and then wait on, not a place you go — it belongs behind the thing that produces
it (the map, a dataset) or under an account menu, not beside Map and Charts. And
the pages that have since become the ones worth landing on are not all in the nav
at all.

Two moves, one question. Remove `/jobs` from the nav (the page stays, reached
from where a job is launched), and add the pages that earn a top-level slot to
it. The open question is which those are, and in what order — the nav is the
app's table of contents, so what is in it is a claim about what the app is for.

### `WANT-7` MaxEnt observation bias correction

Citizen science platforms (iNaturalist, GBIF) produce severe sampling bias because observations cluster near human infrastructure — roads and trails. Uncorrected, MaxEnt learns human travel patterns instead of true ecological niches, producing suitability maps that mirror trail networks rather than habitat.

**Critical rule:** Do not feed human footprint, population density, or trail distance layers as standard MaxEnt environmental covariates. The model treats them as positive habitat preferences (e.g. concluding a species "thrives" on compacted dirt paths). These layers belong only in background sample weighting.

**Strategy A — Bias Grid (recommended):** Build a sampling effort surface by combining trail proximity rasters, human population density (WorldPop or LandScan), and general observation density into a single continuous raster. Pass it to MaxEnt via the `biasfile=sampling_effort.tif` argument. This instructs MaxEnt to draw more pseudo-absence background points near high-traffic areas where observers actually look, and fewer in inaccessible terrain, canceling out human travel bias.

**Strategy B — Target Group Background (TGB) with conspicuousness filtering:** Restrict pseudo-absence background to ecologically and morphologically comparable taxa (e.g. large, charismatic macrofungi — visible boletes and amanitas — that attract the same observers) rather than all species. Programmatically drop records from casual or one-time users; strictly prioritise Research Grade observations to reduce misidentification noise.

`WANT-1` already names this as an open item; this entry spells out the implementation path so it can be tracked and closed on its own. `V12-ETH-1` in Future Enhancements covers the automated thinning and bias file generation that the bias grid strategy requires.

### `WANT-8` Vercel and Netlify cross-compatibility

The backend currently targets Netlify Functions exclusively (`netlify/functions/`, `netlify/lib/`). There is no path to deploying on Vercel without rewriting the serverless layer.

The goal is for the same codebase to build and deploy on either platform without forking. The two surfaces are close but not identical: Netlify Functions use `handler(event, context)` with a `netlify.toml` routing config; Vercel uses `api/` file-based routing with a `vercel.json` config and its own `VercelRequest`/`VercelResponse` types. Environment variable naming conventions also differ by convention (Netlify exports `NETLIFY=true`; Vercel exports `VERCEL=1`).

The practical approach is an adapter layer — a thin request/response normalisation shim that each platform's entry point calls — so the business logic in `netlify/lib/` is platform-agnostic and the adapters are the only platform-specific code. That means:

1. Extract all handler logic into framework-free modules (most of `netlify/lib/` already qualifies).
2. Write a Netlify adapter and a Vercel adapter, each no more than a request unwrap and response wrap.
3. CI should build and lint both targets so neither rots.

A secondary benefit: Vercel's edge runtime and image optimisation improve cold-start latency for the Nuxt SSR layer, which Netlify's edge functions approximate but don't match exactly.

Worth doing when there is a concrete reason to deploy on Vercel — a cost comparison, a team preference, or a feature only one platform offers — rather than speculatively.

### `WANT-9` GBIF and iNat unified import flow

Today GBIF data enters only through a manual CSV export. iNat is fetched live by
`fetch-species.mjs` (on-demand) and `refresh-observations.mjs` (scheduled). The
two sources have no coordination and no shared enrichment step.

**Routing rule**: prefer GBIF for large or historical pulls (anything beyond a
few thousand records or older than ~30 days) and iNat for small, recent updates
(new sightings within the last month). GBIF has deeper historical depth and bulk
download; iNat has fresher real-time coverage and richer photo/annotation data.

What is needed:

1. **`netlify/functions/gbif-fetch.mjs`** — wraps the GBIF Occurrence API
   (`https://api.gbif.org/v1/occurrence/search`), accepts the same filter shape
   the app already uses (taxon, bbox, date range, coordinate uncertainty
   threshold), paginates at 300/page, streams results into Supabase Storage
   incrementally rather than building the whole GeoJSON in memory, and surfaces a
   record count before committing the fetch so a member does not accidentally
   queue a 90k-record pull on a slow connection.

2. **Routing logic in `fetch-species.mjs`** — when a member requests a species,
   check the estimated record count and date range; route to GBIF if count > 5000
   or date range > 60 days, otherwise use iNat. Surface the routing decision in
   the UI so members understand what they are getting.

3. **Shared enrichment step** (see `WANT-10`) — both sources produce raw
   coordinates and a date; the EE enrichment pipeline runs on both outputs
   identically. The enrichment queue should be triggered automatically after any
   import completes, not require a manual follow-up job.

`ISSUE-2` (large imports timing out) is a prerequisite: streaming to Supabase
Storage rather than buffering in memory is the fix for both sources.

Worth doing before `WANT-2` (member-defined enrichment stages) since both GBIF
and iNat records are natural sources for those.

### `WANT-10` JS enrichment pipeline (replaces Python)

The Python pipeline (`enrich_with_rasters.py`, `ee_enrich.py`, `run_pipeline.py`,
`iNat.py`, `cluster.py`, `export_geojson.py`) is obsolete. `netlify/lib/ee-pipeline.mjs`
and `netlify/lib/ee-runner.mjs` already implement the same EE enrichment in
Node and run it server-side on every submitted job. The GitHub Action that ran
the Python pipeline no longer has a reason to exist once the JS pipeline covers
all the stages.

What the JS pipeline still lacks that the Python pipeline provided:

1. **Automatic enrichment after import** — when `refresh-observations.mjs` or a
   new GBIF/iNat import writes fresh features, it should enqueue an enrichment
   job for those features automatically rather than leaving them with
   `enrichment_level: 'none'` until a member manually submits a job. The queue
   already exists (`job-queue.mjs`); the missing piece is a call to `submitJob`
   at the end of every write path.

2. **Enrich endpoint** — a Netlify function (or extension of `ee-jobs.mjs`) that
   accepts a dataset path and enriches only the features missing one or more
   enrichment columns, then writes the result back. This covers the incremental
   re-enrichment case (new columns added to `STAGES`, or records that partially
   failed).

3. **Clustering** — `netlify/lib/cluster.mjs` already does k-means; it needs to
   run as a post-enrichment step on the full dataset, writing the `cluster` label
   back into Supabase Storage. Today it runs only on on-demand `fetch-species`
   results.

4. **Deprecation of the Python scripts and the GitHub Action** — once the above
   three are working end-to-end, `iNat.py`, `enrich_with_rasters.py`,
   `cluster.py`, `export_geojson.py`, `run_pipeline.py`, and the
   `refresh-data.yml` workflow should be removed. The README section describing
   them has been updated to mark them as legacy; the deletion waits for the JS
   path to be verified in production.

Priority: blocking `WANT-9` (both sources need the enrichment step to be
automatic) and `WANT-4` (coverage page reframe needs per-column counts that the
enrichment step now owns).

### `WANT-11` Job → dataset linkage in composable state

After a pipeline or model job completes, `pages/jobs.vue` manually calls
`addInlineDataset()` or `useModelOverlay().show()` and then navigates. If the
user leaves `/jobs` before the download finishes, the enriched dataset is lost
and they must re-download from the jobs page.

The fix is to persist job→dataset linkage in `useEeJobs` or a new
`useJobResults` composable, keyed by job ID, so the result can be recovered
after a page reload or a missed navigation. The jobs page would call this instead
of `addInlineDataset` directly, and any page that renders a dataset selector
would check the composable for results not yet applied.

Worth doing alongside `WANT-10` since the enrichment pipeline will produce more
frequent job completions that members should not have to babysit.

### `WANT-6` A dashboard worth landing on

The default view is thin: it opens on not much, and the interesting state — how
many observations, how fresh, what has been enriched, what is worth looking at
today — is scattered across pages a member has to go find. A landing dashboard
should carry that at a glance.

More widgets, and better ones: totals and recency, a small map or heat preview,
the enrichment coverage the reframed Coverage page will compute, recent jobs and
datasets, maybe a "finds like today's conditions" prompt once the model can
answer it. The work is partly which widgets (each has to answer a real question,
not decorate), partly the layout that makes them read as one view rather than a
pile, and partly the shared state so a widget reflects the same filters the rest
of the app is under.

---

## Future Enhancements

### UI & UX Improvements

- [ ] `V12-UI-1` **Intuitive Navigation**: Streamline the path from data import to model training to reduce friction
- [ ] `V12-UI-2` **Contextual Onboarding**: Implement "empty state" guides and tooltips for complex modeling parameters
- [ ] `V12-UI-3` **Visual Hierarchy Refinement**: Improve contrast and layout of side panels for better focus on the map
- [ ] `V12-UI-4` **Interactive Data Previews**: Enhance dataset selection with instant visual summaries before committing to a model run
- [x] `V12-UI-5` **Accessibility Pass**: ~1/3 of interactive elements lack `aria-label`; add `aria-pressed` to login mode tabs, `aria-hidden` to decorative icons, and a visible label on the ObservationsTable search input (WCAG 2.1 AA)
- [x] `V12-UI-6` **Responsive Table Columns**: ObservationsTable has no `@media` rules — add column prioritisation and a pinned first column for small screens
- [ ] `V12-UI-7` **Mobile Map Controls Discovery**: compact mode hides the basemap picker and heatmap controls entirely — replace with an accessible bottom-sheet or collapsible toolbar row so features remain reachable without opening LayerManager
- [ ] `V12-UI-8` **Fluid Breakpoints**: most responsive behaviour is a binary compact/not-compact prop — supplement with CSS `@media` rules at tablet widths where the binary split creates awkward layouts

### Advanced Modeling Features

- [ ] `V12-MOD-1` **Ensemble Modeling**: Average predictions from multiple model runs
- [ ] `V12-MOD-2` **Projection Tools**: Project models to future climate scenarios (CMIP6 integration)
- [ ] `V12-MOD-3` **Batch Processing**: Train models for multiple species simultaneously
- [ ] `V12-MOD-4` **Model Export**: Download suitability rasters as GeoTIFF
- [ ] `V12-MOD-5` **Threshold Optimization**: Automatic threshold selection (MaxSSS, 10th percentile)

### Data Quality & Ethics

- [ ] `V12-ETH-1` **Sampling Bias Correction**: Automated thinning and bias file generation
- [ ] `V12-ETH-2` **Spatial Autocorrelation Checks**: Warn about clustered occurrence records
- [ ] `V12-ETH-3` **Extrapolation Risk Maps**: Highlight areas outside training environmental space (MOP/MEX analysis)
- [ ] `V12-ETH-4` **Sensitive Species Protection**: Automatic coordinate obscuring for threatened species

### Collaboration & Sharing

- [ ] `V12-COLL-1` **Public Model Gallery**: Browse and reuse models from other users
- [ ] `V12-COLL-2` **Team Workspaces**: Shared projects for research groups
- [ ] `V12-COLL-3` **Model Citation Generator**: Auto-generate citations for published models
- [ ] `V12-COLL-4` **Export to R/Python**: Generate reproducible scripts for external analysis

### Performance & Scalability


### Documentation & Onboarding

- [x] `V12-DOC-1` **Interactive Tutorial**: Step-by-step walkthrough for first MaxEnt run
- [ ] `V12-DOC-2` **Video Guides**: Short screencasts for key workflows
- [x] `V12-DOC-3` **Glossary Tooltips**: Hover explanations for technical terms (AUC, regularization, etc.)
- [ ] `V12-DOC-4` **Example Datasets**: Pre-loaded sample data for practice runs

---

## Verification Debt

### `VDEBT-1` No Earth Engine layer has rendered against the real API

The catalogue's asset IDs, band names and `system:index` values were written
from documentation and from working Code Editor scripts, never executed against
Earth Engine from this codebase. The soil and forest layers in particular carry
band names (`r_cm_p`, `r_0_cm_p`) taken from a script rather than verified here.

This is why `ee-tiles` reports failures loudly and by name: a layer that comes
back blank looks exactly like ground with nothing on it. It is not a substitute
for rendering each one once and looking at it.

There is now a way to run that pass rather than only describe it:
`scripts/verify_ee_layers.mjs` resolves every layer at its defaults, runs its
prepare and count steps, mints a tile template, and prints a pass/fail line per
layer, exiting non-zero if any failed. What is left is not code — it is running
it once on a deployment that has Earth Engine credentials and eyeballing the
layers that pass, because a mint proves the asset and bands but not that the
pixels are right. Until then the band names remain unverified; the harness just
makes verifying them a command rather than a project.

---

## Technical Debt

- [x] `DEBT-1` **TypeScript Migration**: Convert remaining `.js` files to `.ts` for better type safety (Core composables migration substantially complete)
- [x] `DEBT-2` **Test Coverage**:
  - [x] `DEBT-2.1` Unit tests for new MaxEnt visualization components (`ResponseCurve`, `ROCCurve`, etc.)
  - [x] `DEBT-2.2` Integration tests for full modeling pipeline (API → GEE → Supabase)
  - [x] `DEBT-2.3` E2E tests for dashboard customization and widget system
  - [x] `DEBT-2.4` Edge-case expansion for `tests/maxent.test.mjs`
- [x] `DEBT-3` **Error Handling**: Standardize error messages and recovery flows
- [x] `DEBT-4` **Accessibility Audit**: Ensure WCAG 2.1 compliance across new features
- [x] `DEBT-5` **Performance Monitoring**: Add logging for Earth Engine job durations and failures
- [x] `DEBT-6` **Type `any` Cleanup**: `ramps.ts` and `useMapHeatmaps.ts` use `any` for nearly all parameters — replace with proper interfaces for color stops, field metadata, and polygon types
- [ ] `DEBT-7` **Netlify Backend TypeScript Migration**: all `netlify/lib/*.mjs` and `netlify/functions/*.mjs` are untyped — migrate to `.ts` with esbuild/tsup for the Netlify edge runtime
- [ ] `DEBT-8` **MushroomMap Decomposition**: at ~2,800 lines with 25 watchers, split into focused composables (pin logic, heatmap logic, cluster logic, model-overlay logic) and extract the toolbar into its own component
- [x] `DEBT-9` **Silent Error Paths**: audit all `console.error`/`console.warn`-only paths (ChartCard export, map export, GbifImporter, MaxEnt polling, cloud sync) and wire each to the app's toast/notification system

---

## Known Issues

- [ ] `ISSUE-1` LayerManager state persistence occasionally fails on mobile Safari
- [ ] `ISSUE-2` Large GBIF exports (>10k records) may timeout during import
- [ ] `ISSUE-3` Chart rendering slows with >50 data points in Saved Charts widget
- [ ] `ISSUE-4` Earth Engine asset validation doesn't check geometry types comprehensively
- [x] `ISSUE-5` ObservationsTable virtual scroller uses `window.resize` instead of `ResizeObserver` on the container — viewport height goes stale when sidebars toggle or panels resize, causing too few/many rows to render
- [x] `ISSUE-6` MaxEnt polling errors are silently swallowed (`useMaxEnt.ts`) — if Earth Engine polling fails mid-run, job status freezes with no user feedback or retry escalation
- [x] `ISSUE-7` `Number(v).toFixed()` in ObservationsTable has no `NaN` guard — non-numeric cell values render as `"NaN"` instead of a dash or fallback
- [x] `ISSUE-8` Leaflet CSS is imported statically in MushroomMap even on non-map routes — should be deferred alongside the lazy Leaflet JS import

---

## Contribution Guidelines

1. **Branch Naming**: `feature/<name>`, `fix/<name>`, or `roadmap/<phase>`
2. **Commit Messages**: Follow conventional commits (`feat:`, `fix:`, `docs:`, etc.)
3. **Testing**: All new features require tests before merge
4. **Documentation**: Update guide and tooltips for user-facing changes

---
