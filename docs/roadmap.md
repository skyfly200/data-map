# Roadmap

Open feature work, ordered by what blocks something else first. When an item ships, add it to the commit message and delete the entry and its task file.

See also: `@docs/refactor.md` for technical debt, `@docs/bugs.md` for known issues.

---

## Wanted

- `WANT-1` **Species distribution modeling** — ships but untested against the real EE API; GeoTIFF export added (`model-tiles?download=1`, Export menu on model jobs in `/jobs` and after training in `/model`); live EE verification still open, including whether the 12M-pixel ceiling in `composables/modelExport.ts` matches EE's download limit. See `tasks/WANT-1.md`.
- `WANT-2` **Member-defined enrichment stages** — let members point custom EE layers at enrichment without EE code execution. See `tasks/WANT-2.md`.
- `WANT-7` **MaxEnt observation bias correction** — bias grid or target-group background strategy. See `tasks/WANT-7.md`.
- `WANT-3` **Member-supplied EE credentials** — service-account key storage (encrypted), validation, worker + quota wiring shipped; remaining: settings UI, live-EE verification, apply migration 011 + set `EE_CREDENTIAL_KEY`, OAuth option. See `tasks/WANT-3.md`.
- `WANT-8` **Vercel/Netlify cross-compatibility** — adapter layer so the backend deploys on either platform. Scaffold landed (Vercel preset, `server/adapters/vercel-function.ts`, `scripts/gen-vercel-config.mjs`), unverified on Vercel; remaining: no CI build of both targets. See `tasks/WANT-8.md`.
- `WANT-15` **Pipeline UX overhaul** — guided, step-by-step experience from "pick an area" to "enriched dataset on map"; replaces the current flat form. Wayfinder map: [github.com/skyfly200/data-map/issues/140](https://github.com/skyfly200/data-map/issues/140). Key open decisions: mental model (#141), layout/progressive-disclosure (#142), area picker (#144), post-job navigation (#145).
- `WANT-16` **User observation data uploads** — let members upload CSV or GeoJSON of their own observations as a source for pipeline jobs. Design decisions tracked in [github.com/skyfly200/data-map/issues/143](https://github.com/skyfly200/data-map/issues/143). Implemented (CSV + GeoJSON, 3 MB, private dataset in the datasets bucket; pipeline "Upload" source + jobs-page datasets panel); remaining: browser verification, column-mapping UI, design sign-off on #143.
- `WANT-17` **Foray planner** — score where to look right now by combining in-season species, model suitability, shared high-weight variables, access, land type, plus under-sampled-but-promising areas. Done (unapplied/untested live): migrations 011-016, contributions and `predictorRanges` write-back, PAD-US/OSM/RIDB ingest (`access_ingest` job, `scripts/load-access-region.mjs`, `access` endpoint), fee/collecting estimates incl. BLM/USFS `likely_allowed`, county/city collecting-rule table (`collecting-rules.mjs`), `/foray` phase 1 (3 modes, free/public/collecting switches), phases 2-3 model layers (ensemble, habitat score; `foray-layers` endpoint), phase 5 "Where few have looked" ranking with training-envelope guard, map Access layer + mobile layer sheet, user/club allowed-area sets (`/areas`, `access-sets`), tunable scoring (Scoring panel, relative/fixed grading, URL params), "Why this grade?" breakdown, gradient legend, model blend into grades with predicted places, parking/trailhead ingest (migration 017, `access-nearest` endpoint) with walk-in tiers and accessible-parking filter, round two (25 mi default plus a drive-time area from OpenRouteService/Valhalla via `drive-time`, a Skip towns / Everywhere / Towns only toggle, grading against this date / the whole season / all past seasons, planning on a date or range, nearest place names with coordinates and plus codes, Show on map links, an Edibles filter). Remaining: apply migrations 011-017, load Colorado (re-load for parking points), live EE run of the layers and the blend, ISSUE-6 (model write-back), browser verification (polygon rendering, `/areas`, Escape/popup, model layers). See `tasks/WANT-17.md`, `access-sources.md`.
- `WANT-18` **AlphaEarth satellite embeddings (maybe)** — Google's `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64-band unit vectors, 10 m, annual from 2017, CC-BY 4.0). Shipped: `habitat-similarity` map layer (mean embedding at a taxon's finds, or at up to 10 spots tapped on the map and shared via `?hs=`; cosine similarity per pixel; unverified against live EE, default 70% floor is a guess). Maybe, not started:
  - **MaxEnt predictors** — add the embedding to `MAXENT_PREDICTORS` (`netlify/lib/maxent.mjs`), reduced to 8–16 PCA components to limit overfitting; compare AUC against the named-variable set. Components are uninterpretable, so variable contribution and response curves mean nothing for them.
  - **Foray planner phase 5** (`WANT-17`) — use embedding similarity to known patches as the "promise" signal for under-sampled cells, and distance from the training sites' embeddings as the training-envelope check. Keep phase 3's habitat score on named predictors, which need readable ranges.
  - **Per-observation enrichment** — an `ee_enrich.py` stage sampling `A00`–`A63` at each point for its observation year (clamped to 2017+), stored with pgvector for "finds from similar habitat" queries. Needs a migration; ~9.6k × 64 floats.
- `WANT-19` **Season outlook from WeatherNext** — when each species' remaining season should start and end near a place, as dates with a likely range. Shipped (unverified against live EE): `season-weather` endpoint (`netlify/lib/season-weather.mjs`; ERA5-Land + CHIRPS daily per 0.1° cell, cached forever for past years; latest WeatherNext 2 run `projects/gcp-public-data-weathernext/assets/weathernext_2_0_0`, 64 members × 15 days, plus 30 days of 00z short-range runs for gap fill and bias), `composables/seasonForecast.ts` (per-species start trigger chosen by leave-one-year-out error among calendar / GDD / 30-day rain, end by calendar or days after first hard frost; scenarios = observed to date + each member + each past year; recent-bias correction of temps and rain ratio), Season outlook dashboard widget. Remaining:
  - **Live EE check** — confirm WeatherNext property names (`ensemble_member`, `forecast_hour`, init as `system:time_start`), that the FRMS service account sees real-time runs, and request timings within the function limit.
  - **Hindcast** — 2022–2025: issue outlooks from past WeatherNext runs, score start/end error in days against the climatology-only baseline; ship the forecast blend only where it wins.
  - **Better correction** — quantile mapping on the hindcasts instead of the 30-day mean offset; lapse-rate temperature adjustment from SRTM, since a cell is ~11 km and finds sit at their own elevation; local-time days instead of UTC.
  - **Wire in** — feed outlooks to Now Fruiting and `/foray` date planning (`forayScore.ts` plan day/window); daily scheduled refresh after the 00z run (~07:30 UTC). WeatherNext 3 (0.1°, hourly, percentiles only) once its full members are reachable.

- Explore high level project arcetechture
- Check DB structure for consistency with expectations
- Check for security vulnerabilities
- Use FRMS Google for GEE scripts backend and update to nonprofit status
- Add FRMS users to app atomatically
- Setup Emails

---

## Planned (document only, not started)

Plan from Skyler, grouped. Order runs top to bottom: quick wins and shared layout work first, then the features that build on them. Scope: S under a few days, M a week or so, L longer or needs live EE checks.

**Order at a glance:** `WANT-20` → `WANT-21` → (`WANT-22`, `WANT-23`) in parallel → `WANT-24` → `WANT-25` → `WANT-26`, then `WANT-27` once the rest settles.

- `WANT-20` **Skip towns rename and filtering** (S, do first). Rename "Skip towns" to something that says what it does, e.g. "Exclude built-up areas". Today it only drops towns from the round-two toggle in `/foray` (`WANT-17`). Make it remove cities and built-up cells too, not just named places. Likely needs a built-up source (GHSL settlement layer or OSM place polygons) sampled per cell, so the filter works in the cell grid and not only by name. Depends on: nothing. Blocks: nothing, but `WANT-22` and `WANT-23` use the same copy.

- `WANT-21` **Visual polish, mobile, and balance** (M, shared foundation). Covers three asks that touch the same components:
  - UI polish across the app and better phone support. Check every change at a phone viewport.
  - Less wordy, more visual, better balanced layouts. Move detail into tooltips or expandable sections and use icons, color, and small charts where text is doing the work.
  - Dashboard widgets fill the space available. Switch the grid to flexible sizing so widgets stretch to the container instead of fixed heights.
  Depends on: nothing. Do before `WANT-22` and `WANT-23` so new screens use the same tokens and grid. Relates to `WANT-15` (pipeline UX) for the layout decisions already open there.

- `WANT-22` **Streamlined training flow** (M). Four parts:
  - Better defaults so most users never open advanced settings.
  - Hide advanced settings behind one collapsed section.
  - Auto-name models, and carry the name through the steps (Source → Scout → Train → Evaluate) so a name entered once is reused.
  - Keep the grouped predictor picker from PR #205.
  Depends on: `WANT-21` for the layout. Relates to `WANT-15` and `WANT-7` (bias correction adds a new advanced option, so keep it in the hidden section). Known gap to settle first: the /jobs "Enrich & model" path still runs a separate enrich job before training.

- `WANT-23` **Expanding runs and datasets in place** (L, data model). Let a user add new sources or a wider area to an existing enrichment run or dataset instead of starting fresh. Needs:
  - A stable record key per observation (source + source id) so reruns dedupe instead of double counting.
  - A per-dataset watermark or query window so only new records are fetched.
  - Migration and schema notes in `docs/schema.md`.
  Depends on: `WANT-16` (uploads share the same key rules). Blocks: `WANT-24` (auto enrichment reruns use the same append path).

- `WANT-24` **Automatic enrichment for admin-defined taxa** (L). Admins set a list of taxa to scan and keep up to date, on a schedule. Needs an admin-only taxa table, a scheduled job that runs the append path from `WANT-23` per taxon, and a status view showing last run and counts. Depends on: `WANT-23`, and jobs running reliably in the background worker. Check that before scheduling, since scheduled runs are only as good as the worker path.

- `WANT-25` **Charts: more types and a stronger creator** (M). Add Violin, Trend line, and 3D plot, and make the chart creator handle series, grouping, and axis choices in one place. Use the dataviz palette rules for color. Depends on: `WANT-21` for layout. 3D plot is the riskiest part on mobile, so ship it last and fall back to a 2D view under phone width.

- `WANT-26` **Pattern search in enriched data** (L). Tools to find patterns across enriched records, such as "finds that look like this one" and grouped comparisons over predictors. Could reuse the pgvector idea in `WANT-18` for similarity search, so decide that before building. Depends on: `WANT-23` for consistent records, and `WANT-25` for the charts the results show.

- `WANT-27` **Docs and modeling-flow visuals** (S to M, rolling). Update the docs as each item above lands, and add diagrams of the modeling flow (Source → Scout → Train → Evaluate, and how enrichment feeds it). Start with the flow diagram once `WANT-22` ships, so it shows the final steps. Keep `docs/` in sync with the change in the same PR, not in a batch at the end.
