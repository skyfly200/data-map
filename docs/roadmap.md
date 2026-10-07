# Roadmap

Open feature work, ordered by what blocks something else first. When an item ships, add it to the commit message and delete the entry and its task file.

See also: `@docs/refactor.md` for technical debt, `@docs/bugs.md` for known issues.

---

## Wanted

- `WANT-1` **Species distribution modeling** — ships but untested against the real EE API; GeoTIFF export backend added (`model-tiles?download=1`), UI button and live EE verification still open. See `tasks/WANT-1.md`.
- `WANT-2` **Member-defined enrichment stages** — let members point custom EE layers at enrichment without EE code execution. See `tasks/WANT-2.md`.
- `WANT-7` **MaxEnt observation bias correction** — bias grid or target-group background strategy. See `tasks/WANT-7.md`.
- `WANT-3` **Member-supplied EE credentials** — service-account key storage (encrypted), validation, worker + quota wiring shipped; remaining: settings UI, live-EE verification, apply migration 011 + set `EE_CREDENTIAL_KEY`, OAuth option. See `tasks/WANT-3.md`.
- `WANT-8` **Vercel/Netlify cross-compatibility** — adapter layer so the backend deploys on either platform. Scaffold landed (Vercel preset, `server/adapters/vercel-function.ts`, `scripts/gen-vercel-config.mjs`), unverified on Vercel; remaining: no CI build of both targets. See `tasks/WANT-8.md`.
- `WANT-15` **Pipeline UX overhaul** — guided, step-by-step experience from "pick an area" to "enriched dataset on map"; replaces the current flat form. Wayfinder map: [github.com/skyfly200/data-map/issues/140](https://github.com/skyfly200/data-map/issues/140). Key open decisions: mental model (#141), layout/progressive-disclosure (#142), area picker (#144), post-job navigation (#145).
- `WANT-16` **User observation data uploads** — let members upload CSV or GeoJSON of their own observations as a source for pipeline jobs. Design decisions tracked in [github.com/skyfly200/data-map/issues/143](https://github.com/skyfly200/data-map/issues/143). Implemented (CSV + GeoJSON, 3 MB, private dataset in the datasets bucket; pipeline "Upload" source + jobs-page datasets panel); remaining: browser verification, column-mapping UI, design sign-off on #143.
- `WANT-17` **Foray planner** — score where to look right now by combining in-season species, model suitability, shared high-weight variables, access, land type, plus under-sampled-but-promising areas. Done (unapplied/untested live): migrations 011-015, contributions write-back, PAD-US/OSM/RIDB ingest (`access_ingest` job, `scripts/load-access-region.mjs`, `access` endpoint), fee/collecting estimates incl. BLM/USFS `likely_allowed`, `/foray` phase 1 (3 modes, free/public/collecting switches), map Access layer + mobile layer sheet, user/club allowed-area sets (`/areas`, `access-sets`). Remaining: apply migrations, load Colorado, browser verification (polygon rendering, `/areas`, Escape/popup), `viewport-fit` meta for iOS safe-area, `predictorRanges` in runner, phases 2-3 and 5 (ensemble, habitat score, under-sampled ranking UI), county/city collecting-rule table. See `tasks/WANT-17.md`, `access-sources.md`.
- `WANT-18` **AlphaEarth satellite embeddings (maybe)** — Google's `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` (64-band unit vectors, 10 m, annual from 2017, CC-BY 4.0). Shipped: `habitat-similarity` map layer (mean embedding at a taxon's finds, cosine similarity per pixel; unverified against live EE, default 70% floor is a guess). Maybe, not started:
  - **MaxEnt predictors** — add the embedding to `MAXENT_PREDICTORS` (`netlify/lib/maxent.mjs`), reduced to 8–16 PCA components to limit overfitting; compare AUC against the named-variable set. Components are uninterpretable, so variable contribution and response curves mean nothing for them.
  - **Foray planner phase 5** (`WANT-17`) — use embedding similarity to known patches as the "promise" signal for under-sampled cells, and distance from the training sites' embeddings as the training-envelope check. Keep phase 3's habitat score on named predictors, which need readable ranges.
  - **Per-observation enrichment** — an `ee_enrich.py` stage sampling `A00`–`A63` at each point for its observation year (clamped to 2017+), stored with pgvector for "finds from similar habitat" queries. Needs a migration; ~9.6k × 64 floats.

- Explore high level project arcetechture
- Check DB structure for consistency with expectations
- Check for security vulnerabilities
- Use FRMS Google for GEE scripts backend and update to nonprofit status
- Add FRMS users to app atomatically
- Setup Emails
