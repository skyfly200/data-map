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
- `WANT-17` **Foray planner** — score where to look right now by combining in-season species, model suitability, shared high-weight variables, access, land type, plus under-sampled-but-promising areas. Done (unapplied/untested live): migration 012, contributions write-back, PAD-US/OSM parsers, scoring + ranking libs. Remaining: apply migration, runner computing `predictorRanges`, ingest job that fetches/loads PAD-US + OSM, wire scoring into heatmap/UI (phases 1-2, 5). See `tasks/WANT-17.md`.

- Explore high level project arcetechture
- Check DB structure for consistency with expectations
- Check for security vulnerabilities
- Use FRMS Google for GEE scripts backend and update to nonprofit status
- Add FRMS users to app atomatically
- Setup Emails
