# Roadmap

Open feature work, ordered by what blocks something else first. When an item ships, add it to the commit message and delete the entry and its task file.

See also: `@docs/refactor.md` for technical debt, `@docs/bugs.md` for known issues.

---

## Wanted

- `WANT-1` **Species distribution modeling** — ships but untested against the real EE API; GeoTIFF export still open. See `tasks/WANT-1.md`.
- `WANT-11` **Job → dataset linkage in composable state** — persist job results in `useEeJobs` / `useJobResults` so enriched datasets survive navigation away from `/jobs`. See `tasks/WANT-11.md`.
- `WANT-2` **Member-defined enrichment stages** — let members point custom EE layers at enrichment without EE code execution. See `tasks/WANT-2.md`.
- `WANT-4` **Coverage page, reframed** — replace raster cache inventory with per-column field coverage stats. See `tasks/WANT-4.md`.
- `WANT-7` **MaxEnt observation bias correction** — bias grid or target-group background strategy. See `tasks/WANT-7.md`.
- `WANT-3` **Member-supplied EE credentials** — members run jobs under their own EE project/budget. See `tasks/WANT-3.md`.
- `WANT-8` **Vercel/Netlify cross-compatibility** — adapter layer so the backend deploys on either platform. See `tasks/WANT-8.md`.
- `WANT-13` **Custom 404 Page** — add our own 404 branded with our site not just falling back on the one from Netlify.
- `WANT-15` **Pipeline UX overhaul** — guided, step-by-step experience from "pick an area" to "enriched dataset on map"; replaces the current flat form. Wayfinder map: [github.com/skyfly200/data-map/issues/140](https://github.com/skyfly200/data-map/issues/140). Key open decisions: mental model (#141), layout/progressive-disclosure (#142), area picker (#144), post-job navigation (#145).
- `WANT-16` **User observation data uploads** — let members upload CSV or GeoJSON of their own observations as a source for pipeline jobs. Design decisions tracked in [github.com/skyfly200/data-map/issues/143](https://github.com/skyfly200/data-map/issues/143). Expected entry points: pipeline source picker and datasets page.

