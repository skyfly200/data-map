# Roadmap

Open feature work, ordered by what blocks something else first. When an item ships, add it to the commit message and delete the entry and its task file.

See also: `@docs/refactor.md` for technical debt, `@docs/bugs.md` for known issues.

---

## Wanted

- `WANT-1` **Species distribution modeling** — ships but untested against the real EE API; GeoTIFF export still open. See `tasks/WANT-1.md`.
- `WANT-9` **GBIF + iNat unified import flow** — prefer GBIF for large/historical pulls (> 5 000 records or > 60 days), iNat for small/recent updates; new `gbif-fetch.mjs` function with streaming to Supabase Storage. Prerequisite: fix `ISSUE-2`. See `tasks/WANT-9.md`.
- `WANT-10` **JS enrichment pipeline (replaces Python)** — auto-enrich after import, incremental enrich endpoint, post-enrichment clustering, deprecation of Python scripts and `refresh-data.yml`. Blocks `WANT-9` and `WANT-4`. See `tasks/WANT-10.md`.
- `WANT-11` **Job → dataset linkage in composable state** — persist job results in `useEeJobs` / `useJobResults` so enriched datasets survive navigation away from `/jobs`. See `tasks/WANT-11.md`.
- `WANT-2` **Member-defined enrichment stages** — let members point custom EE layers at enrichment without EE code execution. See `tasks/WANT-2.md`.
- `WANT-4` **Coverage page, reframed** — replace raster cache inventory with per-column field coverage stats. See `tasks/WANT-4.md`.
- `WANT-5` **Navigation** — remove `/jobs` from top nav; add pages that earn a slot. See `tasks/WANT-5.md`.
- `WANT-6` **Dashboard** — landing view with observation totals, enrichment coverage, recent jobs, model prompt. See `tasks/WANT-6.md`.
- `WANT-7` **MaxEnt observation bias correction** — bias grid or target-group background strategy. See `tasks/WANT-7.md`.
- `WANT-3` **Member-supplied EE credentials** — members run jobs under their own EE project/budget. See `tasks/WANT-3.md`.
- `WANT-8` **Vercel/Netlify cross-compatibility** — adapter layer so the backend deploys on either platform. See `tasks/WANT-8.md`.
