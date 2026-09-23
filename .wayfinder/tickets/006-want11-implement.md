---
label: wayfinder:task
status: closed
assignee: claude
---

# WANT-11: Implement job→dataset linkage

## Question

Implement the design settled in [005-want11-design](./005-want11-design.md): persist job results in composable state so enriched datasets survive navigation away from `/jobs`.

Acceptance:
- Member submits or completes an enrichment job
- Navigates to `/map`, `/charts`, or `/models`
- Returns to `/jobs` — their result is still there, no re-run required
- Page reload recovers the result (if durable storage was chosen in design)

## Blocked by

→ [005-want11-design](./005-want11-design.md)

## Resolution

- **`composables/useDatasets.ts`**: Added `activeDataset`, `activeGeojson` state (via `useState`) and `activate(slug)` + `loadActiveGeojson()` methods.
- **`pages/jobs.vue`**: Added auto-save watch — when a named enrich job transitions to `succeeded`, `saveJob()` fires automatically; added "Open on charts" button on succeeded enrich job rows; added "Open on charts" button on saved dataset rows; added `openOnCharts(job)` and `openDatasetOnCharts(dataset)` functions.
- **`pages/charts.vue`**: Added `?dataset=slug` URL param handling on mount — activates the dataset and feeds GeoJSON through `addInlineDataset()` so all charts load with the dataset active. URL is set by `openDatasetOnCharts` so the resulting page is bookmarkable and shareable.
- **Not wired**: `/map?dataset=slug` — the map is rendered by `MushroomMap.vue` (DEBT-8, decomposition post-launch). `openOnMap` from the jobs page already works via in-memory `addInlineDataset`.
