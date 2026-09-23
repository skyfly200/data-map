---
label: wayfinder:grilling
status: closed
assignee: claude
---

# WANT-11: Design job→dataset linkage in composable state

## Question

Currently, enriched job results do not survive navigation away from `/jobs` — the composable state is lost and the member has to re-run or re-navigate to recover their dataset. Before implementing, we need to decide:

1. Does `useEeJobs` grow a "persist results" responsibility, or does a new `useJobResults` composable own this?
2. Where does the persisted state live — Pinia store, `useState` (Nuxt SSR-safe), or Supabase (durable across sessions)?
3. What is the minimal shape of a "job result" that downstream pages (charts, map, model) need to reference — is it the full job row, a dataset ID, or a materialized view of the enriched columns?
4. How does a member re-attach to a result from a previous session (link from job list, auto-restore on page load)?

Call the `grilling` and `domain-modeling` skills when resolving this ticket.

## Blocked by

→ [001-vdebt1-research](./001-vdebt1-research.md) (informs job result shape)

## Resolution

Design settled through grilling:

1. **Name at submission** — job title field on the submission form; no post-completion prompt.
2. **Auto-save on completion** — client detects `status === 'succeeded'` in poll, calls `useDatasets.saveJob(job)` automatically with the pre-provided title.
3. **Extend `useDatasets`** — add `activeDataset: useState<Dataset|null>`, `activeGeojson: useState<any|null>`, `activate(slug)`, `loadActiveGeojson()`.
4. **URL persistence** — `?dataset=slug` on `/charts`, `/map`, `/model`; each page reads param on mount to re-attach from any session.
5. **`useObservations` integration** — reads from `activeGeojson` when set; existing filters/charts/model work unchanged.
