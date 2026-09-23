---
label: wayfinder:task
status: closed
assignee: claude-fork
priority: nice-to-have
---

# ISSUE-3 + PERF-1/2/3: Charts performance improvements

## Question

Nice-to-have before the foray if time allows. Four independent improvements to the charts page:

- **ISSUE-3**: Chart rendering slows with >50 data points — profile and fix the bottleneck.
- **PERF-1**: `GalleryChart` slots render eagerly; wrap in `LazyVisible` to defer below-fold computation.
- **PERF-2**: `tempHighLowDist` / `elevationData` bin in O(n×bins); replace with single-pass bucketing as `clusterProfile` already does.
- **PERF-3**: `coverageData` makes 8 separate array passes; replace with a single-pass counter.

Any of these can ship independently. Take the highest-impact one if only partial time is available.

## Blocked by

_(none — frontier ticket, lower priority)_

## Resolution

- **PERF-1**: `GalleryChart.vue` — below-fold preset charts deferred with `<LazyVisible min-height="260px">`.
- **PERF-2**: `pages/charts.vue` — `tempHighLowDist` and `elevationData` O(n×bins) `.filter()` loops replaced with single-pass bucketing.
- **PERF-3**: `pages/charts.vue` — `coverageData` 8-pass filter replaced with one loop counting all 8 attributes.
- **ISSUE-3**: Already fixed in `ScatterChart.vue` (cell deduplication + off-screen point drop). No change needed.

## Resolution

**PERF-1** — Fixed. Added `<LazyVisible min-height="260px">` wrapping the `<slot />` inside `GalleryChart.vue`. All preset gallery cards now defer rendering until they approach the viewport (300px rootMargin), matching what the saved-charts section already did.

**PERF-2** — Fixed. `tempHighLowDist` and `elevationData` in `pages/charts.vue` now compute the bin index per datum in a single forward pass instead of `filter()`-ing the whole value array per bin. Both were O(n×bins); both are now O(n).

**PERF-3** — Fixed. `coverageData` in `pages/charts.vue` replaced 8 separate `.filter()` passes with one loop that increments a counter per attribute. Result is identical; work is one pass over `rows.value`.

**ISSUE-3** — Already resolved. `ScatterChart.vue` (lines 256–289) already contains a cell-deduplication pass that collapses overlapping marks and drops off-screen points before building SVG path nodes. The comment documents the original symptom ("~138k nodes… half a minute to appear") and the fix. No further work needed.

Files changed: `components/GalleryChart.vue`, `pages/charts.vue`

Status: closed
