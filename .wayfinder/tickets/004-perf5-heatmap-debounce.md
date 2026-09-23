---
label: wayfinder:task
status: closed
assignee: claude
---

# PERF-5: Debounce heatmap rebuild

## Question

The `heatmapResult` computed rebuilds the full heatmap grid on every `filteredData` change. With large datasets this fires expensively on every filter interaction.

Replace the eager computed with a debounced `watchEffect` writing to a ref, so rapid filter changes collapse into one rebuild.

Acceptance: changing filters does not produce multiple sequential heatmap redraws; the map stays responsive.

## Blocked by

_(none — frontier ticket)_
