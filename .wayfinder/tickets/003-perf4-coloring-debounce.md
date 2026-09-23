---
label: wayfinder:task
status: closed
assignee: claude
---

# PERF-4: Debounce map coloring watch

## Question

The `coloring` watch fires a full `eachLayer` restyle — up to 48 000 markers — on every reactive change. Rapid-fire changes (e.g. filter slider drag) chain restyles with no breathing room, visibly freezing the map.

Add a 30–50 ms debounce that collapses rapid-fire changes into a single restyle. See `tasks/PERF-4.md` for the specific watch location.

Acceptance: drag a filter slider quickly; the map doesn't stutter or lock up.

## Blocked by

_(none — frontier ticket)_
