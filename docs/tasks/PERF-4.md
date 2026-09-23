# PERF-4 Map page: debounce `coloring` watch

`watch([coloring, sizeScale, activeColors, colorOverrides, pointRadius, ...])` triggers a loop over every marker (up to 48k) on each reactive change. Rapid-fire changes — e.g. typing in a filter — fire one full restyle pass per keystroke.

Fix: wrap the `eachLayer` restyle in a 30–50ms debounce so rapid changes collapse into one pass.
