# PERF-3 Charts page: `coverageData` does 8 separate array passes

`coverageData` calls `.filter()` once per attribute — 8 passes over ~48k rows.

Fix: replace with a single pass that counts all 8 attributes simultaneously using an accumulator object.
