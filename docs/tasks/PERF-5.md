# PERF-5 Map page: debounce heatmap rebuild

`heatmapResult` is a computed that rebuilds the full heatmap grid from scratch on every `filteredData` change. Any filter interaction triggers an expensive rebuild synchronously.

Fix: replace the computed with a debounced `watchEffect` that writes to a ref. Filter interactions stay snappy and the expensive heatmap rebuild is batched to fire once after the interaction settles.
