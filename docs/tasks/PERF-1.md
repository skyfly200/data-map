# PERF-1 Charts page: LazyVisible for gallery charts

The 20+ preset charts all render eagerly on mount. `LazyVisible` is only applied to saved custom charts. Each chart runs its computed data (some expensive, like scatter plots and heatmaps) immediately on page load.

Fix: wrap `GalleryChart`'s slot in `LazyVisible` so below-the-fold chart data computation is deferred until each chart scrolls near the viewport. This mirrors the pattern already used for saved custom charts.
