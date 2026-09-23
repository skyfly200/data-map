# PERF-2 Charts page: O(n×bins) binning in `tempHighLowDist` / `elevationData`

`tempHighLowDist` filters both `highVals` and `lowVals` per bin — O(n × bins). `elevationData` also re-filters per elevation band. Both scan the full dataset once per bucket.

Fix: bucket in a single pass over the data, accumulating counts per bin in one loop — the same pattern already used in `clusterProfile` and `speciesLandcover`.
