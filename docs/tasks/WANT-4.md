# WANT-4 Coverage page, reframed

`/coverage` currently inventories the local raster cache — 25.9 GB of CHIRPS, ERA5 and NDVI files on disk. As enrichment moves to Earth Engine that cache stops existing and the page describes an artifact of the architecture being replaced.

The surviving question is field coverage: what fraction of records actually carry each enrichment column. A mean over the 23% of rows that have soil moisture is a different claim from a mean over all of them, and nothing in the app currently says which you are looking at.

Keep the URL, replace the contents, let the raster inventory go when the cache does.
