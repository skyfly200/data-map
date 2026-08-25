# data-map

A map of environmental conditions around mushroom observations. iNaturalist
observations are enriched with remotely-sensed environmental layers and then
clustered by environmental similarity. The frontend is a Nuxt app.

## Python data pipeline

Install the pipeline dependencies:

```bash
pip install -r requirements.txt
```

### Credentials

| For | Set |
| --- | --- |
| NDVI / soil / satellite-moisture (Earth Engine) | run `python gauth.py` once; set `EARTHENGINE_PROJECT` to your Google Cloud project id |
| DEM ([OpenTopography](https://portal.opentopography.org/login)) | `OPENTOPOGRAPHY_API_KEY` |
| Soil moisture ([Copernicus CDS](https://cds.climate.copernicus.eu)) | copy `.cdsapirc.example` to `~/.cdsapirc` with your key (and accept the ERA5-Land license on the CDS site) |

Stages (run in order):

1. **`iNat.py`** — pull mushroom observations from iNaturalist (+ elevation and
   weather) → `mushroom_observations.csv`.
2. **`fetch.py`** — download *all* the environmental data for those observations,
   end to end: NDVI (Sentinel-2 via Earth Engine), soil moisture (ERA5-Land),
   precipitation (CHIRPS, 7-day history), land cover (ESA WorldCover tiles,
   auto-downloaded), and **topography (SRTM DEM)**. After the DEM downloads it
   runs `terrain_pipeline.process_dem` to derive the terrain-exposure layers.
   NDVI is exported asynchronously to Google Drive (folder `EarthEngineNDVI`) —
   download those GeoTIFFs into `ndvi/` before enriching.
3. **`terrain_pipeline.py`** — turn the raw DEM into terrain-exposure layers
   (see below). Runs automatically from `fetch.py`, or standalone:
   `python terrain_pipeline.py --dem dem/dem_SRTMGL3.tif`.
4. **`enrich_with_rasters.py`** — sample every raster (including the terrain
   layers) at each observation point → `mushroom_observations_enriched.csv`.
5. **`cluster.py`** — KMeans-cluster observations by environmental similarity →
   `mushroom_clusters.csv`.

### Running in a notebook / Colab

Every module is import-safe — the run logic lives in functions behind an
`if __name__ == "__main__"` guard — so you can drive stages cell by cell:

```python
import ee; ee.Initialize(project="your-gcp-project")   # or fetch.init_earth_engine()
import fetch, enrich_with_rasters as enrich, terrain_pipeline as terrain

fetch.download_worldcover_tiles(df)          # call individual steps...
fetch.main()                                  # ...or run the whole download
terrain.process_dem("dem/dem_SRTMGL3.tif")
```

In Colab, `pip install rasterio netCDF4 earthengine-api cdsapi`, authenticate
Earth Engine with `ee.Authenticate()`, and note that NDVI/satellite-moisture
exports still land in Google Drive (download them before enriching).

### Topographic exposure layers

`terrain_pipeline.py` reads the DEM and writes these GeoTIFFs to
`dem/derived/`, which `enrich_with_rasters.py` samples as columns:

| Layer | Meaning |
| --- | --- |
| `slope`, `aspect` | Steepness (degrees) and downhill compass bearing. |
| `solar_exposure` | Potential incoming solar radiation (0–1), from slope + aspect integrated over sun positions across the seasons. South-facing slopes score high in the northern hemisphere. |
| `wind_exposure` | Topographic wind exposure (0–1): multi-scale topographic position (ridges exposed, valleys sheltered) combined with how much a slope faces the prevailing wind. Set the wind direction with `--wind-dir` (default 270°/westerly). |
| `water_retention` | Topographic Wetness Index (0–1): `ln(a / tan(slope))` from D8 flow accumulation. Flat, converging, valley-bottom terrain retains water; steep ridges shed it. |

The DEM download needs a free [OpenTopography](https://portal.opentopography.org/login)
API key in `OPENTOPOGRAPHY_API_KEY`.

### Validating water retention against observed moisture

`water_retention` is a *static* terrain prediction; `validate_wetness.py` checks
how well it agrees with independently observed moisture. Because TWI is a
potential (not an instantaneous state) and is log-scaled, agreement is measured
with **Spearman rank** correlation and is strongest right after rain.

```bash
# Quick check against the columns already in the enriched CSV
python validate_wetness.py points

# Rigorous pixel-wise check against a satellite moisture raster you export
# (Sentinel-1 VV backscatter, Sentinel-2 NDMI, or SMAP), masking water/built-up
python validate_wetness.py raster --satellite ndmi.tif --landcover world_cover/<tile>.tif
```

ERA5-Land soil moisture (~9 km) is only a coarse sanity check; for a meaningful
comparison export a fine-resolution satellite moisture layer over the DEM
footprint. `fetch.py` can export two, both via Earth Engine to Google Drive
(folder `EarthEngineMoisture`):

- **`fetch_sentinel1_moisture`** — Sentinel-1 VV backscatter (90 m), the
  strongest soil-moisture proxy (SAR, all-weather).
- **`fetch_sentinel2_ndmi`** — Sentinel-2 NDMI `(B8−B11)/(B8+B11)` (20 m),
  optical vegetation/surface moisture.

These are large exports, so they only run when opted in:

```bash
EXPORT_SATELLITE_MOISTURE=1 python fetch.py   # queues the exports for the observed date span
# ...download the GeoTIFF from Drive, then:
python validate_wetness.py raster --satellite s1_vv_<window>.tif \
    --landcover world_cover/<tile>.tif --scatter wetness_check.png
```

Sentinel-1 VV tracks bare/low-vegetation soil moisture best, so masking dense
vegetation and built-up/water (via `--landcover`) sharpens the comparison.

## Nuxt frontend & Netlify deploy

The frontend is a Nuxt 3 app that renders the observations on a Leaflet map,
coloured by environmental cluster, with the enriched attributes in each popup.
It reads a **static GeoJSON** file — no backend or database.

Data flow: the Python pipeline runs **offline** and produces a small GeoJSON that
the app serves statically. The heavy raster processing never runs on Netlify.

```
enrich_with_rasters.py → cluster.py → export_geojson.py
                                          → public/data/observations.geojson
                                                → committed → Netlify redeploys
```

Regenerate the map data after re-running the pipeline:

```bash
python export_geojson.py          # writes public/data/observations.geojson
```

Run the site locally:

```bash
npm install
npm run dev        # http://localhost:3000
```

**Netlify:** `netlify.toml` pins the build (`npm run build`, publish `dist`,
Node 20); Nuxt's Nitro auto-selects the Netlify preset. Only the small GeoJSON
in `public/data/` is served — keep the raster folders (`soil/ ndvi/ dem/ …`)
out of the deploy (they are gitignored). Do **not** put Earth Engine /
OpenTopography / CDS credentials in Netlify; those belong only to the offline
pipeline.

### Keeping the data fresh (automated)

Two schedules refresh the map, split by what each environment can run:

- **GitHub Action** (`.github/workflows/refresh-data.yml`, daily) runs the
  Python pipeline headless — the parts that need Python (raster download,
  terrain derivation, enrichment, clustering) — and commits an updated
  `public/data/observations.geojson`, which triggers a Netlify redeploy. Earth
  Engine steps are skipped (`SKIP_EARTH_ENGINE=1`, since EE exports to Drive
  can't run headless). Put `OPENTOPOGRAPHY_API_KEY` (and optionally
  `CDSAPI_URL` / `CDSAPI_KEY`) in the repo's Actions secrets.

- **Scheduled Netlify Function** (`netlify/functions/refresh-observations.mjs`,
  every 6 h) does a fast, light refresh in Node: fetches recent iNaturalist
  sightings, merges any new ones onto the committed baseline, samples the
  terrain rasters for those points (if `data/terrain/*.tif` are committed), and
  writes the result to **Netlify Blobs**. The serving function
  (`netlify/functions/observations.mjs`) returns that fresh copy, or the
  baseline file if the blob isn't present. The map fetches
  `/.netlify/functions/observations` and falls back to the static file.

  Netlify functions can't run Python/GDAL/Earth Engine, so the light refresh
  only adds new sightings with terrain context; the satellite-derived columns
  (NDVI, soil moisture) are filled in on the next GitHub Action run.

  Configure the iNaturalist query with env vars (`INAT_TAXON`, `INAT_LAT`,
  `INAT_LNG`, `INAT_RADIUS`) in the Netlify site settings.

### Serving datasets from Supabase Storage (optional)

By default the datasets live in `public/data/` (committed) and the interim
refresh uses Netlify Blobs. You can instead store them in **Supabase Storage**,
which becomes the source of truth the frontend and functions read from. It's
entirely opt-in — with no `SUPABASE_*` env set, everything falls back to the
committed files / Blobs.

Setup:

1. Pick (or create) a Supabase project and a **public** Storage bucket, e.g.
   `datasets`.
2. Set env vars:
   - GitHub Actions secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
     (and optionally `SUPABASE_DATASETS_BUCKET`). The `refresh-data` workflow
     runs `node scripts/upload_datasets.mjs` after export to push
     `observations.geojson`, `species/*.geojson`, and a rewritten
     `datasets.json` (with Supabase public URLs) to the bucket.
   - Netlify env: `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (so the
     functions read/write Storage instead of Blobs), and
     `NUXT_PUBLIC_DATASETS_MANIFEST_URL` = the bucket's public
     `datasets.json` URL (so the frontend loads datasets from Supabase).
3. The service-role key is **write-only server-side** — it lives in Actions /
   Netlify env, never in the browser bundle.

Data flow: Python pipeline → `public/data/` → `upload_datasets.mjs` → Supabase
Storage → frontend + `observations` function read from Supabase; the scheduled
`refresh-observations` function writes `new-observations.geojson` to the same
bucket.

Look at the [Nuxt documentation](https://nuxt.com/docs/getting-started/introduction) to learn more.

## Setup

Make sure to install dependencies:

```bash
# npm
npm install

# pnpm
pnpm install

# yarn
yarn install

# bun
bun install
```

## Development Server

Start the development server on `http://localhost:3000`:

```bash
# npm
npm run dev

# pnpm
pnpm dev

# yarn
yarn dev

# bun
bun run dev
```

## Production

Build the application for production:

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build

# bun
bun run build
```

Locally preview production build:

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview

# bun
bun run preview
```

Check out the [deployment documentation](https://nuxt.com/docs/getting-started/deployment) for more information.
