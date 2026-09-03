# Nexstrata

A map of environmental conditions around mushroom observations. iNaturalist
observations are enriched with remotely-sensed environmental layers and then
clustered by environmental similarity. The frontend is a Nuxt app.

**The name.** *strata* — layers. Where a mushroom grows is not one fact but a
stack of them: the weather of the week before, the canopy over it, the moisture
in the soil, the shape and aspect of the ground, how much sun and wind that
shape lets through. *nex* — from *nexus*, a binding together. An observation is
the one place all those layers meet. The mark shows it: a beam passing down
through every stratum onto a geotag on the ground.

The repository is still `data-map`; the app is Nexstrata.

## Python data pipeline

Install the pipeline dependencies:

```bash
pip install -r requirements.txt
```

### Credentials

Earth Engine supplies every environmental layer, so it is the only credential
the pipeline needs. The rest are for the raster fallback described below.

**Finding `EARTHENGINE_PROJECT`** — it is the *project ID* of a Google Cloud
project registered for Earth Engine (e.g. `my-project-451208`), not the display
name and not the project number:

- [console.cloud.google.com](https://console.cloud.google.com/) — the project
  picker lists every project with its ID column
- [code.earthengine.google.com](https://code.earthengine.google.com/) — the Code
  Editor shows the active project top-right and in the Assets tab
- not registered yet? [code.earthengine.google.com/register](https://code.earthengine.google.com/register)
  attaches a Cloud project to Earth Engine (free for noncommercial use)
- already using gcloud? `gcloud config get-value project`

`python scripts/preflight.py --ee-project` prints the one this checkout will use
(reading `EARTHENGINE_PROJECT`, then the stored Earth Engine credential, then
gcloud), or explains where to find one. Put it in `.env` at the repo root:

```
EARTHENGINE_PROJECT=your-project-id
```

| For | Set |
| --- | --- |
| **Everything environmental (Earth Engine)** | run `python gauth.py` once; set `EARTHENGINE_PROJECT` to your Google Cloud project id |
| DEM ([OpenTopography](https://portal.opentopography.org/login)) — *fallback only* | `OPENTOPOGRAPHY_API_KEY` |
| Soil moisture ([Copernicus CDS](https://cds.climate.copernicus.eu)) — *fallback only* | copy `.cdsapirc.example` to `~/.cdsapirc` with your key (and accept the ERA5-Land license on the CDS site) |

### Data layout (per-species CSV store)

Observations live as one lightweight CSV per species — not a single monolithic
file — under a dedicated folder, with enriched copies alongside:

```
data/
  species/<slug>.csv     raw observations, one file per species (tracked)
  enriched/<slug>.csv    enriched observations, clusters folded in (git-ignored, regenerable)
  archive/               originals moved by the migration (git-ignored, local backup)
```

`species_store.py` is the shared accessor every stage reads/writes through; set
`DATA_DIR` to relocate the whole store (defaults to `data`). Coming from the old
monolithic `mushroom_observations*.csv` files? Run the one-shot migration — it
merges every root CSV (de-duped on uuid), splits by species, and archives the
originals:

```bash
python migrate_data_layout.py --dry-run   # preview
python migrate_data_layout.py             # migrate (archives originals)
```

Stages (run in order — or just `python run_pipeline.py`, which chains them):

1. **`iNat.py`** — pull mushroom observations from iNaturalist (+ elevation and
   weather) → per-species CSVs in `data/species/` (incremental runs merge and
   de-dupe on uuid; `REFRESH_ALL=1` overwrites).
2. **`enrich_with_rasters.py`** — fill in every environmental column for those
   observations → per-species files in `data/enriched/` (resumable; a `.done`
   marker signals completion). `ee_enrich.py` samples each layer from Earth
   Engine *at the observation points*, so nothing bulky is downloaded:

   | Column | Earth Engine dataset |
   | --- | --- |
   | `ndvi` | `COPERNICUS/S2_SR_HARMONIZED` |
   | `soil_moisture` | `ECMWF/ERA5_LAND/DAILY_AGGR` |
   | `prcp_d0..d6` | `UCSB-CHG/CHIRPS/DAILY` |
   | `tmax_d0..d6`, `tmin_d0..d6` | `ECMWF/ERA5_LAND/DAILY_AGGR` |
   | `land_cover` | `ESA/WorldCover/v200` |
   | `elevation`, `slope`, `aspect` | `USGS/SRTMGL1_003` |
   | `solar_exposure`, `wind_exposure`, `water_retention` | derived from the sampled terrain + `MERIT/Hydro/v1_0_1` upstream drainage area |

   Requests are batched by observation *date*: points sharing a date share one
   composite and one `reduceRegions`, so a 7-day weather history costs one round
   trip per date rather than one per observation (the old Open-Meteo stage made
   an HTTP request per observation) or seven file downloads.

3. **`cluster.py`** — KMeans-cluster observations by environmental similarity
   *globally* across all species, writing the `cluster` label back into each
   `data/enriched/` file. Tune the count with `CLUSTER_COUNT` or `--clusters`.

#### Raster fallback

The original path — download the source rasters, then sample them locally — is
still there and runs automatically whenever Earth Engine is unavailable. Each
raster stage only touches rows still missing its column, so after a successful
Earth Engine pass they are no-ops.

To use it deliberately (it is also what `validate_wetness.py raster` and the
Coverage page read):

```bash
FETCH_RASTERS=1 python fetch.py            # ERA5 via CDS, CHIRPS, WorldCover, SRTM DEM
python terrain_pipeline.py                 # derive the terrain-exposure layers
python enrich_with_rasters.py
```

`USE_EARTH_ENGINE=0` (or `SKIP_EARTH_ENGINE=1`) turns the Earth Engine stages off
entirely; `SEQUENTIAL_FETCH=1` stops the fallback downloads running concurrently.
`fetch.py` and `terrain_pipeline.py` are skipped by `run_pipeline.py` while Earth
Engine is available.

### Running in a notebook / Colab

The notebook and the command line run **the same code**. `run_pipeline.run_all()`
is the entry point behind `python run_pipeline.py`, so a notebook never restates
the stage order or repeats a skip rule:

```python
import run_pipeline
run_pipeline.run_all()                 # identical to `python run_pipeline.py`
run_pipeline.run_all(root="/kaggle/working")   # or point it at another checkout
```

`run_all` runs the stages from the repo root — where the per-species store and
raster caches live — and restores the caller's working directory afterwards.
`notebooks/kaggle_pipeline.ipynb` is exactly this: configure credentials,
authenticate Earth Engine, one `run_all()` call, then review the results.

Every module is also import-safe — the run logic lives in functions behind an
`if __name__ == "__main__"` guard — so you can still drive individual steps:

```python
import ee_enrich, species_store as store
df = store.load_all(store.SPECIES_DIR)
ee_enrich.enrich_precip_ee(df, max_workers=4)   # one stage, gentler on EE quota
```

In Colab, `pip install earthengine-api pyinaturalist python-dotenv scikit-learn`
and authenticate Earth Engine with `ee.Authenticate()`.

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

### Raster coverage summary

`raster_coverage.py` scans the environmental-layer cache (CHIRPS precip, ERA5
soil, NDVI, tree cover, DEM, WorldCover) and writes `public/data/coverage.json`
— per layer: file count, date range, on-disk size, and geographic extent, plus
a date→layers index. `run_pipeline.py` runs it after the export step, and the
frontend **Coverage** tab renders it as layer cards plus a date × layer matrix
so gaps are obvious at a glance.

```bash
python raster_coverage.py            # → public/data/coverage.json
python raster_coverage.py --pretty   # human-readable
```

## Nuxt frontend & Netlify deploy

The frontend is a Nuxt 3 app that renders the observations on a Leaflet map,
coloured by environmental cluster, with the enriched attributes in each popup.
It reads a **static GeoJSON** file — no backend or database.

**Filtering.** The **Data** tab is the control centre: pick species, and narrow
by **location** (country / state / county — parsed from each record's place
string — or a lat/lng centre + radius in km) and **time** (year, month, ISO
week, or a from/to date range). Filters live in shared state, so they apply
everywhere at once (map, table, charts, explore); a "Filters: N" chip in the
header links back to the Data tab from any view. When you fetch a *new* species
while location/time filters are set, the fetch is scoped to match (iNaturalist
radius + observed-date range) instead of pulling the whole history.

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

### Sign-in to protect the live-fetch endpoints (Supabase Auth)

Browsing (map, table, charts, explore) is fully open. The endpoints that make
**on-demand outbound API calls** — `fetch-species` (Data tab) and
`run-data-pipeline` — are gated behind **Supabase Auth** so they can't be
hammered anonymously.

- **Server side:** `netlify/lib/auth.mjs` validates the caller's Supabase
  access token (`Authorization: Bearer <jwt>`) via `auth.getUser`. Enforcement
  turns on automatically whenever Supabase is configured (`SUPABASE_URL` +
  `SUPABASE_ANON_KEY` in the Netlify env). Overrides: `AUTH_DISABLED=true`
  forces the endpoints open even when configured; `AUTH_REQUIRED=true` fails
  closed and refuses traffic until Supabase is configured.
- **Browser side:** set the public keys so the login UI works and fetches send
  the token:
  - Netlify / `.env`: `NUXT_PUBLIC_SUPABASE_URL`,
    `NUXT_PUBLIC_SUPABASE_ANON_KEY` (the anon key is public by design; the
    service-role key stays server-only).
- **Enable sign-in methods** in the Supabase dashboard (Authentication →
  Providers): Email (password + magic link) is on by default; enable **GitHub**
  and **Google** OAuth and add your site URL + `…/login` to the redirect
  allow-list. Turn on **Passkeys / WebAuthn** there too. The `/login` page
  surfaces all of these — "Sign in with a passkey" for returning users, and
  "Add a passkey to this account" once you're signed in (the client enables the
  experimental passkey API automatically).

When Supabase public keys are **not** set, the login UI shows a
“not configured” notice and fetches run unauthenticated (which the functions
allow only because the server is likewise unconfigured) — so local dev works
with zero credentials.

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
