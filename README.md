# data-map

A map of environmental conditions around mushroom observations. iNaturalist
observations are enriched with remotely-sensed environmental layers and then
clustered by environmental similarity. The frontend is a Nuxt app.

## Python data pipeline

Install the pipeline dependencies:

```bash
pip install -r requirements.txt
```

Stages (run in order):

1. **`iNat.py`** — pull mushroom observations from iNaturalist (+ elevation and
   weather) → `mushroom_observations.csv`.
2. **`fetch.py`** — download the environmental rasters for those observations:
   NDVI (Sentinel-2 via Earth Engine), soil moisture (ERA5-Land), precipitation
   (CHIRPS), and **topography (SRTM DEM)**. After the DEM downloads it runs
   `terrain_pipeline.process_dem` to derive the terrain-exposure layers.
3. **`terrain_pipeline.py`** — turn the raw DEM into terrain-exposure layers
   (see below). Runs automatically from `fetch.py`, or standalone:
   `python terrain_pipeline.py --dem dem/dem_SRTMGL3.tif`.
4. **`enrich_with_rasters.py`** — sample every raster (including the terrain
   layers) at each observation point → `mushroom_observations_enriched.csv`.
5. **`cluster.py`** — KMeans-cluster observations by environmental similarity →
   `mushroom_clusters.csv`.

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
comparison export a fine-resolution satellite moisture layer (10–20 m) over the
DEM footprint via Earth Engine.

## Nuxt frontend

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
