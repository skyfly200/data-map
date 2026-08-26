import argparse
import concurrent.futures
import time
import pandas as pd
from datetime import timedelta
import rasterio
from rasterio.warp import transform
import xarray as xr
import numpy as np
import math
import os
import shutil


def stage_output_path(input_path, suffix, output_dir='.'):
    if not input_path:
        raise ValueError('Input path is required')
    stem = os.path.splitext(os.path.basename(input_path))[0]
    filename = f"{stem}{suffix}.csv"
    if output_dir in (None, '', '.'):
        return filename
    return os.path.join(output_dir, filename)

# ─── Raster Data Sources ────────────────────────────────────────────────────────
# https://worldcover2020.esa.int/downloader
# https://viewer.terrascope.be/?language=en&bbox=-105.28106099415182,40.204737550661235,-105.23939445065368,40.219963411726525&overlay=true&bgLayer=MapBox_Satellite&date=2025-04-23&layer=WORLDCOVER_2021_MAP
# https://wiki.openstreetmap.org/wiki/Overpass_API
# 
# https://msc.fema.gov/portal/search?AddressQuery=-105.0%2C%2040.0

# ─── Utility Function to Sample Raster Value ──────────────────────────────────

def resolve_raster_path(path):
    """Return an existing raster path, tolerating COG compression.

    ``fetch.py`` compresses downloaded rasters to Cloud-Optimized GeoTIFFs and
    renames ``<name>.tif`` → ``<name>.cog.tif`` (deleting the original). The
    enrichment lookups still ask for ``<name>.tif``; this resolves to the
    ``.cog.tif`` sibling when the plain file is gone, so compression no longer
    breaks enrichment. Returns None if neither exists.
    """
    if os.path.exists(path):
        return path
    stem, _ext = os.path.splitext(path)
    cog = f"{stem}.cog.tif"
    return cog if os.path.exists(cog) else None


def _progress(label, i, total, start, stride=None):
    """Throttled one-line [i/total] progress with percent, elapsed and ETA.

    Prints at most ~40 updates across the loop (plus the final one), so long
    enrichment passes show steady overall progress without flooding the log.
    """
    stride = stride or max(1, total // 40)
    if i != total and i % stride:
        return
    pct = i / total * 100 if total else 100.0
    elapsed = time.monotonic() - start
    eta = (elapsed / i) * (total - i) if i else 0.0
    print(f"  [{i}/{total}] {pct:5.1f}% {label} · {elapsed:4.0f}s elapsed, ~{eta:4.0f}s left", flush=True)


def sample_raster_value(tif_path, lon, lat, scale_factor=1.0, nodata_val=None):
    """
    Samples a raster file at the given longitude and latitude.
    Optionally applies a scale factor and respects a nodata value.
    """
    try:
        with rasterio.open(tif_path) as src:
            x, y = transform('EPSG:4326', src.crs, [lon], [lat])
            row, col = src.index(x[0], y[0])

            if not (0 <= row < src.height and 0 <= col < src.width):
                print(f"[!] Point ({lat}, {lon}) is outside raster bounds of {tif_path}")
                return None

            value = src.read(1)[row, col]

            # Apply nodata masking if defined
            if nodata_val is not None and value == nodata_val:
                return None
            if src.nodata is not None and value == src.nodata:
                return None

            return value * scale_factor
    except Exception as e:
        print(f"[!] Error sampling raster at ({lon}, {lat}) in {tif_path}: {e}")
        return None

def get_needed_raster_dates(df, buffer_days=6):
    if 'date' not in df.columns:
        raise ValueError("CSV must contain a 'date' column in YYYY-MM-DD format.")

    all_dates = set()
    for d in pd.to_datetime(df['date'].dropna()):
        for i in range(buffer_days + 1):
            all_dates.add((d - timedelta(days=i)).strftime('%Y-%m-%d'))

    return sorted(all_dates)

# ─── Precipitation Utilities ──────────────────────────────────────────────────
def enrich_with_precip(df, precip_dir="precip/"):
    for d in range(7):
        df[f'prcp_d{d}'] = None

    unique_dates = pd.to_datetime(df['date'].dropna().unique())
    total = len(unique_dates)
    total_rows = int(df['date'].notna().sum())
    print(f"Adding 7-day precipitation history — {total} dates, {total_rows} observations...")

    start = time.monotonic()
    missing = set()
    done_rows = 0
    for i, date in enumerate(sorted(unique_dates), 1):
        dstr = date.strftime('%Y-%m-%d')
        day_rows = df[df['date'] == dstr]          # compute the date group once
        for d in range(7):
            target_date = (date - timedelta(days=d)).strftime('%Y-%m-%d')
            tif_path = resolve_raster_path(os.path.join(precip_dir, f"precip_{target_date}.tif"))
            if not tif_path:
                missing.add(target_date)
                continue
            for idx, row in day_rows.iterrows():
                df.at[idx, f'prcp_d{d}'] = sample_raster_value(tif_path, row.lon, row.lat)

        done_rows += len(day_rows)
        pct = i / total * 100
        elapsed = time.monotonic() - start
        eta = (elapsed / i) * (total - i)
        print(f"  [{i}/{total}] {pct:5.1f}%  {dstr} ({len(day_rows)} obs)  "
              f"· {done_rows}/{total_rows} obs · {elapsed:4.0f}s elapsed, ~{eta:4.0f}s left",
              flush=True)

    if missing:
        print(f"[!] {len(missing)} precip date(s) had no raster (points keep null rain for those days).")
    print(f"✅ Precipitation history done in {time.monotonic() - start:.0f}s.")
    return df

# ─── Land Cover Utilities ──────────────────────────────────────────────────
def get_worldcover_tile_name(lat, lon):
    # These tiles start at whole degrees divisible by 3
    lat_deg = math.floor(lat / 3) * 3
    lon_deg = math.floor(lon / 3) * 3

    lat_prefix = "N" if lat_deg >= 0 else "S"
    lon_prefix = "E" if lon_deg >= 0 else "W"
    lat_str = f"{abs(lat_deg):02d}"
    lon_str = f"{abs(lon_deg):03d}"
    return f"ESA_WorldCover_10m_2020_v100_{lat_prefix}{lat_str}{lon_prefix}{lon_str}_Map.tif"

def enrich_with_worldcover(df, base_dir="./world_cover/"):
    df['land_cover'] = None
    total = len(df)
    print(f"Adding WorldCover land class — {total} observations...")

    start = time.monotonic()
    resolved = {}   # tile name → resolved path (or None); warn once per tile
    missing = set()
    for n, (idx, row) in enumerate(df.iterrows(), 1):
        tile_name = get_worldcover_tile_name(row.lat, row.lon)
        if tile_name not in resolved:
            resolved[tile_name] = resolve_raster_path(os.path.join(base_dir, tile_name))
            if not resolved[tile_name]:
                missing.add(tile_name)
                print(f"[!] WorldCover tile not cached: {tile_name} (run fetch.py to download it)")

        tile_path = resolved[tile_name]
        if tile_path:
            df.at[idx, 'land_cover'] = sample_raster_value(tile_path, row.lon, row.lat, scale_factor=1, nodata_val=255)
        _progress("land cover", n, total, start)

    if missing:
        print(f"[!] {len(missing)} WorldCover tile(s) missing; those points have no land cover.")
    return df

ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Water",
    90: "Wetland",
    95: "Mangroves",
    100: "Moss and lichen"
}

NON_PRODUCTIVE_LANDCOVER_CODES = {50, 70, 80}


def should_filter_non_productive_landcover(env=None):
    values = {**os.environ, **(env or {})}
    raw = values.get('FILTER_NON_PRODUCTIVE_LANDCOVER', values.get('DISABLE_NON_PRODUCTIVE_FILTER', '1'))
    if raw is None:
        return True
    value = str(raw).strip().lower()
    if value in {'0', 'false', 'no', 'n', 'off'}:
        return False
    if value in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    return True


def add_worldcover_labels(df):
    df['land_cover_label'] = df['land_cover'].map(ESA_WORLDCOVER_CLASSES)
    return df


def filter_non_productive_landcover(df, landcover_col='land_cover', label_col='land_cover_label'):
    """Flag built-up, snow/ice, and open-water cells before dropping them.

    The boolean flag preserves the information for later analysis while the rows are
    removed from the canonical terrestrial dataset. This keeps the training/analysis
    data cleaner without losing the fact that these observations were excluded.
    """
    if landcover_col not in df.columns:
        return df

    before = len(df)
    df = df.copy()
    df['water_mask'] = df[landcover_col].apply(
        lambda value: bool(value is not None and not pd.isna(value) and int(float(value)) in NON_PRODUCTIVE_LANDCOVER_CODES)
    )
    df['exclude_reason'] = df[landcover_col].apply(
        lambda value: (
            'non_terrestrial' if value is not None and not pd.isna(value) and int(float(value)) in NON_PRODUCTIVE_LANDCOVER_CODES
            else 'keep'
        )
    )
    valid_mask = ~df['water_mask']
    filtered = df.loc[valid_mask].copy()
    if label_col in filtered.columns:
        filtered[label_col] = filtered[label_col].where(~filtered['water_mask'], 'Filtered out')
    dropped = before - len(filtered)
    if dropped:
        print(f"Filtering out {dropped} non-productive land-cover rows ({sorted(NON_PRODUCTIVE_LANDCOVER_CODES)})")
    return filtered

# ─── Terrain / Topography Utilities ───────────────────────────────────────────
# Layers produced by terrain_pipeline.process_dem(). Each is a single static
# GeoTIFF (topography does not change over time), so we sample every point once.
TERRAIN_LAYERS = ["slope", "aspect", "solar_exposure", "wind_exposure", "water_retention"]


def enrich_with_terrain(df, terrain_dir="dem/derived/"):
    """Sample the DEM-derived terrain layers at each observation point.

    Adds columns: slope, aspect, solar_exposure, wind_exposure, water_retention.
    Run ``fetch.py`` (or ``terrain_pipeline.py``) first to generate the layers.
    """
    print("Adding terrain exposure (solar / wind / water retention)...")

    layer_paths = {}
    missing = []
    for name in TERRAIN_LAYERS:
        df[name] = None
        path = resolve_raster_path(os.path.join(terrain_dir, f"{name}.tif"))
        if path:
            layer_paths[name] = path
        else:
            missing.append(name)

    if not layer_paths:
        print(f"[!] No terrain layers in {terrain_dir} — skipping terrain enrichment.")
        print("    Generate them with `python terrain_pipeline.py` (it needs a DEM in dem/; "
              "set OPENTOPOGRAPHY_API_KEY to auto-download one), then re-run enrichment.")
        return df
    if missing:
        print(f"[!] Terrain layers missing: {', '.join(missing)} (re-run terrain_pipeline.py).")

    total = len(df)
    print(f"Adding terrain exposure ({len(layer_paths)} layers × {total} observations)...")
    start = time.monotonic()
    for n, (idx, row) in enumerate(df.iterrows(), 1):
        if not (pd.isna(row.get("lat")) or pd.isna(row.get("lon"))):
            for name, path in layer_paths.items():
                df.at[idx, name] = sample_raster_value(path, row.lon, row.lat)
        _progress("terrain", n, total, start)

    return df

# ─── Temperature History Utilities ────────────────────────────────────────────
# Daily high/low for the days leading up to each observation, from the keyless
# Open-Meteo historical archive. Pairs with the CHIRPS precip history
# (prcp_d0..d6) to chart the weather run-up to each find. d0 = observation day,
# d6 = six days before.

def enrich_with_temperature_history(df, days=7, max_workers=10):
    import requests

    for d in range(days):
        df[f'tmax_d{d}'] = None
        df[f'tmin_d{d}'] = None

    url = "https://archive-api.open-meteo.com/v1/archive"
    tasks = [
        (idx, float(row['lat']), float(row['lon']), pd.to_datetime(row['date']))
        for idx, row in df.iterrows()
        if not (pd.isna(row.get('lat')) or pd.isna(row.get('lon')) or pd.isna(row.get('date')))
    ]
    total = len(tasks)
    print(f"Adding {days}-day temperature history (Open-Meteo) — {total} points, {max_workers} parallel...")

    def worker(task):
        idx, lat, lon, obs_date = task
        start = (obs_date - timedelta(days=days - 1)).strftime('%Y-%m-%d')
        end = obs_date.strftime('%Y-%m-%d')
        try:
            r = requests.get(url, params={
                "latitude": lat, "longitude": lon,
                "start_date": start, "end_date": end,
                "daily": "temperature_2m_max,temperature_2m_min",
                "timezone": "UTC",
            }, timeout=30)
            r.raise_for_status()
            daily = r.json().get('daily', {})
            highs = daily.get('temperature_2m_max', [])
            lows = daily.get('temperature_2m_min', [])
            # API returns ascending dates: index 0 = oldest (d{days-1}), last = d0.
            n = len(highs)
            out = {}
            for d in range(days):
                j = n - 1 - d
                if 0 <= j < n:
                    out[f'tmax_d{d}'] = highs[j]
                    out[f'tmin_d{d}'] = lows[j]
            return idx, out, None
        except Exception as e:
            return idx, None, str(e)

    start_t = time.monotonic()
    fails = 0
    # Requests are independent and network-bound → thread pool; df writes stay on
    # the main thread as results arrive.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            idx, out, err = fut.result()
            if err:
                fails += 1
            elif out:
                for col, val in out.items():
                    df.at[idx, col] = val
            _progress("temperature", i, total, start_t)

    if fails:
        print(f"[!] {fails}/{total} temperature request(s) failed.")
    return df

# ─── Soil Moisture Utilities ──────────────────────────────────────────────────

def load_soil_moisture_dataset(nc_path):
    return xr.open_dataset(nc_path, engine='netcdf4')

def extract_soil_moisture(ds, lat, lon, date_str):
    try:
        date = np.datetime64(date_str)
        if 'time' in ds.dims:
            time_dim = 'time'
        elif 'valid_time' in ds.dims:
            time_dim = 'valid_time'
        else:
            raise ValueError(f"No recognizable time dimension in dataset: {ds.dims}")

        ds_time = ds.sel({time_dim: date}, method="nearest")
        value = ds_time['swvl1'].interp(latitude=lat, longitude=lon).values.item()
        return value
    except Exception as e:
        print(f"[!] Soil moisture not found for ({lat}, {lon}) on {date_str}: {e}")
        return None

# ─── NDVI Utilities ───────────────────────────────────────────────────────────

def get_ndvi_from_raster(tif_path, lon, lat):
    try:
        with rasterio.open(tif_path) as src:
            x, y = transform('EPSG:4326', src.crs, [lon], [lat])
            row, col = src.index(x[0], y[0])
            ndvi_value = src.read(1)[row, col]
            if ndvi_value != src.nodata:
                return ndvi_value / 10000.0
    except Exception as e:
        print(f"[!] NDVI missing for ({lat}, {lon}) in {tif_path}: {e}")
    return None

def fill_missing_ndvi(df, max_days_gap=7):
    filled = 0
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    for idx, row in df[df['ndvi'].isnull()].iterrows():
        lat = row['lat']
        lon = row['lon']
        date = row['date']

        # Look for exact same location
        candidates = df[
            (df['ndvi'].notnull()) &
            (df['lat'] == lat) &
            (df['lon'] == lon)
        ].copy()

        if candidates.empty:
            continue

        # Find nearest date within allowed window
        candidates['date_diff'] = candidates['date'].apply(lambda d: abs((d - date).days))
        nearest = candidates[candidates['date_diff'] <= max_days_gap].sort_values('date_diff')

        if not nearest.empty:
            fill_val = nearest.iloc[0]['ndvi']
            df.at[idx, 'ndvi'] = fill_val
            filled += 1

    print(f"✅ Filled {filled} NDVI values using same-location fallback.")
    return df

# ─── NDVI via direct Earth Engine point sampling ──────────────────────────────
# The old approach exported a Sentinel-2 NDVI GeoTIFF per observation to Google
# Drive (fetch.py:fetch_sentinel2_ndvi) — asynchronous, and nothing downloaded
# the tiles back, so the `ndvi` column stayed empty. This samples the NDVI value
# directly at each point with reduceRegion().getInfo(): no Drive round-trip, and
# it populates the column in a single (Earth-Engine-authenticated) run.

def enrich_with_ndvi_ee(df, buffer_days=15, scale=10, cloud_pct=60, max_workers=8):
    if os.environ.get('SKIP_EARTH_ENGINE') == '1':
        print('Skipping Earth Engine NDVI sampling (SKIP_EARTH_ENGINE=1).')
        return df

    try:
        import ee
        try:
            ee.Initialize(project=os.environ.get('EARTHENGINE_PROJECT'))
        except Exception:
            ee.Authenticate(quiet=True)
            ee.Initialize(project=os.environ.get('EARTHENGINE_PROJECT'))
    except Exception as e:
        print(f'[!] Earth Engine unavailable — NDVI sampling skipped: {e}')
        return df

    if 'ndvi' not in df.columns:
        df['ndvi'] = None

    tasks = [
        (idx, float(row['lon']), float(row['lat']), pd.to_datetime(row['date']))
        for idx, row in df.iterrows()
        if not (pd.isna(row.get('lat')) or pd.isna(row.get('lon')) or pd.isna(row.get('date')))
    ]
    total = len(tasks)
    print(f'Sampling Sentinel-2 NDVI per point (±{buffer_days}d, {scale} m) via Earth Engine '
          f'— {total} points, {max_workers} parallel...')

    def worker(task):
        idx, lon, lat, date = task
        start = (date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        end = (date + timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        try:
            point = ee.Geometry.Point([lon, lat])
            collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                          .filterDate(start, end)
                          .filterBounds(point)
                          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                          .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')))
            value = (collection.median()
                     .reduceRegion(ee.Reducer.mean(), point, scale)
                     .get('NDVI').getInfo())
            return idx, value, None
        except Exception as e:
            return idx, None, str(e)

    sampled = fails = 0
    t0 = time.monotonic()
    # EE getInfo() calls are independent network requests → run them concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            idx, value, err = fut.result()
            if err:
                fails += 1
            elif value is not None:
                df.at[idx, 'ndvi'] = value
                sampled += 1
            _progress(f"NDVI ({sampled} sampled)", i, total, t0)

    if fails:
        print(f"[!] {fails}/{total} NDVI request(s) failed.")
    print(f'✅ NDVI sampled for {sampled}/{total} points.')
    return df

# ─── Main Enrichment Script ───────────────────────────────────────────────────

def enrich_df_with_rasters(df, ndvi_dir='ndvi/', soil_dir='soil/'):
    df['ndvi'] = None
    df['soil_moisture'] = None

    unique_dates = sorted(set(df['date'].dropna()))
    total = len(unique_dates)
    print(f"Adding NDVI + soil moisture — {total} unique dates...")

    start = time.monotonic()
    n_ndvi = n_soil = n_skip = 0
    for i, date_str in enumerate(unique_dates, 1):
        date_df = df[df['date'] == date_str]
        _progress("NDVI/soil", i, total, start)

        lat = date_df['lat'].iloc[0]
        lon = date_df['lon'].iloc[0]
        ndvi_path = resolve_raster_path(os.path.join(ndvi_dir, f"ndvi_{date_str}_{lat:.4f}_{lon:.4f}.tif"))
        soil_path = os.path.join(soil_dir, f"soil_{date_str}.nc")
        has_ndvi = ndvi_path is not None
        has_soil = os.path.exists(soil_path)

        if not has_ndvi and not has_soil:
            n_skip += 1
            continue
        n_ndvi += has_ndvi
        n_soil += has_soil
        soil_ds = load_soil_moisture_dataset(soil_path) if has_soil else None

        for idx, row in date_df.iterrows():
            if has_ndvi:
                df.at[idx, 'ndvi'] = get_ndvi_from_raster(ndvi_path, row.lon, row.lat)
            if has_soil:
                df.at[idx, 'soil_moisture'] = extract_soil_moisture(soil_ds, row.lat, row.lon, row.date)

    print(f"✅ NDVI/soil pass done — {n_ndvi} date(s) with NDVI, {n_soil} with soil, "
          f"{n_skip} with neither ({total} total).")
    return df

# ─── Script Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich observation rows with raster and terrain data")
    parser.add_argument("--input", default="mushroom_observations.csv", help="Input observation CSV")
    parser.add_argument("--output", default=None, help="Output enriched CSV path")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output or stage_output_path(input_file, '_enriched')

    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)

    core_dates = get_needed_raster_dates(df, 0)
    precip_dates = get_needed_raster_dates(df)
    print("Total dates needed (for raster):", len(core_dates))
    print("Total dates needed (for precip):", len(precip_dates))
    # print(precip_dates)

    print("Starting raster-based enrichment...")
    df = enrich_df_with_rasters(df, ndvi_dir="ndvi/", soil_dir="soil/")
    df = enrich_with_precip(df, precip_dir="precip/")
    df = enrich_with_worldcover(df)
    df = add_worldcover_labels(df)
    if 'water_mask' not in df.columns:
        df['water_mask'] = False
    if 'exclude_reason' not in df.columns:
        df['exclude_reason'] = 'keep'
    if should_filter_non_productive_landcover():
        df = filter_non_productive_landcover(df)
    df = enrich_with_terrain(df)
    df = enrich_with_temperature_history(df)
    # NDVI sampled directly from Earth Engine (replaces the Drive-export path).
    df = enrich_with_ndvi_ee(df)

    # 🧠 Fill missing NDVI using same-location fallback
    print("Filling missing NDVI...")
    df = fill_missing_ndvi(df, max_days_gap=7)

    print(f"Saving enriched data to {output_file}...")
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    print("Done ✅")
