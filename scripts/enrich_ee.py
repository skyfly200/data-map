%%writefile enrich_with_rasters.py
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
import sys
import glob

import species_store as store

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def stage_output_path(input_path, suffix, output_dir='.'):
    if not input_path:
        raise ValueError('Input path is required')
    stem = os.path.splitext(os.path.basename(input_path))[0]
    filename = f"{stem}{suffix}.csv"
    if output_dir in (None, '', '.'):
        return filename
    return os.path.join(output_dir, filename)

# ─── Raster Data Sources ────────────────────────────────────────────────────────

def resolve_raster_path(path):
    """Return an existing raster path, tolerating COG compression.

    ``fetch.py`` compresses downloaded rasters to Cloud-Optimized GeoTIFFs and
    renames ``<name>.tif`` → ``<name>.cog.tif`` (deleting the original).
    """
    if os.path.exists(path):
        return path
    stem, _ext = os.path.splitext(path)
    cog = f"{stem}.cog.tif"
    return cog if os.path.exists(cog) else None


def _progress(label, i, total, start, stride=None):
    stride = stride or max(1, total // 40)
    if i != total and i % stride:
        return
    pct = i / total * 100 if total else 100.0
    elapsed = time.monotonic() - start
    eta = (elapsed / i) * (total - i) if i else 0.0
    print(f"  [{i}/{total}] {pct:5.1f}% {label} · {elapsed:4.0f}s elapsed, ~{eta:4.0f}s left", flush=True)


def sample_raster_value(tif_path, lon, lat, scale_factor=1.0, nodata_val=None):
    try:
        with rasterio.open(tif_path) as src:
            x, y = transform('EPSG:4326', src.crs, [lon], [lat])
            row, col = src.index(x[0], y[0])

            if not (0 <= row < src.height and 0 <= col < src.width):
                return None

            value = src.read(1)[row, col]

            if nodata_val is not None and value == nodata_val:
                return None
            if src.nodata is not None and value == src.nodata:
                return None

            return value * scale_factor
    except Exception as e:
        return None


def sample_raster_points(tif_path, points, scale_factor=1.0, nodata_val=None):
    out = [None] * len(points)
    if not points:
        return out
    try:
        with rasterio.open(tif_path) as src:
            band = src.read(1)
            h, w = band.shape
            src_nodata = src.nodata
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            X, Y = transform('EPSG:4326', src.crs, xs, ys)
            for i, (x, y) in enumerate(zip(X, Y)):
                r, c = src.index(x, y)
                if not (0 <= r < h and 0 <= c < w):
                    continue
                v = band[r, c]
                if nodata_val is not None and v == nodata_val:
                    continue
                if src_nodata is not None and v == src_nodata:
                    continue
                if isinstance(v, float) and math.isnan(v):
                    continue
                out[i] = float(v) * scale_factor
    except Exception as e:
        pass
    return out

def get_needed_raster_dates(df, buffer_days=6):
    if 'date' not in df.columns:
        raise ValueError("CSV must contain a 'date' column in YYYY-MM-DD format.")

    all_dates = set()
    for d in pd.to_datetime(df['date'].dropna()):
        for i in range(buffer_days + 1):
            all_dates.add((d - timedelta(days=i)).strftime('%Y-%m-%d'))

    return sorted(all_dates)

# ─── Precipitation Utilities ──────────────────────────────────────────────────
def enrich_with_precip(df, precip_dir="precip/", checkpoint=None):
    for d in range(7):
        if f'prcp_d{d}' not in df.columns:
            df[f'prcp_d{d}'] = None

    all_dates = sorted(pd.to_datetime(df['date'].dropna().unique()))
    pending = [d for d in all_dates
               if df.loc[df['date'] == d.strftime('%Y-%m-%d'), 'prcp_d0'].isna().any()]
    total = len(pending)
    if not total:
        return df
    total_rows = int(sum(len(df[df['date'] == d.strftime('%Y-%m-%d')]) for d in pending))
    print(f"Adding 7-day precipitation history — {total} dates, {total_rows} observations...")

    start = time.monotonic()
    missing = set()
    done_rows = 0
    for i, date in enumerate(pending, 1):
        dstr = date.strftime('%Y-%m-%d')
        day_rows = df[df['date'] == dstr]
        idxs = list(day_rows.index)
        coords = list(zip(day_rows['lon'], day_rows['lat']))
        for d in range(7):
            target_date = (date - timedelta(days=d)).strftime('%Y-%m-%d')
            tif_path = resolve_raster_path(os.path.join(precip_dir, f"precip_{target_date}.tif"))
            if not tif_path:
                missing.add(target_date)
                continue
            vals = sample_raster_points(tif_path, coords)
            for idx, v in zip(idxs, vals):
                df.at[idx, f'prcp_d{d}'] = v

        done_rows += len(day_rows)
        if checkpoint and i % 50 == 0:
            checkpoint()

    print(f"✅ Precipitation history done in {time.monotonic() - start:.0f}s.")
    return df

# ─── Land Cover Utilities ──────────────────────────────────────────────────
def get_worldcover_tile_name(lat, lon):
    lat_deg = math.floor(lat / 3) * 3
    lon_deg = math.floor(lon / 3) * 3

    lat_prefix = "N" if lat_deg >= 0 else "S"
    lon_prefix = "E" if lon_deg >= 0 else "W"
    lat_str = f"{abs(lat_deg):02d}"
    lon_str = f"{abs(lon_deg):03d}"
    return f"ESA_WorldCover_10m_2020_v100_{lat_prefix}{lat_str}{lon_prefix}{lon_str}_Map.tif"

def enrich_with_worldcover(df, base_dir="./world_cover/", checkpoint=None):
    if 'land_cover' not in df.columns:
        df['land_cover'] = None
    pending = [(idx, row) for idx, row in df.iterrows() if pd.isna(row.get('land_cover'))]
    total = len(pending)
    if not total:
        return df
    print(f"Adding WorldCover land class — {total} observations...")

    start = time.monotonic()
    by_tile = {}
    for idx, row in pending:
        by_tile.setdefault(get_worldcover_tile_name(row.lat, row.lon), []).append((idx, row.lon, row.lat))

    missing = set()
    done = 0
    for t, (tile_name, group) in enumerate(by_tile.items(), 1):
        tile_path = resolve_raster_path(os.path.join(base_dir, tile_name))
        if not tile_path:
            missing.add(tile_name)
            print(f"[!] WorldCover tile not cached: {tile_name}")
        else:
            vals = sample_raster_points(tile_path, [(lon, lat) for _, lon, lat in group], scale_factor=1, nodata_val=255)
            for (idx, _, _), v in zip(group, vals):
                df.at[idx, 'land_cover'] = v
        done += len(group)
        _progress("land cover", done, total, start)
        if checkpoint:
            checkpoint()
    return df

ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare / sparse vegetation", 70: "Snow and ice",
    80: "Water", 90: "Wetland", 95: "Mangroves", 100: "Moss and lichen"
}

NON_PRODUCTIVE_LANDCOVER_CODES = {50, 70, 80}

def should_filter_non_productive_landcover(env=None):
    return True

def add_worldcover_labels(df):
    df['land_cover_label'] = df['land_cover'].map(ESA_WORLDCOVER_CLASSES)
    return df

def filter_non_productive_landcover(df, landcover_col='land_cover', label_col='land_cover_label'):
    if landcover_col not in df.columns:
        return df
    before = len(df)
    df = df.copy()
    df['water_mask'] = df[landcover_col].apply(
        lambda value: bool(value is not None and not pd.isna(value) and int(float(value)) in NON_PRODUCTIVE_LANDCOVER_CODES)
    )
    df['exclude_reason'] = df[landcover_col].apply(
        lambda value: ('non_terrestrial' if value is not None and not pd.isna(value) and int(float(value)) in NON_PRODUCTIVE_LANDCOVER_CODES else 'keep')
    )
    valid_mask = ~df['water_mask']
    filtered = df.loc[valid_mask].copy()
    if label_col in filtered.columns:
        filtered[label_col] = filtered[label_col].where(~filtered['water_mask'], 'Filtered out')
    return filtered

# ─── Terrain / Topography Utilities ───────────────────────────────────────────
TERRAIN_LAYERS = ["slope", "aspect", "solar_exposure", "wind_exposure", "water_retention"]

def enrich_with_terrain(df, terrain_dir="dem/derived/", checkpoint=None):
    layer_paths = {}
    missing = []
    for name in TERRAIN_LAYERS:
        if name not in df.columns:
            df[name] = None
        # Use glob to support coordinate-encoded filenames like slope_N46.3_N37.0_W124.6_W102.0.tif
        matches = glob.glob(os.path.join(terrain_dir, f"{name}*.tif"))
        path = resolve_raster_path(matches[0]) if matches else None
        if path:
            layer_paths[name] = path
        else:
            missing.append(name)

    if not layer_paths:
        print(f"[!] No terrain layers in {terrain_dir} — skipping terrain enrichment.")
        return df

    first = next(iter(layer_paths))
    pending = [(idx, row) for idx, row in df.iterrows()
               if not (pd.isna(row.get("lat")) or pd.isna(row.get("lon"))) and pd.isna(row.get(first))]
    total = len(pending)
    if not total:
        return df
    print(f"Adding terrain exposure ({len(layer_paths)} layers × {total} observations)...")
    start = time.monotonic()
    idxs = [idx for idx, _ in pending]
    coords = [(row.lon, row.lat) for _, row in pending]
    for li, (name, path) in enumerate(layer_paths.items(), 1):
        vals = sample_raster_points(path, coords)
        for idx, v in zip(idxs, vals):
            df.at[idx, name] = v
        _progress(f"terrain ({name})", li, len(layer_paths), start)
        if checkpoint:
            checkpoint()
    return df

# ─── Temperature History Utilities ────────────────────────────────────────────
def enrich_with_temperature_history(df, days=7, max_workers=10, checkpoint=None):
    import requests
    for d in range(days):
        for col in (f'tmax_d{d}', f'tmin_d{d}'):
            if col not in df.columns:
                df[col] = None

    url = "https://archive-api.open-meteo.com/v1/archive"
    tasks = [
        (idx, float(row['lat']), float(row['lon']), pd.to_datetime(row['date']))
        for idx, row in df.iterrows()
        if not (pd.isna(row.get('lat')) or pd.isna(row.get('lon')) or pd.isna(row.get('date')))
        and pd.isna(row.get('tmax_d0'))
    ]
    total = len(tasks)
    if not total:
        return df
    print(f"Adding {days}-day temperature history (Open-Meteo) — {total} points...")

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
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            idx, out, err = fut.result()
            if out:
                for col, val in out.items():
                    df.at[idx, col] = val
            if checkpoint and i % 500 == 0:
                checkpoint()
    return df

# ─── Soil Moisture Utilities ──────────────────────────────────────────────────
def load_soil_moisture_dataset(nc_path):
    return xr.open_dataset(nc_path, engine='netcdf4')

def extract_soil_moisture(ds, lat, lon, date_str):
    try:
        date = np.datetime64(date_str)
        time_dim = 'time' if 'time' in ds.dims else ('valid_time' if 'valid_time' in ds.dims else None)
        if not time_dim:
            return None
        ds_time = ds.sel({time_dim: date}, method="nearest")
        value = ds_time['swvl1'].interp(latitude=lat, longitude=lon).values.item()
        return value
    except Exception as e:
        return None

# ─── NDVI Utilities & EE ──────────────────────────────────────────────────────
def get_ndvi_from_raster(tif_path, lon, lat):
    try:
        with rasterio.open(tif_path) as src:
            x, y = transform('EPSG:4326', src.crs, [lon], [lat])
            row, col = src.index(x[0], y[0])
            ndvi_value = src.read(1)[row, col]
            if ndvi_value != src.nodata:
                return ndvi_value / 10000.0
    except Exception as e:
        pass
    return None

def fill_missing_ndvi(df, max_days_gap=7):
    return df

def enrich_with_ndvi_ee(df, buffer_days=15, scale=10, cloud_pct=60, max_workers=8, checkpoint=None):
    if os.environ.get('SKIP_EARTH_ENGINE') == '1':
        return df
    try:
        import ee
        try:
            ee.Initialize(project=os.environ.get('EARTHENGINE_PROJECT'))
        except Exception:
            ee.Authenticate(quiet=True)
            ee.Initialize(project=os.environ.get('EARTHENGINE_PROJECT'))
    except Exception as e:
        return df

    if 'ndvi' not in df.columns:
        df['ndvi'] = None

    tasks = [
        (idx, float(row['lon']), float(row['lat']), pd.to_datetime(row['date']))
        for idx, row in df.iterrows()
        if not (pd.isna(row.get('lat')) or pd.isna(row.get('lon')) or pd.isna(row.get('date')))
        and pd.isna(row.get('ndvi'))
    ]
    total = len(tasks)
    if not total:
        return df
    print(f'Sampling Sentinel-2 NDVI per point (±{buffer_days}d, {scale} m) via Earth Engine — {total} points...')

    by_date = {}
    for idx, lon, lat, date in tasks:
        by_date.setdefault(pd.Timestamp(date).normalize(), []).append((idx, lon, lat))

    def worker(item):
        date, pts = item
        start = (date - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        end = (date + timedelta(days=buffer_days)).strftime('%Y-%m-%d')
        results = {}
        try:
            lons = [p[1] for p in pts]
            lats = [p[2] for p in pts]
            region = ee.Geometry.Rectangle([min(lons) - 0.05, min(lats) - 0.05,
                                            max(lons) + 0.05, max(lats) + 0.05])
            # Updated to COPERNICUS/S2_SR_HARMONIZED
            ndvi = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterDate(start, end)
                    .filterBounds(region)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                    .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
                    .median())
            fc = ee.FeatureCollection([
                ee.Feature(ee.Geometry.Point([lon, lat]), {'ridx': int(idx)})
                for idx, lon, lat in pts
            ])
            reduced = ndvi.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=scale).getInfo()
            for feat in reduced.get('features', []):
                props = feat.get('properties', {})
                results[props.get('ridx')] = props.get('NDVI')
            return date, results, None
        except Exception as e:
            return date, results, str(e)

    sampled = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, item) for item in by_date.items()]
        for fut in concurrent.futures.as_completed(futures):
            _, results, _ = fut.result()
            for ridx, val in results.items():
                if ridx is not None and val is not None:
                    df.at[ridx, 'ndvi'] = val
                    sampled += 1
    return df

def enrich_df_with_rasters(df, ndvi_dir='ndvi/', soil_dir='soil/'):
    for col in ('ndvi', 'soil_moisture'):
        if col not in df.columns:
            df[col] = None

    unique_dates = sorted(set(df['date'].dropna()))
    total = len(unique_dates)
    print(f"Adding NDVI + soil moisture — {total} unique dates...")

    for i, date_str in enumerate(unique_dates, 1):
        date_df = df[df['date'] == date_str]
        soil_path = os.path.join(soil_dir, f"soil_{date_str}.nc")
        has_soil = os.path.exists(soil_path)
        soil_ds = load_soil_moisture_dataset(soil_path) if has_soil else None

        for idx, row in date_df.iterrows():
            if has_soil:
                df.at[idx, 'soil_moisture'] = extract_soil_moisture(soil_ds, row.lat, row.lon, row.date)
    return df

def _checkpoint(df, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _merge_enriched(base, prior):
    if prior is None or prior.empty or 'uuid' not in base.columns or 'uuid' not in prior.columns:
        return base
    prior = prior.drop_duplicates(subset='uuid', keep='last').set_index('uuid')
    df = base.copy()
    for col in prior.columns:
        if col not in df.columns:
            df[col] = df['uuid'].map(prior[col])
    return df

def run_stage(label, fn, df, path):
    print(f"\n=== {label} ===")
    try:
        df = fn(df)
    except Exception as exc:
        print(f"[!] Stage '{label}' failed ({exc}); continuing with partial data.")
    _checkpoint(df, path)
    print(f"  💾 checkpoint → {path}")
    return df

def _postprocess_landcover(df):
    df = add_worldcover_labels(df)
    if 'water_mask' not in df.columns:
        df['water_mask'] = False
    if 'exclude_reason' not in df.columns:
        df['exclude_reason'] = 'keep'
    if should_filter_non_productive_landcover():
        df = filter_non_productive_landcover(df)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    store_mode = args.input is None
    if store_mode:
        base = store.load_all(store.SPECIES_DIR)
        os.makedirs(store.ENRICHED_DIR, exist_ok=True)
        checkpoint_file = os.path.join(store.ENRICHED_DIR, '_checkpoint.csv')
        done_marker = store.ENRICHED_DONE
        prior = pd.read_csv(checkpoint_file) if os.path.exists(checkpoint_file) else None
        df = _merge_enriched(base, prior)
        def save():
            _checkpoint(df, checkpoint_file)
    else:
        input_file = args.input
        output_file = args.output or stage_output_path(input_file, '_enriched')
        done_marker = f"{output_file}.done"
        df = pd.read_csv(input_file)
        def save():
            _checkpoint(df, output_file)

    checkpoint_path = checkpoint_file if store_mode else output_file
    
    PRECIP_DIR = "/kaggle/input/datasets/skylerflywilson/chirps-precipitation/precip/"
    WORLDCOVER_DIR = "/kaggle/input/datasets/skylerflywilson/world_cover/"
    NDVI_DIR = "/kaggle/input/YOUR_DATASET_NAME/ndvi/"
    SOIL_DIR = "/kaggle/input/datasets/skylerflywilson/era5-sat-soil-moisture/soil/"
    TERRAIN_DIR = "/kaggle/working/dem/derived/"

    df = run_stage("NDVI + soil (cached rasters)", lambda d: enrich_df_with_rasters(d, ndvi_dir=NDVI_DIR, soil_dir=SOIL_DIR), df, checkpoint_path)
    df = run_stage("Precipitation history", lambda d: enrich_with_precip(d, precip_dir=PRECIP_DIR, checkpoint=save), df, checkpoint_path)
    df = run_stage("Land cover", lambda d: enrich_with_worldcover(d, base_dir=WORLDCOVER_DIR, checkpoint=save), df, checkpoint_path)
    df = run_stage("Land-cover labels / filter", _postprocess_landcover, df, checkpoint_path)
    df = run_stage("Terrain exposure", lambda d: enrich_with_terrain(d, terrain_dir=TERRAIN_DIR, checkpoint=save), df, checkpoint_path)
    df = run_stage(
        "NDVI (Earth Engine)",
        lambda d: enrich_with_ndvi_ee(d, checkpoint=save),
        df,
        checkpoint_path,
    )

    if store_mode:
        group_by = os.getenv('GROUP_BY', 'genus')
        key_column = group_by if group_by in df.columns else 'species'
        written = store.write_split(df, base=store.ENRICHED_DIR, key=key_column, merge=False)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        open(done_marker, 'w').close()
    else:
        open(done_marker, 'w').close()
