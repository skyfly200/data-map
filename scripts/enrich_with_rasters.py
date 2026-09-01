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
import ee_enrich

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


def sample_raster_points(tif_path, points, scale_factor=1.0, nodata_val=None):
    """Sample many (lon, lat) points from one raster with a SINGLE open+read.

    The per-point ``sample_raster_value`` re-opens the file and reads the whole
    band for every point — catastrophic when sampling hundreds of points from
    the same raster. This opens once, reads the band once, and indexes every
    point, which is orders of magnitude faster for the enrichment loops.

    Returns a list of values aligned with ``points`` (None where out of bounds,
    nodata, or NaN).
    """
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
            X, Y = transform('EPSG:4326', src.crs, xs, ys)  # batch reproject
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
        print(f"[!] Error batch-sampling {tif_path}: {e}")
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

    # Resume: skip dates whose rows already have precip filled.
    all_dates = sorted(pd.to_datetime(df['date'].dropna().unique()))
    pending = [d for d in all_dates
               if df.loc[df['date'] == d.strftime('%Y-%m-%d'), 'prcp_d0'].isna().any()]
    total = len(pending)
    if not total:
        print("Precipitation history already complete — skipping.")
        return df
    total_rows = int(sum(len(df[df['date'] == d.strftime('%Y-%m-%d')]) for d in pending))
    print(f"Adding 7-day precipitation history — {total} dates, {total_rows} observations...")

    start = time.monotonic()
    missing = set()
    done_rows = 0
    for i, date in enumerate(pending, 1):
        dstr = date.strftime('%Y-%m-%d')
        # Only the rows still missing rainfall: this stage runs after the Earth
        # Engine sampler, so filled rows must not be rewritten from local tiles.
        day_rows = df[(df['date'] == dstr) & df['prcp_d0'].isna()]
        if day_rows.empty:
            continue
        idxs = list(day_rows.index)
        coords = list(zip(day_rows['lon'], day_rows['lat']))
        for d in range(7):
            target_date = (date - timedelta(days=d)).strftime('%Y-%m-%d')
            tif_path = resolve_raster_path(os.path.join(precip_dir, f"precip_{target_date}.tif"))
            if not tif_path:
                missing.add(target_date)
                continue
            # One open+read for the whole date group instead of one per point.
            vals = sample_raster_points(tif_path, coords)
            for idx, v in zip(idxs, vals):
                df.at[idx, f'prcp_d{d}'] = v

        done_rows += len(day_rows)
        pct = i / total * 100
        elapsed = time.monotonic() - start
        eta = (elapsed / i) * (total - i)
        print(f"  [{i}/{total}] {pct:5.1f}%  {dstr} ({len(day_rows)} obs)  "
              f"· {done_rows}/{total_rows} obs · {elapsed:4.0f}s elapsed, ~{eta:4.0f}s left",
              flush=True)
        if checkpoint and i % 50 == 0:
            checkpoint()

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

def enrich_with_worldcover(df, base_dir="./world_cover/", checkpoint=None):
    if 'land_cover' not in df.columns:
        df['land_cover'] = None
    # Resume: only rows without a land-cover value yet.
    pending = [(idx, row) for idx, row in df.iterrows() if pd.isna(row.get('land_cover'))]
    total = len(pending)
    if not total:
        print("WorldCover already complete — skipping.")
        return df
    print(f"Adding WorldCover land class — {total} observations...")

    start = time.monotonic()
    # Group points by their 3°×3° tile, then sample each tile once for all of them.
    by_tile = {}
    for idx, row in pending:
        by_tile.setdefault(get_worldcover_tile_name(row.lat, row.lon), []).append((idx, row.lon, row.lat))

    missing = set()
    done = 0
    for t, (tile_name, group) in enumerate(by_tile.items(), 1):
        tile_path = resolve_raster_path(os.path.join(base_dir, tile_name))
        if not tile_path:
            missing.add(tile_name)
            print(f"[!] WorldCover tile not cached: {tile_name} (run fetch.py to download it)")
        else:
            vals = sample_raster_points(tile_path, [(lon, lat) for _, lon, lat in group], scale_factor=1, nodata_val=255)
            for (idx, _, _), v in zip(group, vals):
                df.at[idx, 'land_cover'] = v
        done += len(group)
        _progress("land cover", done, total, start)
        if checkpoint:
            checkpoint()

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


def enrich_with_terrain(df, terrain_dir="dem/derived/", checkpoint=None):
    """Sample the DEM-derived terrain layers at each observation point.

    Adds columns: slope, aspect, solar_exposure, wind_exposure, water_retention.
    Run ``fetch.py`` (or ``terrain_pipeline.py``) first to generate the layers.
    """
    for name in TERRAIN_LAYERS:
        if name not in df.columns:
            df[name] = None

    # Resume first: when Earth Engine (or an earlier run) already filled these
    # columns there is nothing to do, and no reason to warn about rasters that
    # were deliberately never downloaded.
    gate = TERRAIN_LAYERS[0]
    pending = [(idx, row) for idx, row in df.iterrows()
               if not (pd.isna(row.get("lat")) or pd.isna(row.get("lon"))) and pd.isna(row.get(gate))]
    total = len(pending)
    if not total:
        print("Terrain exposure already complete — skipping.")
        return df

    layer_paths = {}
    missing = []
    for name in TERRAIN_LAYERS:
        # Search for the layer with or without the coordinate suffix
        search_pattern = os.path.join(terrain_dir, f"{name}*.tif")
        matches = glob.glob(search_pattern)

        if matches:
            # resolve_raster_path ensures it finds the .cog.tif if compressed
            layer_paths[name] = resolve_raster_path(matches[0])
        else:
            missing.append(name)

    if not layer_paths:
        print(f"[!] No terrain layers in {terrain_dir} — skipping terrain enrichment.")
        print("    Generate them with `python terrain_pipeline.py` (it needs a DEM in dem/; "
              "set OPENTOPOGRAPHY_API_KEY to auto-download one), then re-run enrichment.")
        return df
    if missing:
        print(f"[!] Terrain layers missing: {', '.join(missing)} (re-run terrain_pipeline.py).")
    print(f"Adding terrain exposure ({len(layer_paths)} layers × {total} observations)...")
    start = time.monotonic()
    idxs = [idx for idx, _ in pending]
    coords = [(row.lon, row.lat) for _, row in pending]
    # Each static layer is opened+read once for ALL points, not once per point.
    for li, (name, path) in enumerate(layer_paths.items(), 1):
        vals = sample_raster_points(path, coords)
        for idx, v in zip(idxs, vals):
            df.at[idx, name] = v
        _progress(f"terrain ({name})", li, len(layer_paths), start)
        if checkpoint:
            checkpoint()

    return df

# ─── Temperature History Utilities ────────────────────────────────────────────
# Daily high/low for the days leading up to each observation, from the keyless
# Open-Meteo historical archive. Pairs with the CHIRPS precip history
# (prcp_d0..d6) to chart the weather run-up to each find. d0 = observation day,
# d6 = six days before.

def enrich_with_temperature_history(df, days=7, max_workers=10, checkpoint=None):
    import requests

    for d in range(days):
        for col in (f'tmax_d{d}', f'tmin_d{d}'):
            if col not in df.columns:
                df[col] = None

    url = "https://archive-api.open-meteo.com/v1/archive"
    # Resume: only rows still missing their temperature history.
    tasks = [
        (idx, float(row['lat']), float(row['lon']), pd.to_datetime(row['date']))
        for idx, row in df.iterrows()
        if not (pd.isna(row.get('lat')) or pd.isna(row.get('lon')) or pd.isna(row.get('date')))
        and pd.isna(row.get('tmax_d0'))
    ]
    total = len(tasks)
    if not total:
        print("Temperature history already complete — skipping.")
        return df
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
            if checkpoint and i % 500 == 0:
                checkpoint()

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

# ─── NDVI via Earth Engine ────────────────────────────────────────────────────
# NDVI is sampled directly at each point by ee_enrich.enrich_ndvi_ee, alongside
# every other Earth-Engine-backed column. `fill_missing_ndvi` below still carries
# a value across a short gap for points EE could not resolve (persistent cloud).

# ─── Main Enrichment Script ───────────────────────────────────────────────────

def enrich_df_with_rasters(df, ndvi_dir='ndvi/', soil_dir='soil/'):
    # Don't wipe values from a resumed checkpoint — only initialise if absent.
    for col in ('ndvi', 'soil_moisture'):
        if col not in df.columns:
            df[col] = None

    # Resume: only dates that still have a gap. This stage runs after the Earth
    # Engine samplers, so without the guard a stale cached raster would overwrite
    # a value EE just supplied.
    unique_dates = [
        d for d in sorted(set(df['date'].dropna()))
        if df.loc[df['date'] == d, ['ndvi', 'soil_moisture']].isna().any().any()
    ]
    total = len(unique_dates)
    if not total:
        print("NDVI + soil moisture already complete — skipping.")
        return df
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
            # Fill gaps only — never replace a value another stage already found.
            if has_ndvi and pd.isna(df.at[idx, 'ndvi']):
                df.at[idx, 'ndvi'] = get_ndvi_from_raster(ndvi_path, row.lon, row.lat)
            if has_soil and pd.isna(df.at[idx, 'soil_moisture']):
                df.at[idx, 'soil_moisture'] = extract_soil_moisture(soil_ds, row.lat, row.lon, row.date)

    print(f"✅ NDVI/soil pass done — {n_ndvi} date(s) with NDVI, {n_soil} with soil, "
          f"{n_skip} with neither ({total} total).")
    return df

# ─── Script Entrypoint ────────────────────────────────────────────────────────

def _checkpoint(df, path):
    """Atomically write the enriched CSV so far, so a partial run still produces
    usable output (write to a temp file, then replace — never a half-written CSV)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _merge_enriched(base, prior):
    """Carry enriched columns from a prior (partial) run onto the current raw
    rows, matched by uuid. Rows without a prior value stay unenriched so the
    stages fill them, and newly-added species/observations are picked up too."""
    if prior is None or prior.empty or 'uuid' not in base.columns or 'uuid' not in prior.columns:
        return base
    prior = prior.drop_duplicates(subset='uuid', keep='last').set_index('uuid')
    df = base.copy()
    for col in prior.columns:  # enriched-only columns; raw columns already correct in base
        if col not in df.columns:
            df[col] = df['uuid'].map(prior[col])
    return df


def run_stage(label, fn, df, path):
    """Run one enrichment stage, then checkpoint. A stage failure is logged and
    skipped (keeping every earlier stage's data) rather than aborting the run;
    an interrupt saves progress before exiting. This makes enrichment
    progressive — the output CSV always reflects the last completed stage."""
    print(f"\n=== {label} ===")
    try:
        df = fn(df)
    except KeyboardInterrupt:
        print(f"[!] Interrupted during '{label}' — saving progress and exiting.")
        _checkpoint(df, path)
        raise
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
    parser = argparse.ArgumentParser(description="Enrich observation rows with raster and terrain data")
    parser.add_argument("--input", default=None,
                        help="Single input CSV (default: the per-species store data/species/)")
    parser.add_argument("--output", default=None,
                        help="Single output CSV (default: the per-species enriched store data/enriched/)")
    args = parser.parse_args()

    store_mode = args.input is None

    # Resume in both modes: an interrupted run leaves a checkpoint, and each stage
    # only re-does rows still missing its values, so a stop/start, crash, or hang
    # picks up where it left off. A stale completion marker is cleared while we work.
    if store_mode:
        base = store.load_all(store.SPECIES_DIR)
        if base.empty:
            raise SystemExit(f"No CSVs in {store.SPECIES_DIR}/. "
                             "Run iNat.py or migrate_data_layout.py first.")
        os.makedirs(store.ENRICHED_DIR, exist_ok=True)
        checkpoint_file = os.path.join(store.ENRICHED_DIR, '_checkpoint.csv')
        done_marker = store.ENRICHED_DONE
        print(f"Loaded {len(base)} rows from {store.SPECIES_DIR}/ ({base['species'].nunique()} species).")

        # Prefer a combined checkpoint from an interrupted run; otherwise fold in
        # the per-species enriched files already on disk (matched by uuid).
        prior = None
        if not os.path.exists(done_marker):
            if os.path.exists(checkpoint_file):
                try:
                    prior = pd.read_csv(checkpoint_file)
                    print(f"↻ Resuming from {checkpoint_file} ({len(prior)} rows).")
                except Exception as exc:
                    print(f"[!] Couldn't read checkpoint ({exc}); ignoring.")
            if prior is None:
                existing = store.load_all(store.ENRICHED_DIR)
                if not existing.empty:
                    prior = existing
                    print(f"↻ Resuming from {store.ENRICHED_DIR}/ ({len(prior)} rows).")
        df = _merge_enriched(base, prior)
        if os.path.exists(done_marker):
            os.remove(done_marker)

        def save():
            _checkpoint(df, checkpoint_file)  # single fast combined write during the run
    else:
        input_file = args.input
        output_file = args.output or stage_output_path(input_file, '_enriched')
        done_marker = f"{output_file}.done"
        base = pd.read_csv(input_file)
        if os.path.exists(output_file) and not os.path.exists(done_marker):
            try:
                resumed = pd.read_csv(output_file)
                if len(resumed) == len(base):
                    df = resumed
                    print(f"↻ Resuming from checkpoint {output_file} ({len(df)} rows).")
                else:
                    print(f"[!] Checkpoint row count ({len(resumed)}) != input ({len(base)}); starting fresh.")
                    df = base
            except Exception as exc:
                print(f"[!] Couldn't read checkpoint ({exc}); starting fresh.")
                df = base
        else:
            print(f"Loading {input_file}...")
            df = base
        if os.path.exists(done_marker):
            os.remove(done_marker)

        def save():
            _checkpoint(df, output_file)

    checkpoint_path = checkpoint_file if store_mode else output_file

    # ─── Earth Engine first ───────────────────────────────────────────────────
    # Every environmental column is available from Earth Engine as a point
    # sample, which is far cheaper than downloading the source rasters. These
    # stages fill what they can; the raster stages below then run as a fallback,
    # and since each of those only touches rows still missing its column, they
    # cost nothing when EE has already supplied the data.
    if ee_enrich.earth_engine_enabled():
        for label, fn in ee_enrich.EE_STAGES:
            df = run_stage(label, lambda d, _fn=fn: _fn(d, checkpoint=save), df, checkpoint_path)
    else:
        print("\nEarth Engine sampling disabled — using cached rasters only.")

    # ─── Cached-raster fallback ───────────────────────────────────────────────
    print("Total dates needed (for precip):", len(get_needed_raster_dates(df)))
    df = run_stage("NDVI + soil (cached rasters)", lambda d: enrich_df_with_rasters(d, ndvi_dir="ndvi/", soil_dir="soil/"), df, checkpoint_path)
    df = run_stage("Precipitation history", lambda d: enrich_with_precip(d, precip_dir="precip/", checkpoint=save), df, checkpoint_path)
    df = run_stage("Land cover", lambda d: enrich_with_worldcover(d, checkpoint=save), df, checkpoint_path)
    df = run_stage("Land-cover labels / filter", _postprocess_landcover, df, checkpoint_path)
    df = run_stage("Terrain exposure", lambda d: enrich_with_terrain(d, checkpoint=save), df, checkpoint_path)
    df = run_stage("Temperature history", lambda d: enrich_with_temperature_history(d, checkpoint=save), df, checkpoint_path)
    df = run_stage("Fill missing NDVI", lambda d: fill_missing_ndvi(d, max_days_gap=7), df, checkpoint_path)

    if store_mode:
        # Split the completed frame into per-species enriched files, drop the
        # combined checkpoint, and mark the run complete.
        group_by = os.getenv('GROUP_BY', 'genus')
        key_column = group_by if group_by in df.columns else 'species'
        written = store.write_split(df, base=store.ENRICHED_DIR, key=key_column, merge=False)
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        open(done_marker, 'w').close()
        print(f"\nDone ✅  Enriched {len(df)} rows → {store.ENRICHED_DIR}/ "
              f"({len(written)} {key_column} files).")
    else:
        open(done_marker, 'w').close()
        print(f"\nDone ✅  Enriched data saved to {output_file}")
