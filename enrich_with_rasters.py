import pandas as pd
from datetime import timedelta
import rasterio
from rasterio.warp import transform
import xarray as xr
import numpy as np
import math
import os
import shutil

# ─── Raster Data Sources ────────────────────────────────────────────────────────
# https://worldcover2020.esa.int/downloader
# https://viewer.terrascope.be/?language=en&bbox=-105.28106099415182,40.204737550661235,-105.23939445065368,40.219963411726525&overlay=true&bgLayer=MapBox_Satellite&date=2025-04-23&layer=WORLDCOVER_2021_MAP
# https://wiki.openstreetmap.org/wiki/Overpass_API
# 
# https://msc.fema.gov/portal/search?AddressQuery=-105.0%2C%2040.0

# ─── Utility Function to Sample Raster Value ──────────────────────────────────

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
    print("Adding 7-day precipitation history...")
    for d in range(7):
        col_name = f'prcp_d{d}'
        df[col_name] = None

    unique_dates = pd.to_datetime(df['date'].dropna().unique())

    for date in unique_dates:
        print(f"  → Enriching precip data for {date.strftime('%Y-%m-%d')} ({len(df[df['date'] == date.strftime('%Y-%m-%d')])} rows)")
        for d in range(7):
            target_date = (date - timedelta(days=d)).strftime('%Y-%m-%d')
            # tif_path = os.path.join(precip_dir, f"precip_sample.tif")
            tif_path = os.path.join(precip_dir, f"precip_{target_date}.tif")

            if not os.path.exists(tif_path):
                print(f"[!] Precip raster missing: {tif_path}")
                continue

            # print(f"  ✓ Using {tif_path} for {d}-day offset")
            for idx, row in df[df['date'] == date.strftime('%Y-%m-%d')].iterrows():
                val = sample_raster_value(tif_path, row.lon, row.lat)
                df.at[idx, f'prcp_d{d}'] = val

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
    print("Adding WorldCover land class...")
    df['land_cover'] = None

    for idx, row in df.iterrows():
        tile_name = get_worldcover_tile_name(row.lat, row.lon)
        tile_path = os.path.join(base_dir, tile_name)

        if not os.path.exists(tile_path):
            print(f"[!] Tile not found: {tile_path}")
            continue

        print(f"  ✓ Using {tile_path} for ({row.lat}, {row.lon})")
        val = sample_raster_value(tile_path, row.lon, row.lat, scale_factor=1, nodata_val=255)
        df.at[idx, 'land_cover'] = val

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

def add_worldcover_labels(df):
    df['land_cover_label'] = df['land_cover'].map(ESA_WORLDCOVER_CLASSES)
    return df

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
    for name in TERRAIN_LAYERS:
        df[name] = None
        path = os.path.join(terrain_dir, f"{name}.tif")
        if os.path.exists(path):
            layer_paths[name] = path
        else:
            print(f"[!] Terrain layer missing: {path}")

    if not layer_paths:
        print("[!] No terrain layers found — skipping terrain enrichment.")
        return df

    for idx, row in df.iterrows():
        if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
            continue
        for name, path in layer_paths.items():
            df.at[idx, name] = sample_raster_value(path, row.lon, row.lat)

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

# ─── Main Enrichment Script ───────────────────────────────────────────────────

def enrich_df_with_rasters(df, ndvi_dir='ndvi/', soil_dir='soil/'):
    df['ndvi'] = None
    df['soil_moisture'] = None

    unique_dates = sorted(set(df['date'].dropna()))
    print(f"Processing {len(unique_dates)} unique dates...")

    for date_str in unique_dates:
        date_df = df[df['date'] == date_str]
        print(f"→ Enriching data for {date_str} ({len(date_df)} rows)")

        # Construct expected filenames
        lat = date_df['lat'].iloc[0]
        lon = date_df['lon'].iloc[0]
        ndvi_path = os.path.join(ndvi_dir, f"ndvi_{date_str}_{lat:.4f}_{lon:.4f}.tif")
        soil_path = os.path.join(soil_dir, f"soil_{date_str}.nc")

        # Check existence
        has_ndvi = os.path.exists(ndvi_path)
        has_soil = os.path.exists(soil_path)

        if not has_ndvi and not has_soil:
            print(f"[!] Skipping {date_str}: no NDVI or soil file found")
            continue

        if has_ndvi:
            print(f"  ✓ NDVI file found: {ndvi_path}")
        if has_soil:
            print(f"  ✓ Soil file found: {soil_path}")
            soil_ds = load_soil_moisture_dataset(soil_path)

        for idx, row in df[df['date'] == date_str].iterrows():
            if has_ndvi:
                df.at[idx, 'ndvi'] = get_ndvi_from_raster(ndvi_path, row.lon, row.lat)
            if has_soil:
                df.at[idx, 'soil_moisture'] = extract_soil_moisture(soil_ds, row.lat, row.lon, row.date)

    return df

# ─── Script Entrypoint ────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_file = "mushroom_observations.csv"
    output_file = "mushroom_observations_enriched.csv"

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
    df = enrich_with_terrain(df)

    # 🧠 Fill missing NDVI using same-location fallback
    print("Filling missing NDVI...")
    df = fill_missing_ndvi(df, max_days_gap=7)

    print(f"Saving enriched data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done ✅")
