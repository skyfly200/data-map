import pandas as pd
from datetime import timedelta
import rasterio
from rasterio.warp import transform
import xarray as xr
import numpy as np
import os
import shutil

# ─── Raster Data Sources ────────────────────────────────────────────────────────
# https://worldcover2020.esa.int/downloader
# https://viewer.terrascope.be/?language=en&bbox=-105.28106099415182,40.204737550661235,-105.23939445065368,40.219963411726525&overlay=true&bgLayer=MapBox_Satellite&date=2025-04-23&layer=WORLDCOVER_2021_MAP
# https://wiki.openstreetmap.org/wiki/Overpass_API
# 
# https://msc.fema.gov/portal/search?AddressQuery=-105.0%2C%2040.0

# ─── Utility Function to Sample Raster Value ──────────────────────────────────

def sample_raster_value(tif_path, lon, lat):
    """
    Samples a raster file at the given longitude and latitude.
    """
    try:
        with rasterio.open(tif_path) as src:
            x, y = transform('EPSG:4326', src.crs, [lon], [lat])
            row, col = src.index(x[0], y[0])
            value = src.read(1)[row, col]
            if value != src.nodata:
                return value
    except Exception as e:
        print(f"[!] Error sampling raster at ({lon}, {lat}) in {tif_path}: {e}")
    return None

# ─── Precipitation Utilities ──────────────────────────────────────────────────

def enrich_with_precip(df, precip_dir="precip/"):
    print("Adding 7-day precipitation history...")
    for d in range(7):
        col_name = f'prcp_d{d}'
        df[col_name] = None

    unique_dates = pd.to_datetime(df['date'].dropna().unique())

    for date in unique_dates:
        for d in range(7):
            target_date = (date - timedelta(days=d)).strftime('%Y-%m-%d')
            tif_path = os.path.join(precip_dir, f"precip_sample.tif")
            # tif_path = os.path.join(precip_dir, f"precip_{target_date}.tif")

            if not os.path.exists(tif_path):
                print(f"[!] Precip raster missing: {tif_path}")
                continue

            # print(f"  ✓ Using {tif_path} for {d}-day offset")
            for idx, row in df[df['date'] == date.strftime('%Y-%m-%d')].iterrows():
                val = sample_raster_value(tif_path, row.lon, row.lat)
                df.at[idx, f'prcp_d{d}'] = val

    return df

# ─── Tree Cover Utilities ──────────────────────────────────────────────────

def enrich_with_tree_cover(df, tree_path="treecover/tree_cover.tif"):
    print("Adding tree cover...")
    if not os.path.exists(tree_path):
        print(f"[!] Tree cover raster not found: {tree_path}")
        return df

    df['tree_cover'] = df.apply(lambda row: sample_raster_value(tree_path, row.lon, row.lat), axis=1)
    return df

# ─── Soil Moisture Utilities ──────────────────────────────────────────────────

def load_soil_moisture_dataset(nc_path):
    return xr.open_dataset(nc_path)

def extract_soil_moisture(ds, lat, lon, date_str):
    try:
        date = np.datetime64(date_str)
        ds_time = ds.sel(time=date, method="nearest")
        value = ds_time['swvl1'].interp(latitude=lat, longitude=lon).values.item()
        return value
    except Exception as e:
        print(f"[!] Soil moisture missing for ({lat}, {lon}) on {date_str}: {e}")
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

def download_placeholder_ndvi(date_str, ndvi_path):
    # Placeholder file with mock NDVI values from another day or dummy image
    placeholder = "sample_ndvi.tif"
    if not os.path.exists(placeholder):
        print(f"[!] No sample NDVI to copy for {date_str}. Skipping download.")
        return
    shutil.copyfile(placeholder, ndvi_path)
    print(f"  ↓ Downloaded NDVI placeholder for {date_str} → {ndvi_path}")

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
        # ndvi_path = os.path.join(ndvi_dir, f"ndvi_{date_str}.tif")
        ndvi_path = os.path.join(ndvi_dir, f"sample_ndvi.tif")
        soil_path = os.path.join(soil_dir, f"soil_{date_str}.nc")

        # Check existence
        has_ndvi = os.path.exists(ndvi_path)
        has_soil = os.path.exists(soil_path)

        if not has_ndvi and not has_soil:
            print(f"[!] Skipping {date_str}: no NDVI or soil file found")
            continue
            
        if not has_ndvi:
            print(f"  ↓ NDVI missing for {date_str}")
            download_placeholder_ndvi(date_str, ndvi_path)
            has_ndvi = os.path.exists(ndvi_path)


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
    
    print("Starting raster-based enrichment...")
    df = enrich_df_with_rasters(df, ndvi_dir="ndvi/", soil_dir="soil/")
    df = enrich_with_precip(df, precip_dir="precip/")
    df = enrich_with_tree_cover(df, tree_path="treecover/tree_cover.tif")


    print(f"Saving enriched data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done ✅")
