import argparse
import concurrent.futures
import math
import os
import gzip
import shutil
import zipfile
from datetime import timedelta
from pathlib import Path

import cdsapi
import ee
import pandas as pd
import requests

# Consolidate the local compression import
try:
    from compress_rasters import convert_raster_to_cog
except ImportError:
    convert_raster_to_cog = None


def load_env_file(path=None):
    config_path = Path(path or os.getenv('ENV_FILE') or '.env')
    if not config_path.exists():
        return {}

    values = {}
    for line in config_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, value = stripped.split('=', 1)
        values[key.strip()] = value.strip().strip('"\'')
    return values


def load_env_into_os(path=None):
    for key, value in load_env_file(path).items():
        os.environ.setdefault(key, value)


load_env_into_os()

# Study-area bounding box, [North, West, South, East]. Shared by the ERA5 soil
# download and the DEM download so every layer covers the same footprint.
STUDY_AREA = [42, -106, 39, -102]  # around Colorado


# ─── Topography (Digital Elevation Model) ─────────────────────────────────────
def download_srtm_dem(area=None, output_dir="dem/", dem_type="SRTMGL3", api_key=None):
    """Download a DEM GeoTIFF for the study area from the OpenTopography API."""
    area = area or STUDY_AREA
    north, west, south, east = area
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, f"dem_{dem_type}.tif")
    cog_path = os.path.join(output_dir, f"dem_{dem_type}.cog.tif")

    # Use a cached DEM if present. Prefer the compressed COG, but also accept a
    # raw .tif left behind when a previous compression failed — otherwise the DEM
    # re-downloads on every run. Retry compression opportunistically, never
    # re-download when the data is already on disk.
    if os.path.exists(cog_path):
        print(f"✅ Already downloaded: {cog_path}")
        return cog_path
    if os.path.exists(out_path):
        print(f"✅ Already downloaded: {out_path}")
        if convert_raster_to_cog:
            try:
                return convert_raster_to_cog(out_path, output_path=cog_path, delete_original=True, verify=True)
            except Exception as exc:
                print(f"[!] DEM compression retry failed; using uncompressed file: {exc}")
        return out_path

    api_key = api_key or os.environ.get("OPENTOPOGRAPHY_API_KEY")
    if not api_key:
        print(
            "⚠️  No OpenTopography API key found. Set OPENTOPOGRAPHY_API_KEY "
            "(free at https://portal.opentopography.org/login) to download the DEM."
        )
        return None

    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }

    try:
        print(f"🔽 Downloading {dem_type} DEM for {area}...")
        r = requests.get(url, params=params, stream=True, timeout=120)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        if convert_raster_to_cog:
            try:
                # Explicitly pass the correctly formatted cog_path
                converted = convert_raster_to_cog(out_path, output_path=cog_path, delete_original=True, verify=True)
                print(f"✅ DEM saved to {converted}")
                return converted
            except Exception as exc:
                print(f"[!] DEM compression failed; keeping uncompressed file: {exc}")
                
        return out_path
    except Exception as e:
        print(f"[!] Error fetching DEM: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return None

# ─── Soil Moisture (ERA5-Land) Worker ─────────────────────────────────────────
def download_era5_worker(date_str, output_dir="soil/"):
    """Thread worker for fetching ERA5-Land soil moisture via CDS API."""
    os.makedirs(output_dir, exist_ok=True)
    year, month, day = date_str.split("-")

    zip_path = os.path.join(output_dir, f"soil_{date_str}.zip")
    nc_path = os.path.join(output_dir, f"soil_{date_str}.nc")

    if os.path.exists(nc_path):
        return "cached", date_str, nc_path

    # Initialize a local client for the thread. quiet=True prevents intertwined console spam.
    c = cdsapi.Client(quiet=True)

    dataset = "reanalysis-era5-land"
    request = {
        "variable": ["volumetric_soil_water_layer_1"],
        "year": year,
        "month": month,
        "day": [day],
        "time": [f"{h:02d}:00" for h in range(24)],  # All 24 hours
        "data_format": "netcdf",
        "area": STUDY_AREA,
    }
    
    try:
        c.retrieve(dataset, request, zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files = zip_ref.namelist()
            extracted_nc = [f for f in extracted_files if f.endswith(".nc")]
            if extracted_nc:
                os.rename(os.path.join(output_dir, extracted_nc[0]), nc_path)
        os.remove(zip_path)

        return "downloaded", date_str, nc_path
    except Exception as e:
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return "error", date_str, str(e)


def init_earth_engine():
    """Initialize Earth Engine and auto-authenticate when needed."""
    project = os.environ.get("EARTHENGINE_PROJECT")
    try:
        ee.Initialize(project=project)
        return True
    except Exception as exc:
        print("Earth Engine is not authenticated yet. Starting the authentication flow...")
        try:
            ee.Authenticate(quiet=True)
            ee.Initialize(project=project)
            print("Earth Engine authenticated and initialized successfully.")
            return True
        except Exception as auth_exc:
            print("[!] Earth Engine auth failed or was blocked by Google. Skipping EE stages.")
            print("    Reason:", auth_exc)
            return False


def fetch_sentinel2_ndvi(lat, lon, date_str, output_dir="ndvi/"):
    date = pd.to_datetime(date_str)
    range_val = 5
    start_date = (date - timedelta(days=range_val)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=range_val)).strftime('%Y-%m-%d')

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(500).bounds()

    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(start_date, end_date)
                  .filterBounds(region)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
                  .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI')))

    ndvi_image = collection.median().clip(region)

    task = ee.batch.Export.image.toDrive(
        image=ndvi_image,
        description=f"NDVI_{date_str}_{lat:.4f}_{lon:.4f}",
        folder='EarthEngineNDVI',
        fileNamePrefix=f"ndvi_{date_str}_{lat:.4f}_{lon:.4f}",
        scale=10,
        region=region.getInfo()['coordinates'],
        crs="EPSG:4326",
        maxPixels=1e9
    )

    task.start()
    print(f"📦 Started NDVI export task for {date_str} at ({lat},{lon})")


def _study_region(area=None):
    north, west, south, east = area or STUDY_AREA
    return ee.Geometry.Rectangle([west, south, east, north])


def fetch_sentinel1_moisture(area=None, start_date="2024-04-01", end_date="2024-06-30",
                             scale=90, folder="EarthEngineMoisture"):
    region = _study_region(area)

    collection = (ee.ImageCollection('COPERNICUS/S1_GRD')
                  .filterDate(start_date, end_date)
                  .filterBounds(region)
                  .filter(ee.Filter.eq('instrumentMode', 'IW'))
                  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                  .select('VV'))

    vv_image = collection.median().clip(region)

    task = ee.batch.Export.image.toDrive(
        image=vv_image,
        description=f"S1_VV_{start_date}_{end_date}",
        folder=folder,
        fileNamePrefix=f"s1_vv_{start_date}_{end_date}",
        scale=scale,
        region=region.getInfo()['coordinates'],
        crs="EPSG:4326",
        maxPixels=1e10
    )
    task.start()
    print(f"📦 Started Sentinel-1 VV export ({start_date}→{end_date}) at {scale} m")
    return task


def fetch_sentinel2_ndmi(area=None, start_date="2024-04-01", end_date="2024-06-30",
                         scale=20, cloud_pct=40, folder="EarthEngineMoisture"):
    region = _study_region(area)

    collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterDate(start_date, end_date)
                  .filterBounds(region)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                  .map(lambda img: img.normalizedDifference(['B8', 'B11']).rename('NDMI')))

    ndmi_image = collection.median().clip(region)

    task = ee.batch.Export.image.toDrive(
        image=ndmi_image,
        description=f"S2_NDMI_{start_date}_{end_date}",
        folder=folder,
        fileNamePrefix=f"s2_ndmi_{start_date}_{end_date}",
        scale=scale,
        region=region.getInfo()['coordinates'],
        crs="EPSG:4326",
        maxPixels=1e10
    )
    task.start()
    print(f"📦 Started Sentinel-2 NDMI export ({start_date}→{end_date}) at {scale} m")
    return task


def _remove_stale_chirps_files(*paths):
    for stale_path in paths:
        if not stale_path:
            continue
        try:
            if os.path.exists(stale_path):
                os.remove(stale_path)
        except OSError:
            pass


def fetch_chirps_precip_worker(date_str, output_dir="precip/"):
    """Thread worker for fetching CHIRPS precip."""
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, f"precip_{date_str}.tif")
    cog_path = os.path.join(output_dir, f"precip_{date_str}.cog.tif")
    
    # Check for the compressed version to avoid re-downloading
    if os.path.exists(cog_path):
        return "cached", date_str, cog_path
    elif os.path.exists(out_path) and not convert_raster_to_cog:
        return "cached", date_str, out_path

    year, month, day = date_str.split("-")
    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{year}/chirps-v2.0.{year}.{month}.{day}.tif.gz"
    gz_path = out_path + ".gz"

    try:
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 404:
            _remove_stale_chirps_files(gz_path, out_path)
            return "not_found", date_str, None
        r.raise_for_status()

        _remove_stale_chirps_files(gz_path, out_path)

        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)
        
        if convert_raster_to_cog:
            try:
                # Explicitly set output_path
                out_path = convert_raster_to_cog(out_path, output_path=cog_path, delete_original=True, verify=True)
            except Exception as exc:
                return "error", date_str, f"Compression failed: {exc}"
                
        return "downloaded", date_str, out_path

    except Exception as e:
        _remove_stale_chirps_files(gz_path, out_path)
        return "error", date_str, str(e)


def get_unique_dates(df):
    return sorted(pd.to_datetime(df['date'].dropna()).dt.strftime('%Y-%m-%d').unique())


def get_precip_dates(df, buffer_days=6):
    all_dates = set()
    for d in pd.to_datetime(df['date'].dropna()):
        for i in range(buffer_days + 1):
            all_dates.add((d - timedelta(days=i)).strftime('%Y-%m-%d'))
    return sorted(all_dates)


# ─── Land Cover (ESA WorldCover) ──────────────────────────────────────────────
def _worldcover_tile_name(lat, lon, year=2020, version="v100"):
    lat_deg = math.floor(lat / 3) * 3
    lon_deg = math.floor(lon / 3) * 3
    lat_prefix = "N" if lat_deg >= 0 else "S"
    lon_prefix = "E" if lon_deg >= 0 else "W"
    return (f"ESA_WorldCover_10m_{year}_{version}_"
            f"{lat_prefix}{abs(lat_deg):02d}{lon_prefix}{abs(lon_deg):03d}_Map.tif")


def _download_worldcover_worker(tile, base_url, output_dir):
    """Thread worker for fetching WorldCover tiles."""
    out_path = os.path.join(output_dir, tile)
    
    # Remove the .tif at the end, and append .cog.tif
    cog_path = os.path.splitext(out_path)[0] + ".cog.tif"
    
    if os.path.exists(cog_path):
        return "cached", tile, cog_path
    elif os.path.exists(out_path) and not convert_raster_to_cog:
        return "cached", tile, out_path
        
    url = f"{base_url}/{tile}"
    try:
        r = requests.get(url, stream=True, timeout=120)
        if r.status_code == 404:
            return "not_found", tile, None
        r.raise_for_status()
        
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                
        if convert_raster_to_cog:
            try:
                # Explicitly set output_path
                out_path = convert_raster_to_cog(out_path, output_path=cog_path, delete_original=True, verify=True)
            except Exception as exc:
                return "error", tile, f"Compression failed: {exc}"
                
        return "downloaded", tile, out_path
    except Exception as e:
        if os.path.exists(out_path):
            os.remove(out_path)
        return "error", tile, str(e)


def download_worldcover_tiles(df, output_dir="world_cover/", year=2020, version="v100"):
    os.makedirs(output_dir, exist_ok=True)
    base_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map"

    tiles = set()
    for _, row in df.iterrows():
        if pd.isna(row.get('lat')) or pd.isna(row.get('lon')):
            continue
        tiles.add(_worldcover_tile_name(row['lat'], row['lon'], year, version))

    tiles = sorted(tiles)
    print(f"🗺  Ensuring {len(tiles)} WorldCover tile(s) using multithreading...")
    
    # ThreadPool for I/O bounds
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_tile = {executor.submit(_download_worldcover_worker, tile, base_url, output_dir): tile for tile in tiles}
        
        for future in concurrent.futures.as_completed(future_to_tile):
            status, tile, result = future.result()
            if status == "cached":
                print(f"✅ Cached: {tile}")
            elif status == "downloaded":
                print(f"✅ Downloaded & Processed: {tile} -> {result}")
            elif status == "not_found":
                print(f"⚠️  Not found (ocean or out of coverage?): {tile}")
            elif status == "error":
                print(f"[!] Error fetching {tile}: {result}")


def main(csv_path='mushroom_observations.csv'):
    skip_ee = os.environ.get("SKIP_EARTH_ENGINE") == "1"
    if not skip_ee:
        skip_ee = not init_earth_engine()
    if skip_ee:
        print("Skipping Earth Engine stages because auth is unavailable or blocked by Google.")
        print("Non-Earth-Engine data sources will continue normally.")

    df = pd.read_csv(csv_path)

    # ─── NDVI (Sentinel-2) ────────────────────────────────────────────────────
    if not skip_ee and os.environ.get("EXPORT_NDVI_TILES") == "1":
        print("Exporting Sentinel-2 NDVI tiles to Drive...")
        for idx, row in df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']) or pd.isna(row['date']):
                continue
            print(f"  → NDVI for {row['date']} at ({row['lat']}, {row['lon']})")
            fetch_sentinel2_ndvi(row['lat'], row['lon'], row['date'])

    # ─── Soil moisture (ERA5-Land) ────────────────────────────────────────────
    try:
        era5_dates = get_unique_dates(df)
        total_era = len(era5_dates)
        print(f"Fetching ERA5-Land soil moisture — {total_era} daily files to check using multithreading...")
        
        era_downloaded = era_failed = era_cached = 0
        
        # Keep max_workers low (3) for ERA5. The CDS API restricts concurrent API requests per user.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_date = {executor.submit(download_era5_worker, date_str): date_str for date_str in era5_dates}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_date), 1):
                status, date_str, result = future.result()
                prefix = f"[{i}/{total_era}]"
                
                if status == "cached":
                    era_cached += 1
                    print(f"{prefix} ✅ Cached {date_str}")
                elif status == "downloaded":
                    era_downloaded += 1
                    print(f"{prefix} ✅ Downloaded {date_str} -> {result}")
                elif status == "error":
                    era_failed += 1
                    print(f"{prefix} [!] Error for {date_str}: {result}")

        print(f"✅ ERA5 done — {era_downloaded} downloaded, {era_cached} already cached, "
              f"{era_failed} failed out of {total_era} total.")
    except Exception as e:
        print(f"[!] Soil moisture download skipped: {e}")

    # ─── Precipitation (CHIRPS) ───────────────────────────────────────────────
    try:
        precip_dates = get_precip_dates(df, buffer_days=6)
        total_precip = len(precip_dates)
        print(f"Fetching CHIRPS precipitation — {total_precip} daily tiles to check using multithreading...")
        
        precip_downloaded = precip_failed = precip_cached = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {executor.submit(fetch_chirps_precip_worker, date_str): date_str for date_str in precip_dates}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_date), 1):
                status, date_str, result = future.result()
                prefix = f"[{i}/{total_precip}]"
                
                if status == "cached":
                    precip_cached += 1
                    print(f"{prefix} ✅ Cached {date_str}")
                elif status == "downloaded":
                    precip_downloaded += 1
                    print(f"{prefix} ✅ Downloaded {date_str} -> {result}")
                elif status == "not_found":
                    precip_failed += 1
                    print(f"{prefix} ⚠️  CHIRPS not available for {date_str}")
                elif status == "error":
                    precip_failed += 1
                    print(f"{prefix} [!] Error for {date_str}: {result}")

        print(f"✅ CHIRPS done — {precip_downloaded} downloaded, {precip_cached} already cached, "
              f"{precip_failed} unavailable/failed out of {total_precip} total.")
    except Exception as e:
        print(f"[!] Precipitation download skipped: {e}")

    # ─── Land cover (ESA WorldCover) ──────────────────────────────────────────
    try:
        download_worldcover_tiles(df)
    except Exception as e:
        print(f"[!] WorldCover download skipped: {e}")

    # ─── Topography ───────────────────────────────────────────────────────────
    try:
        dem_path = download_srtm_dem()
        if dem_path:
            from terrain_pipeline import process_dem
            process_dem(dem_path)
    except Exception as e:
        print(f"[!] Terrain processing skipped: {e}")

    # ─── Independent Satellite Moisture ───────────────────────────────────────
    if os.environ.get("EXPORT_SATELLITE_MOISTURE") == "1" and not skip_ee:
        obs_dates = pd.to_datetime(df['date'].dropna())
        win_start = obs_dates.min().strftime('%Y-%m-%d')
        win_end = obs_dates.max().strftime('%Y-%m-%d')
        print(f"🛰  Exporting satellite moisture layers for {win_start}→{win_end}...")
        fetch_sentinel1_moisture(start_date=win_start, end_date=win_end)
        fetch_sentinel2_ndmi(start_date=win_start, end_date=win_end)
        print("   Exports queued to Google Drive (folder 'EarthEngineMoisture'). "
              "Download them, then run validate_wetness.py raster.")


if __name__ == "__main__":
    main()