import argparse
import concurrent.futures
import math
import os
import gzip
import shutil
import sys
import threading
import time
import zipfile
from datetime import timedelta
from pathlib import Path

import cdsapi
import ee
import pandas as pd
import requests
import xarray as xr

import species_store as store

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# When the independent download sources run concurrently, their logs interleave.
# Serialize prints and tag every line with its source so the output stays legible.
_print_lock = threading.Lock()
_progress_last = {}


def _log(tag, msg):
    with _print_lock:
        print(f"[{tag}] {msg}", flush=True)


def _fetch_progress(tag, done, total, t0, min_interval=3.0):
    """Throttled, tagged progress with an ETA — prints at most every `min_interval`
    seconds per source (and always on the final item), so concurrent stages give
    live feedback without flooding the console with one line per file."""
    now = time.monotonic()
    if done < total and now - _progress_last.get(tag, 0.0) < min_interval:
        return
    _progress_last[tag] = now
    elapsed = now - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    pct = (100 * done) // max(total, 1)
    _log(tag, f"{done}/{total} ({pct}%)  ETA {eta:5.0f}s")

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


Python
# ─── Topography (Digital Elevation Model) ─────────────────────────────────────
def download_srtm_dem(area=None, output_dir="dem/", dem_type="SRTMGL3", api_key=None):
    """Download a DEM GeoTIFF for the study area from the OpenTopography API."""
    area = area or STUDY_AREA
    north, west, south, east = area
    os.makedirs(output_dir, exist_ok=True)
    
    # Encode the bounding box coordinates into the filename (rounded to 1 decimal place)
    n_str = f"N{north:.1f}" if north >= 0 else f"S{abs(north):.1f}"
    s_str = f"N{south:.1f}" if south >= 0 else f"S{abs(south):.1f}"
    w_str = f"E{west:.1f}" if west >= 0 else f"W{abs(west):.1f}"
    e_str = f"E{east:.1f}" if east >= 0 else f"W{abs(east):.1f}"
    
    box_suffix = f"{n_str}_{s_str}_{w_str}_{e_str}"
    
    out_path = os.path.join(output_dir, f"dem_{dem_type}_{box_suffix}.tif")
    cog_path = os.path.join(output_dir, f"dem_{dem_type}_{box_suffix}.cog.tif")

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

# ─── (ERA5-Land) Worker - Soil Moisture + More ─────────────────────────────────────────
def download_era5_worker(date_str, output_dir="soil/"):
    """Thread worker for fetching ERA5-Land soil moisture via CDS API."""
    
    os.makedirs(output_dir, exist_ok=True)
    year, month, day = date_str.split("-")

    nc_path = os.path.join(output_dir, f"soil_{date_str}.nc")

    if os.path.exists(nc_path):
        return "cached", date_str, nc_path

    # Initialize a local client for the thread. quiet=True prevents intertwined console spam.
    c = cdsapi.Client(quiet=True)

    dataset = "reanalysis-era5-land"
    
    # Split variables into two groups to avoid "Structural differences" warning
    # Group 1: State variables (instantaneous values)
    state_vars = [
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_2",
        "volumetric_soil_water_layer_3",
        "volumetric_soil_water_layer_4",
        "soil_temperature_level_1",
        "soil_temperature_level_2",
        "soil_temperature_level_3",
        "soil_temperature_level_4",
        "2m_temperature",
        "relative_humidity_2m",
        "surface_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "skin_temperature",
        "snow_depth",
        "snow_water_equivalent",
        "forest_fraction",
        "crop_fraction",
        "grass_fraction",
        "shrub_fraction",
        "bare_ground_fraction",
    ]
    
    # Group 2: Flux/Accumulated variables
    flux_vars = [
        "total_precipitation",
        "surface_solar_radiation_downwards",
        "thermal_radiation_downwards",
        "evaporation",
    ]

    temp_files = []
    
    try:
        # Download Group 1
        request_state = {
            "variable": state_vars,
            "year": year,
            "month": month,
            "day": [day],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "area": STUDY_AREA,
        }
        zip_path_1 = os.path.join(output_dir, f"temp_state_{date_str}.zip")
        c.retrieve(dataset, request_state, zip_path_1)
        
        with zipfile.ZipFile(zip_path_1, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files = zip_ref.namelist()
            extracted_nc = [f for f in extracted_files if f.endswith(".nc")]
            if extracted_nc:
                src = os.path.join(output_dir, extracted_nc[0])
                dst = os.path.join(output_dir, f"temp_state_{date_str}.nc")
                os.rename(src, dst)
                temp_files.append(dst)
        os.remove(zip_path_1)

        # Download Group 2
        request_flux = {
            "variable": flux_vars,
            "year": year,
            "month": month,
            "day": [day],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "area": STUDY_AREA,
        }
        zip_path_2 = os.path.join(output_dir, f"temp_flux_{date_str}.zip")
        c.retrieve(dataset, request_flux, zip_path_2)
        
        with zipfile.ZipFile(zip_path_2, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files = zip_ref.namelist()
            extracted_nc = [f for f in extracted_files if f.endswith(".nc")]
            if extracted_nc:
                src = os.path.join(output_dir, extracted_nc[0])
                dst = os.path.join(output_dir, f"temp_flux_{date_str}.nc")
                os.rename(src, dst)
                temp_files.append(dst)
        os.remove(zip_path_2)

        # Merge the two NetCDF files
        if len(temp_files) == 2:
            ds1 = xr.open_dataset(temp_files[0])
            ds2 = xr.open_dataset(temp_files[1])
            merged = xr.merge([ds1, ds2])
            merged.to_netcdf(nc_path)
            ds1.close()
            ds2.close()
            
            # Cleanup temp files
            for f in temp_files:
                if os.path.exists(f):
                    os.remove(f)
        elif len(temp_files) == 1:
            os.rename(temp_files[0], nc_path)
        else:
            raise Exception("No valid NetCDF files extracted")

        return "downloaded", date_str, nc_path
        
    except Exception as e:
        if "not available yet" in str(e):
            print(f"[!] Soil moisture data is not yet available for {date_str}. The script will skip this date.")
        # Cleanup on error
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        zip_path_1 = os.path.join(output_dir, f"temp_state_{date_str}.zip")
        zip_path_2 = os.path.join(output_dir, f"temp_flux_{date_str}.zip")
        if os.path.exists(zip_path_1): os.remove(zip_path_1)
        if os.path.exists(zip_path_2): os.remove(zip_path_2)
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


# Backwards compatibility wrapper for single-date calls and tests
def fetch_chirps_precip(date_str, output_dir="precip/"):
    status, _, path_or_err = fetch_chirps_precip_worker(date_str, output_dir=output_dir)
    if status in ("downloaded", "cached"):
        return path_or_err
    return None


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


# ─── Independent download-source stages ───────────────────────────────────────
# Each stage below downloads from a *different* server (CDS, UCSB, ESA S3,
# OpenTopography) and writes to a disjoint output dir, so they have no data
# dependency on one another. main() runs them concurrently to overlap their
# latency (total time → max of the stages, not the sum), while each keeps its
# own internal per-source thread pool so per-server rate limits are respected.

def _stage_era5(df, tag="ERA5"):
    """ERA5-Land soil moisture via the CDS API (one netCDF per unique date)."""
    try:
        dates = get_unique_dates(df)
        total = len(dates)
        _log(tag, f"soil moisture — {total} daily file(s) to check")
        downloaded = failed = cached = 0
        t0 = time.monotonic()
        done = 0
        # Low concurrency (3): the CDS API restricts concurrent requests per user.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(download_era5_worker, d): d for d in dates}
            for future in concurrent.futures.as_completed(futures):
                status, date_str, result = future.result()
                done += 1
                if status == "cached":
                    cached += 1
                elif status == "downloaded":
                    downloaded += 1
                elif status == "error":
                    failed += 1
                    _log(tag, f"[!] {date_str}: {result}")
                _fetch_progress(tag, done, total, t0)
        _log(tag, f"✅ done — {downloaded} downloaded, {cached} cached, {failed} failed / {total}")
    except Exception as e:
        _log(tag, f"[!] skipped: {e}")


def _stage_chirps(df, tag="CHIRPS"):
    """CHIRPS daily precipitation (one GeoTIFF per date in the ±buffer window)."""
    try:
        dates = get_precip_dates(df, buffer_days=6)
        total = len(dates)
        _log(tag, f"precipitation — {total} daily tile(s) to check")
        downloaded = failed = cached = 0
        t0 = time.monotonic()
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_chirps_precip_worker, d): d for d in dates}
            for future in concurrent.futures.as_completed(futures):
                status, date_str, result = future.result()
                done += 1
                if status == "cached":
                    cached += 1
                elif status == "downloaded":
                    downloaded += 1
                elif status == "not_found":
                    failed += 1
                elif status == "error":
                    failed += 1
                    _log(tag, f"[!] {date_str}: {result}")
                _fetch_progress(tag, done, total, t0)
        _log(tag, f"✅ done — {downloaded} downloaded, {cached} cached, {failed} unavailable/failed / {total}")
    except Exception as e:
        _log(tag, f"[!] skipped: {e}")


def _stage_worldcover(df, tag="WorldCover"):
    """ESA WorldCover land-cover tiles (a handful covering the study area)."""
    try:
        download_worldcover_tiles(df)
        _log(tag, "✅ done")
    except Exception as e:
        _log(tag, f"[!] skipped: {e}")


def _stage_terrain(df, tag="Terrain"):
    """DEM download + terrain derivation (slope/aspect/exposure)."""
    try:
        dem_path = download_srtm_dem()
        if dem_path:
            _log(tag, "DEM ready — deriving terrain layers...")
            from terrain_pipeline import process_dem
            process_dem(dem_path)
        _log(tag, "✅ done")
    except Exception as e:
        _log(tag, f"[!] skipped: {e}")


def main(csv_path=None):
    skip_ee = os.environ.get("SKIP_EARTH_ENGINE") == "1"
    if not skip_ee:
        skip_ee = not init_earth_engine()
    if skip_ee:
        print("Skipping Earth Engine stages because auth is unavailable or blocked by Google.")
        print("Non-Earth-Engine data sources will continue normally.")

    # Downloads cover the union of all observation dates/locations, so load the
    # whole per-species store (or a single CSV when one is passed explicitly).
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = store.load_all(store.SPECIES_DIR)
        if df.empty:
            print(f"No observations in {store.SPECIES_DIR}/. Run iNat.py or migrate_data_layout.py first.")
            return
        print(f"Loaded {len(df)} observations from {store.SPECIES_DIR}/ for environmental downloads.")

    # ─── NDVI (Sentinel-2) — Earth Engine, main thread ────────────────────────
    # Queues async Drive export tasks; kept off the concurrent pool to avoid EE
    # client thread-safety concerns (and it's fast — just task.start() calls).
    if not skip_ee and os.environ.get("EXPORT_NDVI_TILES") == "1":
        print("Exporting Sentinel-2 NDVI tiles to Drive...")
        for idx, row in df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']) or pd.isna(row['date']):
                continue
            print(f"  → NDVI for {row['date']} at ({row['lat']}, {row['lon']})")
            fetch_sentinel2_ndvi(row['lat'], row['lon'], row['date'])

    # ─── Independent HTTP sources — run concurrently ──────────────────────────
    # These hit different servers with no cross-dependency, so overlapping them
    # collapses total wall time to the slowest single source (usually ERA5/CDS).
    stages = [
        ("ERA5", _stage_era5),
        ("CHIRPS", _stage_chirps),
        ("WorldCover", _stage_worldcover),
        ("Terrain", _stage_terrain),
    ]
    if os.environ.get("SEQUENTIAL_FETCH") == "1":
        print("Fetching data sources sequentially (SEQUENTIAL_FETCH=1)...")
        for _name, fn in stages:
            fn(df)
    else:
        print(f"Fetching {len(stages)} data sources concurrently: "
              f"{', '.join(name for name, _ in stages)}...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(stages)) as executor:
            futures = {executor.submit(fn, df): name for name, fn in stages}
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    _log(name, f"[!] failed: {e}")

    # ─── Independent Satellite Moisture — Earth Engine, main thread ────────────
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