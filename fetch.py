import ee
import pandas as pd
from datetime import timedelta
import math
import os
from pathlib import Path
import cdsapi
import zipfile
import requests


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
    """Download a DEM GeoTIFF for the study area from the OpenTopography API.

    Topography is static, so a single DEM covering the whole study area is
    fetched once (unlike the date-indexed weather layers). ``terrain_pipeline``
    then derives slope, aspect, solar/wind exposure and water retention from it.

    ``dem_type`` selects the global DEM: ``SRTMGL3`` (90 m, small & fast) or
    ``SRTMGL1`` (30 m, higher resolution but much larger). A free OpenTopography
    API key is required — set ``OPENTOPOGRAPHY_API_KEY`` or pass ``api_key``.
    See https://portal.opentopography.org/apidocs/ .
    """
    area = area or STUDY_AREA
    north, west, south, east = area
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"dem_{dem_type}.tif")

    if os.path.exists(out_path):
        print(f"✅ Already downloaded: {out_path}")
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
        print(f"✅ DEM saved to {out_path}")
        return out_path
    except Exception as e:
        print(f"[!] Error fetching DEM: {e}")
        if os.path.exists(out_path):
            os.remove(out_path)
        return None

# Initialize the CDS API client to download ERA5-Land data (Soil Moisture)
def download_era5_soil_moisture(date_str, output_dir="soil/"):
    os.makedirs(output_dir, exist_ok=True)
    year, month, day = date_str.split("-")

    c = cdsapi.Client()

    zip_path = os.path.join(output_dir, f"soil_{date_str}.zip")
    nc_path = os.path.join(output_dir, f"soil_{date_str}.nc")

    if os.path.exists(nc_path):
        print(f"✅ Already downloaded: {nc_path}")
        return nc_path

    print(f"🔽 Downloading ERA5-Land soil moisture for {date_str}...")

    dataset = "reanalysis-era5-land"
    request = {
        "variable": ["volumetric_soil_water_layer_1"],
        "year": year,
        "month": month,
        "day": [day],
        "time": [f"{h:02d}:00" for h in range(24)],  # All 24 hours
        "data_format": "netcdf",
        "area": STUDY_AREA,  # North, West, South, East (bounding box around Colorado, you can adjust)
    }
    
    print(request)

    c.retrieve(
        dataset,
        request,
        zip_path
    )

    # Extract .nc file from the zip
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        extracted_files = zip_ref.namelist()
        extracted_nc = [f for f in extracted_files if f.endswith(".nc")]
        if extracted_nc:
            os.rename(os.path.join(output_dir, extracted_nc[0]), nc_path)
    os.remove(zip_path)

    print(f"✅ Saved NetCDF to {nc_path}")
    return nc_path

def init_earth_engine():
    """Initialize Earth Engine and auto-authenticate when needed.

    Earth Engine can fail on a fresh machine with the "Please authorize access
    to your Earth Engine account" exception. When that happens, the SDK can be
    re-run through ee.Authenticate() once, after which ee.Initialize() succeeds
    normally.

    If Google blocks the OAuth flow for the current account, we gracefully skip
    the Earth Engine stages instead of crashing the rest of the data pipeline.
    """
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
    range = 5
    start_date = (date - timedelta(days=range)).strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=range)).strftime('%Y-%m-%d')

    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(500).bounds()  # ~1km square

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
    """ee.Geometry rectangle for the study area [North, West, South, East]."""
    north, west, south, east = area or STUDY_AREA
    return ee.Geometry.Rectangle([west, south, east, north])


def fetch_sentinel1_moisture(area=None, start_date="2024-04-01", end_date="2024-06-30",
                             scale=90, folder="EarthEngineMoisture"):
    """Export a Sentinel-1 VV backscatter composite as a soil-moisture proxy.

    SAR VV backscatter rises with surface soil moisture (strongest on bare/low
    vegetation), so a median composite over a window is an independent,
    fine-resolution wetness signal to validate the topographic wetness index
    against — see validate_wetness.py (raster mode).

    Covers the whole study area at ``scale`` metres (default 90 m to match the
    SRTMGL3 DEM). Exports to Google Drive; download the resulting GeoTIFF and
    point ``validate_wetness.py raster --satellite`` at it.
    """
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
    """Export a Sentinel-2 NDMI composite (vegetation/surface moisture proxy).

    NDMI = (B8 - B11) / (B8 + B11); higher = wetter. Optical, so it needs
    low-cloud scenes but is easy given the existing Sentinel-2 usage. Another
    independent layer for validate_wetness.py (raster mode).
    """
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


def fetch_chirps_precip(date_str, output_dir="precip/"):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"precip_{date_str}.tif")
    if os.path.exists(out_path):
        return out_path

    year, month, day = date_str.split("-")
    url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/{year}/chirps-v2.0.{year}.{month}.{day}.tif.gz"
    gz_path = out_path + ".gz"

    try:
        print(f"🔽 Downloading CHIRPS for {date_str}...")
        r = requests.get(url, stream=True, timeout=30)
        if r.status_code == 404:
            print(f"⚠️ CHIRPS not available for {date_str}. Skipping.")
            _remove_stale_chirps_files(gz_path, out_path)
            return None
        r.raise_for_status()

        _remove_stale_chirps_files(gz_path, out_path)

        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        import gzip, shutil
        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)
        print(f"✅ CHIRPS saved to {out_path}")
        return out_path

    except Exception as e:
        print(f"[!] Error fetching CHIRPS for {date_str}: {e}")
        _remove_stale_chirps_files(gz_path, out_path)
        return None

def get_unique_dates(df):
    return sorted(pd.to_datetime(df['date'].dropna()).dt.strftime('%Y-%m-%d').unique())


def get_precip_dates(df, buffer_days=6):
    """Every date needed for a ``buffer_days`` precipitation history: each
    observation date plus the preceding days (matches enrich_with_precip)."""
    all_dates = set()
    for d in pd.to_datetime(df['date'].dropna()):
        for i in range(buffer_days + 1):
            all_dates.add((d - timedelta(days=i)).strftime('%Y-%m-%d'))
    return sorted(all_dates)


# ─── Land Cover (ESA WorldCover) ──────────────────────────────────────────────
def _worldcover_tile_name(lat, lon, year=2020, version="v100"):
    """WorldCover tiles are 3°×3°, named by their SW corner (mirrors
    enrich_with_rasters.get_worldcover_tile_name)."""
    lat_deg = math.floor(lat / 3) * 3
    lon_deg = math.floor(lon / 3) * 3
    lat_prefix = "N" if lat_deg >= 0 else "S"
    lon_prefix = "E" if lon_deg >= 0 else "W"
    return (f"ESA_WorldCover_10m_{year}_{version}_"
            f"{lat_prefix}{abs(lat_deg):02d}{lon_prefix}{abs(lon_deg):03d}_Map.tif")


def download_worldcover_tiles(df, output_dir="world_cover/", year=2020, version="v100"):
    """Auto-download the ESA WorldCover tiles covering the observations.

    Tiles are served from the ESA WorldCover open bucket on AWS S3. Only the
    unique tiles spanned by the observation points are fetched, and existing
    files are skipped.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_url = f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map"

    tiles = set()
    for _, row in df.iterrows():
        if pd.isna(row.get('lat')) or pd.isna(row.get('lon')):
            continue
        tiles.add(_worldcover_tile_name(row['lat'], row['lon'], year, version))

    print(f"🗺  Ensuring {len(tiles)} WorldCover tile(s)...")
    for tile in sorted(tiles):
        out_path = os.path.join(output_dir, tile)
        if os.path.exists(out_path):
            print(f"✅ Already downloaded: {out_path}")
            continue
        url = f"{base_url}/{tile}"
        try:
            print(f"🔽 Downloading {tile}...")
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code == 404:
                print(f"⚠️ WorldCover tile not found (ocean or out of coverage?): {tile}")
                continue
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ WorldCover saved to {out_path}")
        except Exception as e:
            print(f"[!] Error fetching WorldCover tile {tile}: {e}")
            if os.path.exists(out_path):
                os.remove(out_path)


def main(csv_path='mushroom_observations.csv'):
    """Download every environmental layer for the observations, end to end.

    Importable so notebooks can call individual fetch functions without
    triggering the whole download; only running the file (or calling main())
    kicks off the pipeline.
    """
    # Earth Engine (NDVI + satellite-moisture exports) needs interactive/service
    # auth and delivers to Google Drive, so it can't run headless. Set
    # SKIP_EARTH_ENGINE=1 (e.g. in CI) to skip every EE step and still fetch the
    # DEM/terrain, precipitation, soil moisture, and land-cover layers.
    skip_ee = os.environ.get("SKIP_EARTH_ENGINE") == "1"
    if not skip_ee:
        skip_ee = not init_earth_engine()
    if skip_ee:
        print("Skipping Earth Engine stages because auth is unavailable or blocked by Google.")
        print("Non-Earth-Engine data sources will continue normally.")

    df = pd.read_csv(csv_path)

    # ─── NDVI (Sentinel-2) ────────────────────────────────────────────────────
    # One Earth Engine export task per observation, delivered to Google Drive
    # (folder 'EarthEngineNDVI'). Download those GeoTIFFs into ndvi/ to enrich.
    if skip_ee:
        print("Skipping Sentinel-2 NDVI exports (SKIP_EARTH_ENGINE=1).")
    else:
        print("Fetching Sentinel-2 NDVI exports...")
        for idx, row in df.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']) or pd.isna(row['date']):
                continue
            print(f"  → NDVI for {row['date']} at ({row['lat']}, {row['lon']})")
            fetch_sentinel2_ndvi(row['lat'], row['lon'], row['date'])

    # Each stage is isolated so one failing data source (e.g. a missing CDS key
    # in CI) doesn't abort the rest of the pipeline.

    # ─── Soil moisture (ERA5-Land) ────────────────────────────────────────────
    try:
        for date_str in get_unique_dates(df):
            download_era5_soil_moisture(date_str)
    except Exception as e:
        print(f"[!] Soil moisture download skipped: {e}")

    # ─── Precipitation (CHIRPS) ───────────────────────────────────────────────
    # Each observation date plus the 6 preceding days, for a 7-day rain history.
    print("Fetching CHIRPS precipitation...")
    try:
        for date_str in get_precip_dates(df, buffer_days=6):
            fetch_chirps_precip(date_str)
    except Exception as e:
        print(f"[!] Precipitation download skipped: {e}")

    # ─── Land cover (ESA WorldCover) ──────────────────────────────────────────
    try:
        download_worldcover_tiles(df)
    except Exception as e:
        print(f"[!] WorldCover download skipped: {e}")

    # Topography is static — fetch the DEM once, then derive the terrain layers
    # (slope, aspect, solar/wind exposure, water retention) from it.
    try:
        dem_path = download_srtm_dem()
        if dem_path:
            from terrain_pipeline import process_dem
            process_dem(dem_path)
    except Exception as e:
        print(f"[!] Terrain processing skipped: {e}")

    # Independent satellite moisture layer for validating the wetness index.
    # Opt-in (these are large Earth Engine exports to Drive): set
    # EXPORT_SATELLITE_MOISTURE=1. Download the GeoTIFF from Drive, then:
    #   python validate_wetness.py raster --satellite s1_vv_<window>.tif
    if os.environ.get("EXPORT_SATELLITE_MOISTURE") == "1" and not skip_ee:
        # Match the export window to the span of observed dates.
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