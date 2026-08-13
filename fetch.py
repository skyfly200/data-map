import ee
import pandas as pd
from datetime import timedelta
import os
import cdsapi
import zipfile
import requests

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

# Initialize Earth Engine
ee.Initialize()

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
            return None
        r.raise_for_status()

        with open(gz_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        # Unzip the file
        import gzip, shutil
        with gzip.open(gz_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(gz_path)
        print(f"✅ CHIRPS saved to {out_path}")
        return out_path

    except Exception as e:
        print(f"[!] Error fetching CHIRPS for {date_str}: {e}")
        return None

def get_unique_dates(df):
    return sorted(pd.to_datetime(df['date'].dropna()).dt.strftime('%Y-%m-%d').unique())


df = pd.read_csv('mushroom_observations.csv')

# for idx, row in df.iterrows():
#     if pd.isna(row['lat']) or pd.isna(row['lon']) or pd.isna(row['date']):
#         continue
#     print(f"Fetching NDVI for {row['date']} at ({row['lat']}, {row['lon']})")
#     fetch_sentinel2_ndvi(row['lat'], row['lon'], row['date'])

needed_dates = get_unique_dates(df)

for date_str in needed_dates:
    download_era5_soil_moisture(date_str)

# Topography is static — fetch the DEM once, then derive the terrain layers
# (slope, aspect, solar/wind exposure, water retention) from it.
dem_path = download_srtm_dem()
if dem_path:
    try:
        from terrain_pipeline import process_dem
        process_dem(dem_path)
    except Exception as e:
        print(f"[!] Terrain processing skipped: {e}")

# Independent satellite moisture layer for validating the wetness index.
# Opt-in (these are large Earth Engine exports to Drive): set
# EXPORT_SATELLITE_MOISTURE=1. Download the resulting GeoTIFF from Drive, then:
#   python validate_wetness.py raster --satellite s1_vv_<window>.tif
if os.environ.get("EXPORT_SATELLITE_MOISTURE") == "1":
    # Match the export window to the span of observed dates.
    obs_dates = pd.to_datetime(df['date'].dropna())
    win_start = obs_dates.min().strftime('%Y-%m-%d')
    win_end = obs_dates.max().strftime('%Y-%m-%d')
    print(f"🛰  Exporting satellite moisture layers for {win_start}→{win_end}...")
    fetch_sentinel1_moisture(start_date=win_start, end_date=win_end)
    fetch_sentinel2_ndmi(start_date=win_start, end_date=win_end)
    print("   Exports queued to Google Drive (folder 'EarthEngineMoisture'). "
          "Download them, then run validate_wetness.py raster.")