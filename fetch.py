import ee
import pandas as pd
from datetime import timedelta
import os
import cdsapi
import zipfile
import requests

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

    c.retrieve(
        'reanalysis-era5-land',
        {
            'variable': ['volumetric_soil_water_layer_1'],
            'year': year,
            'month': month,
            'day': [day],
            'time': [f"{h:02d}:00" for h in range(24)],  # All 24 hours
            'format': 'netcdf',
            'area': [42, -106, 39, -102],  # North, West, South, East (bounding box around Colorado, you can adjust)
        },
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