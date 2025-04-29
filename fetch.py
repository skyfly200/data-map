import ee
import pandas as pd
from datetime import timedelta, datetime
import os
import cdsapi

# Initialize the CDS API client to download ERA5-Land data (Soil Moisture)
def download_era5_soil_moisture(date_str, output_dir="soil/"):
    os.makedirs(output_dir, exist_ok=True)
    year, month, day = date_str.split("-")

    c = cdsapi.Client()

    output_path = os.path.join(output_dir, f"soil_{date_str}.nc")
    if os.path.exists(output_path):
        print(f"✅ Soil moisture already downloaded: {output_path}")
        return output_path

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
        output_path
    )

    print(f"✅ Saved: {output_path}")
    return output_path

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