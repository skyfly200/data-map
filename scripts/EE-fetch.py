"""Standalone script to fetch Sentinel-2 or Landsat NDVI and NDMI GeoTIFFs from Google Earth Engine

Automatically selects the correct satellite collection based on the observation date:
- June 2015 onwards: Sentinel-2 (Harmonized)
- 2013 to May 2015: Landsat 8
- 1999 to 2012: Landsat 7
- 1984 to 1998: Landsat 5
"""

import os
import math
from datetime import datetime, timedelta
import pandas as pd
import urllib.request

try:
    import ee
except ImportError:
    raise ImportError("earthengine-api is required. Run: pip install earthengine-api")

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_CSV = "data/enriched/_checkpoint.csv"
OUTPUT_NDVI_DIR = "ndvi_export/"
OUTPUT_NDMI_DIR = "ndmi_export/"
CLOUD_PCT_MAX = 60
BUFFER_DAYS = 15
SCALE_METRES = 30  # 30m for Landsat, 10m for Sentinel-2 (30m works universally)

def initialize_ee():
    try:
        ee.Initialize(project=os.environ.get('EARTHENGINE_PROJECT'))
        print("✓ Earth Engine initialized successfully.")
    except Exception:
        print("Authenticating Earth Engine...")
        ee.Authenticate(quiet=True)
        ee.Initialize(project=os.environ.get('EARTHENGINE_PROJECT'))


def get_image_collection_and_bands(obs_date_str, start_date, end_date, roi):
    obs_date = datetime.strptime(obs_date_str, '%Y-%m-%d')
    s2_cutoff = datetime(2015, 6, 1)
    l8_cutoff = datetime(2013, 4, 1)
    l7_cutoff = datetime(1999, 4, 1)

    if obs_date >= s2_cutoff:
        # Sentinel-2 Harmonized (NIR: B8, Red: B4, SWIR1: B11)
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterDate(start_date, end_date)
               .filterBounds(roi)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CLOUD_PCT_MAX)))
        return col, 'B8', 'B4', 'B11'

    elif obs_date >= l8_cutoff:
        # Landsat 8 Collection 2 Tier 1 (NIR: SR_B5, Red: SR_B4, SWIR1: SR_B6)
        col = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
               .filterDate(start_date, end_date)
               .filterBounds(roi)
               .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_PCT_MAX)))
        return col, 'SR_B5', 'SR_B4', 'SR_B6'

    elif obs_date >= l7_cutoff:
        # Landsat 7 Collection 2 Tier 1 (NIR: SR_B4, Red: SR_B3, SWIR1: SR_B5)
        col = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
               .filterDate(start_date, end_date)
               .filterBounds(roi)
               .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_PCT_MAX)))
        return col, 'SR_B4', 'SR_B3', 'SR_B5'

    else:
        # Landsat 5 Collection 2 Tier 1 (NIR: SR_B4, Red: SR_B3, SWIR1: SR_B5)
        col = (ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
               .filterDate(start_date, end_date)
               .filterBounds(roi)
               .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_PCT_MAX)))
        return col, 'SR_B4', 'SR_B3', 'SR_B5'


def fetch_raster_indices_for_dates(csv_path):
    os.makedirs(OUTPUT_NDVI_DIR, exist_ok=True)
    os.makedirs(OUTPUT_NDMI_DIR, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"[!] Input CSV not found at {csv_path}. Please adjust path.")
        return

    df = pd.read_csv(csv_path)
    if 'date' not in df.columns or 'lat' not in df.columns or 'lon' not in df.columns:
        print("[!] CSV must contain 'date', 'lat', and 'lon' columns.")
        return

    dates = sorted(df['date'].dropna().unique())
    min_lat, max_lat = df['lat'].min(), df['lat'].max()
    min_lon, max_lon = df['lon'].min(), df['lon'].max()

    print(f"Found {len(dates)} unique dates across bounds: [{min_lat:.2f}, {min_lon:.2f} to {max_lat:.2f}, {max_lon:.2f}]")
    roi = ee.Geometry.Rectangle([min_lon - 0.1, min_lat - 0.1, max_lon + 0.1, max_lon + 0.1])

    for date_str in dates:
        # Get the exact points for this specific date to keep the bounding box small
        date_df = df[df['date'] == date_str]
        d_min_lat, d_max_lat = date_df['lat'].min(), date_df['lat'].max()
        d_min_lon, d_max_lon = date_df['lon'].min(), date_df['lon'].max()
        
        # Create a tight local ROI with a 0.05 degree buffer (~5km)
        local_roi = ee.Geometry.Rectangle([
            d_min_lon - 0.05, d_min_lat - 0.05, 
            d_max_lon + 0.05, d_max_lat + 0.05
        ])

        col, nir_band, red_band, swir_band = get_image_collection_and_bands(date_str, start_date, end_date, local_roi)

        if col.size().getInfo() == 0:
            print(f"  [!] No images found for {date_str} within cloud threshold.")
            continue

        composite = col.median().clip(local_roi)

        ndvi = composite.normalizedDifference([nir_band, red_band]).rename('NDVI')
        ndmi = composite.normalizedDifference([nir_band, swir_band]).rename('NDMI')

        ndvi_scaled = ndvi.multiply(10000).toInt16()
        ndmi_scaled = ndmi.multiply(10000).toInt16()

        for img, out_path, name in [(ndvi_scaled, ndvi_out, 'NDVI'), (ndmi_scaled, ndmi_out, 'NDMI')]:
            url = img.getDownloadURL({
                'scale': SCALE_METRES,
                'crs': 'EPSG:4326',
                'region': local_roi.getInfo()['coordinates'],
                'format': 'GEO_TIFF'
            })
            
            urllib.request.urlretrieve(url, out_path)
            print(f"  ✓ Saved {name} to {out_path}")

    print("\n✅ All available NDVI and NDMI GeoTIFFs fetched successfully!")
    print(f"   - NDVI folder: {OUTPUT_NDVI_DIR}")
    print(f"   - NDMI folder: {OUTPUT_NDMI_DIR}")


if __name__ == "__main__":
    initialize_ee()
    fetch_raster_indices_for_dates(INPUT_CSV)