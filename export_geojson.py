"""Export the enriched/clustered observations to a static GeoJSON file.

The Nuxt frontend (deployed on Netlify) reads this file statically from
``public/data/observations.geojson`` — no backend or database required. Run
this after ``cluster.py`` (or after ``enrich_with_rasters.py`` if you haven't
clustered) and commit the result; the commit triggers a Netlify redeploy.

    python export_geojson.py
    python export_geojson.py --input mushroom_observations_enriched.csv
"""

import argparse
import json
import math
import os

import pandas as pd

# Columns surfaced to the map, in the order shown in popups. Only those present
# in the input CSV are included, so this works before or after the terrain and
# clustering stages have run.
PROPERTY_COLUMNS = [
    "species",
    "date",
    "location",
    "elevation",
    "land_cover_label",
    "ndvi",
    "soil_moisture",
    "solar_exposure",
    "wind_exposure",
    "water_retention",
    "slope",
    "aspect",
    "num_identification_agreements",
    "cluster",
    # Observation-day weather
    "tavg", "tmin", "tmax",
    # 7-day lead-up history (d0 = observation day … d6 = six days before)
    *(f"prcp_d{i}" for i in range(7)),   # rain (CHIRPS)
    *(f"tmax_d{i}" for i in range(7)),   # daily high (Open-Meteo)
    *(f"tmin_d{i}" for i in range(7)),   # daily low (Open-Meteo)
]

INT_COLUMNS = {"cluster", "num_identification_agreements"}


def _clean(value, as_int=False):
    """JSON-safe scalar: NaN/NA → None, numpy types → Python, optional int."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if pd.isna(value):
        return None
    if as_int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    return value


def _day_of_year(date_str):
    """1–366 day-of-year, for sorting/comparing dates across years (phenology)."""
    try:
        return int(pd.to_datetime(date_str).dayofyear)
    except Exception:
        return None


def to_geojson(df):
    features = []
    present = [c for c in PROPERTY_COLUMNS if c in df.columns]

    for _, row in df.iterrows():
        lon, lat = _clean(row.get("lon")), _clean(row.get("lat"))
        if lon is None or lat is None:
            continue  # can't place a point without coordinates

        props = {c: _clean(row.get(c), as_int=c in INT_COLUMNS) for c in present}
        if "date" in df.columns:
            props["day_of_year"] = _day_of_year(row.get("date"))
        if "uuid" in df.columns:
            props["uuid"] = _clean(row.get("uuid"))
        if "inat_id" in df.columns:
            props["inat_id"] = _clean(row.get("inat_id"), as_int=True)

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def _stage_output_path(input_path, suffix, output_dir='.'):
    if not input_path:
        raise ValueError('Input path is required')
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{stem}{suffix}.geojson")


def main():
    parser = argparse.ArgumentParser(description="Export observations to GeoJSON for the map")
    parser.add_argument("--input", default=None,
                        help="Input CSV (default: mushroom_clusters.csv, else "
                             "mushroom_observations_enriched.csv)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = args.input
    if input_path is None:
        for candidate in ("mushroom_clusters.csv", "mushroom_observations_enriched.csv"):
            if os.path.exists(candidate):
                input_path = candidate
                break
    if not input_path or not os.path.exists(input_path):
        raise SystemExit("No input CSV found. Run enrich_with_rasters.py / cluster.py first.")

    input_stem = os.path.splitext(os.path.basename(input_path))[0]
    output_path = args.output or os.path.join('public', 'data', f"{input_stem}.geojson")

    print(f"📂 Loading {input_path}...")
    df = pd.read_csv(input_path)

    geojson = to_geojson(df)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"✅ Wrote {len(geojson['features'])} features to {output_path}")


if __name__ == "__main__":
    main()
