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
import re

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


def _slugify(value):
    slug = re.sub(r'[^a-z0-9]+', '-', str(value).strip().lower()).strip('-')
    return slug or 'unknown'


def _write_geojson(df, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    geojson = to_geojson(df)
    with open(path, "w") as f:
        json.dump(geojson, f)
    return len(geojson["features"])


def export_all(df, data_dir=os.path.join('public', 'data'), combined_path=None):
    """Write the combined dataset, one GeoJSON per species, and a manifest.

    Layout served by the frontend:
        public/data/observations.geojson        – all species combined
        public/data/species/<slug>.geojson       – one per species
        public/data/datasets.json                – manifest the UI reads
    """
    data_dir = data_dir or os.path.join('public', 'data')
    species_dir = os.path.join(data_dir, 'species')
    os.makedirs(species_dir, exist_ok=True)

    if combined_path is None:
        combined_path = os.path.join(data_dir, 'observations.geojson')
    total = _write_geojson(df, combined_path)
    print(f"✅ Wrote {total} features to {combined_path}")

    manifest = [{
        "id": "all",
        "label": f"All species ({total})",
        "path": "/data/observations.geojson",
        "count": total,
    }]

    if "species" in df.columns:
        counts = df["species"].fillna("Unknown").value_counts()
        for species, count in counts.items():
            slug = _slugify(species)
            rel = f"/data/species/{slug}.geojson"
            n = _write_geojson(df[df["species"].fillna("Unknown") == species],
                               os.path.join(species_dir, f"{slug}.geojson"))
            manifest.append({"id": slug, "label": f"{species} ({n})", "path": rel, "count": n})
            print(f"   ✓ {species}: {n} → {rel}")

    manifest_path = os.path.join(data_dir, 'datasets.json')
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"✅ Wrote manifest ({len(manifest)} datasets) to {manifest_path}")
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description="Export observations to GeoJSON for the map")
    parser.add_argument("--input", default=None,
                        help="Input CSV (default: mushroom_clusters.csv, else "
                             "mushroom_observations_enriched.csv)")
    parser.add_argument("--output", default=None,
                        help="Exact combined GeoJSON output path; also sets the data directory")
    parser.add_argument("--data-dir", default=None,
                        help="Output directory served by the frontend (legacy alias)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join('public', 'data')
    combined_path = args.output
    if combined_path is not None:
        data_dir = os.path.dirname(combined_path) or data_dir
        if os.path.splitext(combined_path)[1].lower() != '.geojson':
            combined_path = os.path.join(data_dir, os.path.basename(combined_path))

    input_path = args.input
    if input_path is None:
        for candidate in ("mushroom_clusters.csv", "mushroom_observations_enriched.csv"):
            if os.path.exists(candidate):
                input_path = candidate
                break
    if not input_path or not os.path.exists(input_path):
        raise SystemExit("No input CSV found. Run enrich_with_rasters.py / cluster.py first.")

    print(f"📂 Loading {input_path}...")
    df = pd.read_csv(input_path)
    export_all(df, data_dir=data_dir, combined_path=combined_path)


if __name__ == "__main__":
    main()
