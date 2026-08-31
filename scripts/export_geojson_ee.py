import argparse
import glob
import json
import math
import os
import re
import sys
import pandas as pd
import species_store as store

PROPERTY_COLUMNS = [
    "species", "date", "location", "elevation", "land_cover_label",
    "water_mask", "exclude_reason", "ndvi", "soil_moisture",
    "solar_exposure", "wind_exposure", "water_retention", "slope", "aspect",
    "num_identification_agreements", "cluster", "tavg", "tmin", "tmax",
    *(f"prcp_d{i}" for i in range(7)),
    *(f"tmax_d{i}" for i in range(7)),
    *(f"tmin_d{i}" for i in range(7)),
]
INT_COLUMNS = {"cluster", "num_identification_agreements"}

def _clean(value, as_int=False):
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return None
    if as_int:
        try:
            return int(value)
        except:
            return None
    if hasattr(value, "item"):
        value = value.item()
    return value

def to_geojson(df):
    features = []
    present = [c for c in PROPERTY_COLUMNS if c in df.columns]
    for _, row in df.iterrows():
        lon, lat = _clean(row.get("lon")), _clean(row.get("lat"))
        if lon is None or lat is None:
            continue
        props = {c: _clean(row.get(c), as_int=c in INT_COLUMNS) for c in present}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props,
        })
    return {"type": "FeatureCollection", "features": features}

def export_all(df, data_dir='public/data', combined_path=None, group_by='genus'):
    species_dir = os.path.join(data_dir, 'species')
    os.makedirs(species_dir, exist_ok=True)
    group_col = group_by if group_by in df.columns else 'species'
    if group_col in df.columns:
        for group_name, _ in df[group_col].fillna("Unknown").value_counts().items():
            slug = re.sub(r'[^a-z0-9]+', '-', str(group_name).strip().lower()).strip('-')
            subset = df[df[group_col].fillna("Unknown") == group_name]
            with open(os.path.join(species_dir, f"{slug}.geojson"), "w") as f:
                json.dump(to_geojson(subset), f)
    with open(os.path.join(data_dir, 'observations.geojson'), "w") as f:
        json.dump(to_geojson(df), f)

if __name__ == "__main__":
    base = store.ENRICHED_DIR if store.list_species_files(store.ENRICHED_DIR) else store.SPECIES_DIR
    df = store.load_all(base)
    export_all(df, data_dir='public/data')
