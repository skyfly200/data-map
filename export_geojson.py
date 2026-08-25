"""Export the enriched/clustered observations to a static GeoJSON file.

The Nuxt frontend (deployed on Netlify) reads this file statically from
``public/data/observations.geojson`` — no backend or database required. Run
this after ``cluster.py`` (or after ``enrich_with_rasters.py`` if you haven't
clustered) and commit the result; the commit triggers a Netlify redeploy.

    python export_geojson.py
    python export_geojson.py --input mushroom_observations_enriched.csv
"""

import argparse
import glob
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
    "water_mask",
    "exclude_reason",
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


def _read_features(path):
    """Features from a GeoJSON file, or [] if unreadable."""
    try:
        with open(path) as f:
            gj = json.load(f)
        return gj.get("features", []) if isinstance(gj, dict) else []
    except (OSError, ValueError):
        return []


def _species_label(features, fallback):
    for feat in features:
        s = (feat.get("properties") or {}).get("species")
        if s:
            return s
    return fallback


def rebuild_from_species_dir(data_dir=os.path.join('public', 'data')):
    """Rebuild the canonical combined dataset + manifest from EVERY per-species
    file on disk.

    ``export_all`` overwrites the combined file and manifest on each run but
    never clears ``species/``. A later, smaller run (or an on-demand fetch)
    would otherwise shrink "All species" to just that run's species while the
    other per-species files lingered — leaving the manifest listing a fraction
    of what was actually processed. Deriving both artifacts from the union of
    the species directory makes every export self-healing: partial runs add to,
    and never subtract from, what the frontend can see.
    """
    data_dir = data_dir or os.path.join('public', 'data')
    species_dir = os.path.join(data_dir, 'species')
    files = sorted(glob.glob(os.path.join(species_dir, '*.geojson')))

    all_features = []
    entries = []
    for path in files:
        feats = _read_features(path)
        if not feats:
            continue
        all_features.extend(feats)
        slug = os.path.splitext(os.path.basename(path))[0]
        label = _species_label(feats, slug)
        entries.append({
            "id": slug,
            "label": f"{label} ({len(feats)})",
            "path": f"/data/species/{slug}.geojson",
            "count": len(feats),
        })

    combined_path = os.path.join(data_dir, 'observations.geojson')
    with open(combined_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f)

    entries.sort(key=lambda e: e["count"], reverse=True)
    manifest = [{
        "id": "all",
        "label": f"All species ({len(all_features)})",
        "path": "/data/observations.geojson",
        "count": len(all_features),
    }] + entries

    manifest_path = os.path.join(data_dir, 'datasets.json')
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Rebuilt combined ({len(all_features)} features) + manifest "
          f"({len(manifest)} datasets) from {len(entries)} species files")
    return manifest


def export_all(df, data_dir=os.path.join('public', 'data'), combined_path=None):
    """Write one GeoJSON per species, then rebuild the combined dataset and
    manifest from the union of everything on disk.

    Layout served by the frontend:
        public/data/observations.geojson        – all species combined
        public/data/species/<slug>.geojson       – one per species
        public/data/datasets.json                – manifest the UI reads
    """
    data_dir = data_dir or os.path.join('public', 'data')
    species_dir = os.path.join(data_dir, 'species')
    os.makedirs(species_dir, exist_ok=True)

    if "species" in df.columns:
        counts = df["species"].fillna("Unknown").value_counts()
        for species, count in counts.items():
            slug = _slugify(species)
            rel = f"/data/species/{slug}.geojson"
            n = _write_geojson(df[df["species"].fillna("Unknown") == species],
                               os.path.join(species_dir, f"{slug}.geojson"))
            print(f"   ✓ {species}: {n} → {rel}")
    else:
        # No species column — write the whole frame as a single species file so
        # the union rebuild still has something to combine.
        _write_geojson(df, os.path.join(species_dir, 'unknown.geojson'))

    # A caller-requested combined path (run_pipeline's CSV-derived name) still
    # gets a copy, but the manifest always points at the canonical union file.
    if combined_path and os.path.abspath(combined_path) != os.path.abspath(
            os.path.join(data_dir, 'observations.geojson')):
        total = _write_geojson(df, combined_path)
        print(f"✅ Wrote {total} features to {combined_path}")

    return rebuild_from_species_dir(data_dir)


def build_parser():
    parser = argparse.ArgumentParser(description="Export observations to GeoJSON for the map")
    parser.add_argument("--input", default=None,
                        help="Input CSV (default: mushroom_clusters.csv, else "
                             "mushroom_observations_enriched.csv)")
    parser.add_argument("--output", default=None,
                        help="Exact combined GeoJSON output path; also sets the data directory")
    parser.add_argument("--data-dir", default=None,
                        help="Output directory served by the frontend (legacy alias)")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="Skip the CSV; just rebuild observations.geojson + "
                             "datasets.json from the existing species/ files.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join('public', 'data')

    if args.reconcile_only:
        rebuild_from_species_dir(data_dir)
        return

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
