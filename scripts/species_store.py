"""Per-species CSV data store.

Source observations live as one lightweight CSV per species under
``data/species/``, and their enriched counterparts under ``data/enriched/`` —
there is no monolithic combined CSV. Every script reads and writes through the
helpers here so the layout stays consistent.

    data/
      species/<slug>.csv     raw observations, one file per species
      enriched/<slug>.csv    enriched observations (clusters folded in)

Set DATA_DIR to relocate the whole store (defaults to ``data``).
"""
import glob
import json
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = os.getenv('DATA_DIR', 'data')
SPECIES_DIR = os.path.join(DATA_DIR, 'species')
ENRICHED_DIR = os.path.join(DATA_DIR, 'enriched')

# Marker written next to the enriched files when a full enrichment run finishes,
# so the pipeline can tell a complete store from an interrupted one.
ENRICHED_DONE = os.path.join(ENRICHED_DIR, '.done')


def slugify(name):
    """Filesystem-safe species slug, matching export_geojson's slugs so a
    species maps to the same name across CSV and GeoJSON stores."""
    slug = re.sub(r'[^a-z0-9]+', '-', str(name).strip().lower()).strip('-')
    return slug or 'unknown'


def species_csv_path(species, base=SPECIES_DIR):
    return os.path.join(base, f"{slugify(species)}.csv")


def list_species_files(base=SPECIES_DIR):
    return sorted(glob.glob(os.path.join(base, '*.csv')))


def species_slugs(base=SPECIES_DIR):
    return [os.path.splitext(os.path.basename(p))[0] for p in list_species_files(base)]


def load_all(base=SPECIES_DIR):
    """Concatenate every per-species CSV in `base` into one DataFrame (empty
    DataFrame if the store has no files). Used wherever a stage needs the whole
    dataset in memory (raster downloads, global clustering, GeoJSON export)."""
    frames = []
    for path in list_species_files(base):
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:  # noqa: BLE001 — skip an unreadable file, keep the rest
            print(f"[!] Could not read {path}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def write_split(df, base=SPECIES_DIR, key='species', dedup='uuid', merge=True):
    """Split `df` by the `key` column and write one CSV per species.

    When `merge` is set, each group is unioned with any CSV already on disk for
    that species and de-duplicated on `dedup` (keeping the first / existing row)
    — so an incremental fetch adds to a species file without dropping history.
    Set merge=False to overwrite (e.g. re-writing enriched files in place).

    Returns {slug: row_count} for what was written.
    """
    if df is None or df.empty or key not in df.columns:
        return {}
    os.makedirs(base, exist_ok=True)
    written = {}
    for species, group in df.groupby(df[key].fillna('Unknown')):
        path = species_csv_path(species, base)
        if merge and os.path.exists(path):
            try:
                existing = pd.read_csv(path)
                # Drop entirely-empty columns from each side before concat: an
                # all-NA column (e.g. an enrichment field not yet filled for this
                # species) triggers a pandas FutureWarning about dtype inference,
                # and carries no data anyway — the other side supplies it if it
                # has values, and stages re-create any missing column.
                parts = [p.dropna(axis=1, how='all') for p in (existing, group) if not p.empty]
                group = pd.concat(parts, ignore_index=True) if parts else group
            except Exception:  # noqa: BLE001 — a corrupt file shouldn't lose the new rows
                pass
        if dedup and dedup in group.columns:
            group = group.drop_duplicates(subset=dedup, keep='first')
        group.to_csv(path, index=False)
        written[slugify(species)] = len(group)
    return written


def store_counts(base=SPECIES_DIR):
    """{slug: row_count} across the store, cheap enough for progress summaries."""
    counts = {}
    for path in list_species_files(base):
        try:
            counts[os.path.splitext(os.path.basename(path))[0]] = sum(1 for _ in open(path, encoding='utf-8', errors='ignore')) - 1
        except OSError:
            pass
    return counts


def write_geojson_tiles(df, base_path=None):
    """Write observations to per‑species, per‑tile GeoJSON files."""
    if df is None or df.empty or not all(c in df.columns for c in ['species', 'lat', 'lon', 'olc']):
        return {}

    base_path = Path(base_path or 'public/data')
    written = {}

    valid_df = df.dropna(subset=['lat', 'lon', 'olc', 'species'])

    for (raw_species, tile), group in valid_df.groupby(['species', 'olc']):
        species_slug = slugify(raw_species)
        species_path = base_path / species_slug
        species_path.mkdir(parents=True, exist_ok=True)
        file_path = species_path / f"{tile}.geojson"

        features = []
        for _, row in group.iterrows():
            properties = {}
            for k, v in row.items():
                if k not in ['lat', 'lon', 'geometry']:
                    if pd.isna(v):
                        properties[k] = None
                    elif isinstance(v, (datetime, pd.Timestamp)):
                        properties[k] = v.isoformat()
                    else:
                        properties[k] = v

            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(row.lon), float(row.lat)]
                }
            }
            features.append(feature)

        if file_path.exists():
            try:
                existing = json.loads(file_path.read_text(encoding='utf-8'))
                existing_features = existing.get('features', [])
                seen = {f['properties'].get('inat_id') for f in existing_features if f.get('properties')}
                for f in features:
                    inat_id = f['properties'].get('inat_id')
                    if inat_id is None or inat_id not in seen:
                        existing_features.append(f)
                features = existing_features
            except Exception:
                pass

        geojson = {"type": "FeatureCollection", "features": features}
        file_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding='utf-8')
        written[(species_slug, tile)] = len(features)

    return written