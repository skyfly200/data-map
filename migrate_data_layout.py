"""One-shot migration to the per-species CSV layout.

Moves away from the monolithic ``mushroom_observations*.csv`` files in the repo
root to a per-species store:

    data/species/<slug>.csv     raw observations, one file per species
    data/enriched/<slug>.csv    enriched observations (if any enriched CSVs exist)

What it does:
  1. Finds every ``mushroom_observations*.csv`` (and the legacy
     ``mushroom_clusters.csv``) in the root.
  2. Classifies each as raw or enriched by its columns.
  3. Merges all raw files (union, de-duplicated on uuid) and splits them by
     species into data/species/. Enriched files are split into data/enriched/.
  4. Archives the original root CSVs into data/archive/ (non-destructive) unless
     --delete is passed.

Idempotent: safe to re-run. Use --dry-run to preview.

    python migrate_data_layout.py            # migrate, archiving originals
    python migrate_data_layout.py --dry-run  # show what would happen
    python migrate_data_layout.py --delete   # migrate and delete originals
"""
import argparse
import glob
import os
import shutil

import pandas as pd

import species_store as store

# A CSV is "enriched" if it carries any of these derived columns.
ENRICHED_MARKERS = ('ndvi', 'cluster', 'soil_moisture', 'solar_exposure', 'land_cover_label')


def _classify(path):
    """('raw' | 'enriched', DataFrame) for a CSV, or (None, None) if unreadable
    or missing the columns we need to place it in the store."""
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Skipping unreadable {path}: {exc}")
        return None, None
    if 'species' not in df.columns:
        print(f"[!] Skipping {path}: no 'species' column")
        return None, None
    kind = 'enriched' if any(c in df.columns for c in ENRICHED_MARKERS) else 'raw'
    return kind, df


def find_source_csvs(root='.'):
    """Root CSVs that belong to the observation store, newest last so that when
    duplicates collide the newest row wins in enriched files."""
    patterns = ['mushroom_observations*.csv', 'mushroom_clusters.csv']
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(root, pat)))
    # De-dupe and sort by mtime so ordering is deterministic.
    paths = sorted(set(paths), key=lambda p: os.path.getmtime(p))
    return paths


def migrate(root='.', dry_run=False, delete=False):
    sources = find_source_csvs(root)
    if not sources:
        print("No mushroom_observations*.csv files found in the root — nothing to migrate.")
        return

    raw_frames, enriched_frames, archived = [], [], []
    print(f"Found {len(sources)} source CSV(s):")
    for path in sources:
        kind, df = _classify(path)
        if kind is None:
            continue
        print(f"  {kind:8}  {len(df):5d} rows  {os.path.basename(path)}")
        (raw_frames if kind == 'raw' else enriched_frames).append(df)
        archived.append(path)

    def _merge_split(frames, base, label):
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        before = len(merged)
        if 'uuid' in merged.columns:
            merged = merged.drop_duplicates(subset='uuid', keep='last')
        species_n = merged['species'].nunique()
        print(f"\n{label}: {before} rows → {len(merged)} after dedup, {species_n} species")
        if dry_run:
            print(f"  (dry-run) would write {species_n} file(s) under {base}/")
            return
        # Fresh store: write=merge False so a re-run replaces rather than stacks.
        written = store.write_split(merged, base=base, merge=False)
        print(f"  ✅ wrote {len(written)} file(s) under {base}/ "
              f"({sum(written.values())} rows total)")

    _merge_split(raw_frames, store.SPECIES_DIR, "Raw observations")
    _merge_split(enriched_frames, store.ENRICHED_DIR, "Enriched observations")

    # Archive (or delete) the originals so the root is clean and the store is the
    # single source of truth.
    if not archived:
        return
    if dry_run:
        action = "delete" if delete else "archive to data/archive/"
        print(f"\n(dry-run) would {action} {len(archived)} original file(s).")
        return

    if delete:
        for path in archived:
            os.remove(path)
        print(f"\n🗑  Deleted {len(archived)} original root CSV(s).")
    else:
        archive_dir = os.path.join(store.DATA_DIR, 'archive')
        os.makedirs(archive_dir, exist_ok=True)
        for path in archived:
            shutil.move(path, os.path.join(archive_dir, os.path.basename(path)))
        print(f"\n📦 Archived {len(archived)} original root CSV(s) → {archive_dir}/")

    print("\n✅ Migration complete. The per-species store under "
          f"{store.DATA_DIR}/ is now the source of truth.")


def main():
    parser = argparse.ArgumentParser(description="Migrate root CSVs to the per-species store")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--delete", action="store_true",
                        help="Delete originals instead of archiving them")
    parser.add_argument("--root", default=".", help="Directory to scan for source CSVs")
    args = parser.parse_args()
    migrate(root=args.root, dry_run=args.dry_run, delete=args.delete)


if __name__ == "__main__":
    main()
