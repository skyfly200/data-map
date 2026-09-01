import os
import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import argparse

import species_store as store

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def stage_output_path(input_path, suffix, output_dir='.'):
    if not input_path:
        raise ValueError('Input path is required')
    stem = os.path.splitext(os.path.basename(input_path))[0]
    filename = f"{stem}{suffix}.csv"
    if output_dir in (None, '', '.'):
        return filename
    return os.path.join(output_dir, filename)


def cluster_environmental(df, features=None, n_clusters=4):
    if features is None:
        features = [
            'ndvi',
            'soil_moisture',
            'prcp_d0', 'prcp_d1', 'prcp_d2', 'prcp_d3', 'prcp_d4', 'prcp_d5', 'prcp_d6',
            # Terrain exposure derived from the DEM (terrain_pipeline.py)
            'solar_exposure', 'wind_exposure', 'water_retention'
        ]

    # Only use features that are actually present and not entirely empty. A layer
    # that never got downloaded (e.g. NDVI still queued in Drive) would otherwise
    # drop almost every row via listwise deletion, collapsing the clustering.
    usable = [f for f in features
              if f in df.columns and df[f].notna().any()]
    dropped = [f for f in features if f not in usable]
    if dropped:
        print(f"⚠️  Skipping features with no data: {', '.join(dropped)}")
    if not usable:
        print("[!] No usable features present — cannot cluster. Run the pipeline first.")
        df['cluster'] = None
        return df

    # Keep any row that has at least one usable feature; impute the rest (column
    # mean) so a single missing layer no longer excludes the observation.
    df_cluster = df.dropna(subset=usable, how='all').copy()

    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df_cluster[usable])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Can't form more clusters than we have samples.
    k = min(n_clusters, len(df_cluster))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

    # Assign clusters back onto the original DataFrame by index
    df['cluster'] = None
    df.loc[df_cluster.index, 'cluster'] = df_cluster['cluster'].values
    print(f"✅ Assigned {df['cluster'].notnull().sum()} rows to {k} clusters "
          f"using {len(usable)} feature(s): {', '.join(usable)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Cluster mushroom observations by environmental similarity")
    parser.add_argument("--input", default=None,
                        help="Single enriched CSV (default: the per-species enriched store data/enriched/)")
    parser.add_argument("--output", default=None,
                        help="Single output CSV (store mode writes cluster labels back into data/enriched/)")
    # Cluster count: --clusters wins, else CLUSTER_COUNT env var, else 4. This lets
    # run_pipeline.py (which invokes cluster.py without --clusters) tune k via the env.
    default_clusters = 4
    env_clusters = os.getenv('CLUSTER_COUNT', '').strip()
    if env_clusters:
        try:
            default_clusters = max(1, int(env_clusters))
        except ValueError:
            print(f"⚠️  Ignoring invalid CLUSTER_COUNT={env_clusters!r}; using {default_clusters}.")
    parser.add_argument("--clusters", type=int, default=default_clusters,
                        help="Number of clusters to form (default: CLUSTER_COUNT env var or 4)")
    args = parser.parse_args()

    store_mode = args.input is None

    if store_mode:
        # Cluster globally across every enriched species file (KMeans on the
        # shared environmental features), then write the cluster label back into
        # each per-species enriched file so the labels stay with their data.
        df = store.load_all(store.ENRICHED_DIR)
        if df.empty:
            raise SystemExit(f"No enriched CSVs in {store.ENRICHED_DIR}/. Run enrich_with_rasters.py first.")
        print(f"📂 Loaded {len(df)} enriched rows from {store.ENRICHED_DIR}/ ({df['species'].nunique()} species).")
        df = cluster_environmental(df, n_clusters=args.clusters)
        # Use GROUP_BY env var to determine grouping for enriched files too
        group_by = os.getenv('GROUP_BY', 'genus')
        key_column = group_by if group_by in df.columns else 'species'
        written = store.write_split(df, base=store.ENRICHED_DIR, key=key_column, merge=False)
        print(f"💾 Wrote cluster labels back into {len(written)} {key_column} file(s) under {store.ENRICHED_DIR}/.")
        print("✅ Done.")
        return

    output_path = args.output or stage_output_path(args.input, '_clusters')

    print(f"📂 Loading {args.input}...")
    df = pd.read_csv(args.input)

    df = cluster_environmental(df, n_clusters=args.clusters)

    print(f"💾 Saving with clusters to {output_path}...")
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print("✅ Done.")


if __name__ == "__main__":
    main()
