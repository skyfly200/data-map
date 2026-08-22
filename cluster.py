import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import argparse


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

    # Merge back into original DataFrame
    df = df.merge(df_cluster[['uuid', 'cluster']], on='uuid', how='left')
    print(f"✅ Assigned {df['cluster'].notnull().sum()} rows to {k} clusters "
          f"using {len(usable)} feature(s): {', '.join(usable)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Cluster mushroom observations by environmental similarity")
    parser.add_argument("--input", default="mushroom_observations_enriched.csv", help="Path to enriched CSV file")
    parser.add_argument("--output", default="mushroom_clusters.csv", help="Output CSV with cluster labels")
    parser.add_argument("--clusters", type=int, default=4, help="Number of clusters to form")
    args = parser.parse_args()

    print(f"📂 Loading {args.input}...")
    df = pd.read_csv(args.input)

    df = cluster_environmental(df, n_clusters=args.clusters)

    print(f"💾 Saving with clusters to {args.output}...")
    df.to_csv(args.output, index=False)
    print("✅ Done.")


if __name__ == "__main__":
    main()
