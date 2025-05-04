import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse


def cluster_environmental(df, features=None, n_clusters=4):
    if features is None:
        features = [
            'ndvi',
            'soil_moisture',
            'prcp_d0', 'prcp_d1', 'prcp_d2', 'prcp_d3', 'prcp_d4', 'prcp_d5', 'prcp_d6'
        ]

    # Drop rows with missing values in those features
    df_cluster = df.dropna(subset=features).copy()

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[features])

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)

    # Merge back into original DataFrame
    df = df.merge(df_cluster[['uuid', 'cluster']], on='uuid', how='left')
    print(f"✅ Assigned {df['cluster'].notnull().sum()} rows to {n_clusters} clusters")
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
