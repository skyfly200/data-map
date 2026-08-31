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
            'ndvi', 'soil_moisture',
            'prcp_d0', 'prcp_d1', 'prcp_d2', 'prcp_d3', 'prcp_d4', 'prcp_d5', 'prcp_d6',
            'solar_exposure', 'wind_exposure', 'water_retention'
        ]
    usable = [f for f in features if f in df.columns and df[f].notna().any()]
    if not usable:
        df['cluster'] = None
        return df
    df_cluster = df.dropna(subset=usable, how='all').copy()
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(df_cluster[usable])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    k = min(n_clusters, len(df_cluster))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
    df['cluster'] = None
    df.loc[df_cluster.index, 'cluster'] = df_cluster['cluster'].values
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    default_clusters = 4
    parser.add_argument("--clusters", type=int, default=default_clusters)
    args = parser.parse_args()
    store_mode = args.input is None
    if store_mode:
        df = store.load_all(store.ENRICHED_DIR)
        if df.empty:
            return
        df = cluster_environmental(df, n_clusters=args.clusters)
        group_by = os.getenv('GROUP_BY', 'genus')
        key_column = group_by if group_by in df.columns else 'species'
        store.write_split(df, base=store.ENRICHED_DIR, key=key_column, merge=False)
        return
    output_path = args.output or stage_output_path(args.input, '_clusters')
    df = pd.read_csv(args.input)
    df = cluster_environmental(df, n_clusters=args.clusters)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
