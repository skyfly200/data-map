import os
import subprocess
import sys
from pathlib import Path


def run_step(label, command):
    print(f"\n=== {label} ===")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    env_file = Path(os.getenv('ENV_FILE', '.env'))
    if env_file.exists():
        print(f"Loading env file: {env_file}")

    run_step("Fetch iNaturalist observations", 'python iNat.py')
    run_step("Download environmental layers", 'python fetch.py')
    run_step("Enrich observations with rasters", 'python enrich_with_rasters.py')
    run_step("Cluster observations", 'python cluster.py')
    run_step("Export GeoJSON for map", 'python export_geojson.py')

    print("\n✅ Full data pipeline completed successfully.")


if __name__ == "__main__":
    main()
