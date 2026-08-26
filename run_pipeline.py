import os
import subprocess
import sys
from pathlib import Path


def stage_output_path(input_path, suffix, output_dir='.'):
    if not input_path:
        raise ValueError('Input path is required')
    stem = Path(input_path).stem
    filename = f"{stem}{suffix}"
    if output_dir in (None, '', '.'):
        return filename
    return str(Path(output_dir) / filename)


def latest_observation_csv(root):
    matches = sorted(root.glob('mushroom_observations*.csv'))
    unique_matches = [p for p in matches if p.name != 'mushroom_observations.csv']
    if unique_matches:
        return str(unique_matches[-1])
    if matches:
        return str(matches[-1])
    return 'mushroom_observations.csv'


def should_skip_fetch(root):
    refresh_all = os.getenv('REFRESH_ALL', '').strip().lower()
    if refresh_all in {'1', 'true', 'yes', 'y', 'on'}:
        return False
    canonical = root / 'mushroom_observations.csv'
    if canonical.exists():
        return True
    return False


def should_skip_stage(path):
    if not path:
        return False
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def load_env_file(path=None):
    config_path = Path(path or os.getenv('ENV_FILE') or '.env')
    if not config_path.exists():
        return {}

    values = {}
    for line in config_path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, value = stripped.split('=', 1)
        values[key.strip()] = value.strip().strip('"\'')
    return values


def load_env_into_os(path=None):
    for key, value in load_env_file(path).items():
        os.environ.setdefault(key, value)


def _python_candidates():
    candidates = []
    env_python = os.getenv('DATA_MAP_PYTHON') or os.getenv('PYTHON_EXECUTABLE')
    if env_python:
        candidates.append(env_python)
    candidates.append(sys.executable)
    candidates.append(r"C:\Users\skyfl\AppData\Local\Python\pythoncore-3.14-64\python.exe")
    deduped = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _resolve_python():
    for candidate in _python_candidates():
        try:
            result = subprocess.run(
                [candidate, '-c', 'import pyinaturalist, meteostat; print("ok")'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if result.returncode == 0:
                return candidate
        except Exception:
            pass
    return sys.executable


def run_step(label, python_executable, script_name, *args):
    print(f"\n=== {label} ===")
    cmd = [python_executable, script_name, *args]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    env_file = Path(os.getenv('ENV_FILE', '.env'))
    if env_file.exists():
        print(f"Loading env file: {env_file}")
        load_env_into_os(env_file)

    # Report which data sources are configured before doing any work, so the
    # run's gaps (skipped terrain / NDVI / soil) are predictable up front.
    try:
        from preflight import print_preflight
        print_preflight()
    except Exception as exc:
        print(f"[!] Pre-flight check skipped: {exc}")

    refresh_all = os.getenv('REFRESH_ALL', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

    python_executable = _resolve_python()
    print(f"Using Python interpreter: {python_executable}")

    # The whole pipeline reads and writes the per-species store under data/;
    # each script defaults to it, so stages need no CSV paths passed between them.
    import species_store as store

    # 1. Observations — skip the network fetch when the store already has data.
    species_files = store.list_species_files(store.SPECIES_DIR)
    if not refresh_all and species_files:
        print(f"Using cached observations in {store.SPECIES_DIR}/ "
              f"({len(species_files)} species files); skipping iNaturalist fetch.")
    else:
        run_step("Fetch iNaturalist observations", python_executable, "iNat.py")

    # 2. Enrichment — env-layer downloads + raster sampling → per-species enriched
    # store. Only skip when a full run finished (.done marker); a bare checkpoint
    # means an interrupted run, which enrich_with_rasters resumes automatically.
    if not refresh_all and os.path.exists(store.ENRICHED_DONE):
        print(f"Skipping enrichment: {store.ENRICHED_DIR}/ already complete.")
    else:
        run_step("Download environmental layers", python_executable, "fetch.py")
        run_step("Process terrain DEM", python_executable, "terrain_pipeline.py")
        run_step("Enrich observations with rasters", python_executable, "enrich_with_rasters.py")

    # 3. Clustering — global KMeans, cluster labels written back into the store.
    run_step("Cluster observations", python_executable, "cluster.py")

    # 4. GeoJSON export for the map (per-species files + combined + manifest).
    run_step("Export GeoJSON for map", python_executable, "export_geojson.py")

    # 5. Summarize the raster cache for the Coverage page (best effort — a missing
    # rasterio or empty cache just yields a smaller summary, never a hard fail).
    try:
        run_step("Summarize raster coverage", python_executable, "raster_coverage.py")
    except Exception as exc:
        print(f"[!] Raster coverage summary skipped: {exc}")

    print("\n✅ Full data pipeline completed successfully.")


if __name__ == "__main__":
    main()
