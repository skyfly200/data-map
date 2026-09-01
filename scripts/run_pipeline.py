"""The one pipeline entry point, shared by the CLI and the Kaggle notebook.

    python run_pipeline.py          # command line
    run_pipeline.run_all()          # notebook / Colab / any Python session

``run_all`` is the whole sequence — pre-flight, iNaturalist fetch, enrichment,
clustering, GeoJSON export, coverage summary — with the skip rules that decide
what actually needs doing. Callers get identical behaviour, so the notebook
never has to restate the stage order or duplicate a skip rule.
"""

import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# scripts/ holds the stage scripts; the repo root above it is the working
# directory they all expect — the per-species store (data/) and the raster
# caches (dem/, precip/, ndvi/, soil/, world_cover/) are relative to it.
#
# The chdir happens in run_all()'s `working_directory`, not at import: importing
# a module should not move the caller, and the notebook imports this before it
# is ready to run.
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent

# Stage modules import each other by bare name (`import species_store`), so the
# directory holding them has to be importable for anyone importing run_pipeline.
for _path in (SCRIPTS_DIR, ROOT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
load_dotenv(dotenv_path=ROOT_DIR / ".env")

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
    """Run one stage script by absolute path, so the caller's cwd stays the repo root."""
    print(f"\n=== {label} ===")
    cmd = [python_executable, str(SCRIPTS_DIR / script_name), *args]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(f"{label} failed with exit code {result.returncode}")


@contextlib.contextmanager
def working_directory(path):
    """Run inside ``path``, restoring the previous cwd afterwards.

    The stages address their inputs and outputs relative to the repo root, so
    the pipeline has to run from there. Restoring matters when run_all() is
    called from a notebook that then keeps working in its own directory.
    """
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield Path(path)
    finally:
        os.chdir(previous)


def run_all(python_executable=None, root=None):
    """Run the full pipeline. Shared by ``main()`` and the notebook."""
    with working_directory(root or ROOT_DIR):
        _run_stages(python_executable)


def _run_stages(python_executable=None):
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

    python_executable = python_executable or _resolve_python()
    print(f"Using Python interpreter: {python_executable}")
    print(f"Working directory: {os.getcwd()}")

    # The whole pipeline reads and writes the per-species store under data/;
    # each script defaults to it, so stages need no CSV paths passed between them.
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    import species_store as store

    # 1. Observations — run incremental fetch by default so new taxa or fresh sightings are captured.
    skip_fetch = os.getenv('SKIP_INAT_FETCH', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
    species_files_before = set(store.species_slugs(store.SPECIES_DIR))
    counts_before = store.store_counts(store.SPECIES_DIR)

    if skip_fetch and species_files_before:
        print(f"Using cached observations in {store.SPECIES_DIR}/ "
              f"({len(species_files_before)} species files); skipping iNaturalist fetch (SKIP_INAT_FETCH=1).")
    else:
        run_step("Fetch iNaturalist observations", python_executable, "iNat.py")
        species_files_after = set(store.species_slugs(store.SPECIES_DIR))
        counts_after = store.store_counts(store.SPECIES_DIR)
        # If new species files or new rows were added, ensure enrichment runs for them
        if species_files_after != species_files_before or counts_after != counts_before:
            if os.path.exists(store.ENRICHED_DONE):
                try:
                    os.remove(store.ENRICHED_DONE)
                except OSError:
                    pass

    # 2. Enrichment — env-layer downloads + raster sampling → per-species enriched
    # store. Only skip when a full run finished (.done marker); a bare checkpoint
    # means an interrupted run, which enrich_with_rasters resumes automatically.
    if not refresh_all and os.path.exists(store.ENRICHED_DONE):
        print(f"Skipping enrichment: {store.ENRICHED_DIR}/ already complete.")
    else:
        # fetch.py and terrain_pipeline.py download and derive the bulk rasters.
        # Earth Engine serves the same layers as point samples during enrichment,
        # so both are skipped unless EE is off or FETCH_RASTERS=1 asks for the
        # local rasters (still used by validate_wetness.py and the Coverage page).
        from fetch import skip_raster_downloads
        from preflight import earth_engine_ready
        ee_ready, _note = earth_engine_ready()
        if skip_raster_downloads(ee_available=ee_ready):
            print("\nSkipping raster downloads and DEM processing — enrichment samples "
                  "every layer from Earth Engine.\nSet FETCH_RASTERS=1 to download them anyway.")
        else:
            run_step("Download environmental layers", python_executable, "fetch.py")
            run_step("Process terrain DEM", python_executable, "terrain_pipeline.py")
        run_step("Enrich observations", python_executable, "enrich_with_rasters.py")

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


def main():
    run_all()


if __name__ == "__main__":
    main()
