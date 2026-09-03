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


# ─── Stages ───────────────────────────────────────────────────────────────────
# Each stage is a standalone function so the notebook (or any caller) can re-run
# just one part — e.g. re-do enrichment without re-fetching. ``run_all`` runs
# them in order; every stage does its own chdir + env load + interpreter
# resolution, so the stages are safe to call individually and in any order.

_RESOLVED_PYTHON = None

def _prepare(python_executable=None):
    """Load .env, ensure scripts/ is importable, and resolve the interpreter.

    Idempotent and cheap to call once per stage: the resolved interpreter is
    cached so repeated stage calls don't re-probe it."""
    global _RESOLVED_PYTHON
    env_file = Path(os.getenv('ENV_FILE', '.env'))
    if env_file.exists():
        load_env_into_os(env_file)
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if python_executable:
        return python_executable
    if _RESOLVED_PYTHON is None:
        _RESOLVED_PYTHON = _resolve_python()
    return _RESOLVED_PYTHON


def _refresh_all():
    return os.getenv('REFRESH_ALL', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def run_preflight(root=None):
    """Print which data sources are configured, so gaps are predictable up front."""
    with working_directory(root or ROOT_DIR):
        _prepare()
        try:
            from preflight import print_preflight
            print_preflight()
        except Exception as exc:
            print(f"[!] Pre-flight check skipped: {exc}")


def run_fetch(python_executable=None, root=None):
    """Stage 1 — fetch iNaturalist observations into the per-species store.

    Honours SKIP_INAT_FETCH (reuse the cache). When the fetch adds species or
    rows, the enrichment ``.done`` marker is cleared so enrichment re-runs for
    the new data."""
    with working_directory(root or ROOT_DIR):
        py = _prepare(python_executable)
        import species_store as store
        skip_fetch = os.getenv('SKIP_INAT_FETCH', '').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
        species_files_before = set(store.species_slugs(store.SPECIES_DIR))
        counts_before = store.store_counts(store.SPECIES_DIR)
        if skip_fetch and species_files_before:
            print(f"Using cached observations in {store.SPECIES_DIR}/ "
                  f"({len(species_files_before)} species files); skipping iNaturalist fetch (SKIP_INAT_FETCH=1).")
            return
        run_step("Fetch iNaturalist observations", py, "iNat.py")
        species_files_after = set(store.species_slugs(store.SPECIES_DIR))
        counts_after = store.store_counts(store.SPECIES_DIR)
        if species_files_after != species_files_before or counts_after != counts_before:
            if os.path.exists(store.ENRICHED_DONE):
                try:
                    os.remove(store.ENRICHED_DONE)
                except OSError:
                    pass


def run_enrichment(python_executable=None, root=None):
    """Stage 2 — enrich observations (Earth Engine samples, or cached rasters).

    Skips when a full run already finished (the ``.done`` marker) unless
    REFRESH_ALL is set; an interrupted run resumes automatically."""
    with working_directory(root or ROOT_DIR):
        py = _prepare(python_executable)
        import species_store as store
        if not _refresh_all() and os.path.exists(store.ENRICHED_DONE):
            print(f"Skipping enrichment: {store.ENRICHED_DIR}/ already complete "
                  "(set REFRESH_ALL=1 to force).")
            return
        # Earth Engine serves every layer as a point sample, so the bulk raster
        # downloads are skipped unless EE is off or FETCH_RASTERS=1 asks for them.
        from fetch import skip_raster_downloads
        from preflight import earth_engine_ready
        ee_ready, _note = earth_engine_ready()
        if skip_raster_downloads(ee_available=ee_ready):
            print("\nSkipping raster downloads and DEM processing — enrichment samples "
                  "every layer from Earth Engine.\nSet FETCH_RASTERS=1 to download them anyway.")
        else:
            run_step("Download environmental layers", py, "fetch.py")
            run_step("Process terrain DEM", py, "terrain_pipeline.py")
        run_step("Enrich observations", py, "enrich_with_rasters.py")


def run_clustering(python_executable=None, root=None):
    """Stage 3 — global KMeans; cluster labels written back into the store."""
    with working_directory(root or ROOT_DIR):
        py = _prepare(python_executable)
        run_step("Cluster observations", py, "cluster.py")


def run_export(python_executable=None, root=None):
    """Stage 4 — export the map GeoJSON (per-species + combined + manifest)."""
    with working_directory(root or ROOT_DIR):
        py = _prepare(python_executable)
        run_step("Export GeoJSON for map", py, "export_geojson.py")


def run_coverage(python_executable=None, root=None):
    """Stage 5 — summarize the raster cache for the Coverage page (best effort)."""
    with working_directory(root or ROOT_DIR):
        py = _prepare(python_executable)
        try:
            run_step("Summarize raster coverage", py, "raster_coverage.py")
        except Exception as exc:
            print(f"[!] Raster coverage summary skipped: {exc}")


def report_precision(root=None):
    """Stage 6 — report how much of the terrain enrichment can be trusted."""
    with working_directory(root or ROOT_DIR):
        _prepare()
        try:
            _report_location_precision()
        except Exception as exc:
            print(f"[!] Location precision summary skipped: {exc}")


# The stages in run order, so run_all and the notebook agree on the sequence.
STAGES = [run_fetch, run_enrichment, run_clustering, run_export, run_coverage]


def run_all(python_executable=None, root=None):
    """Run the full pipeline. Shared by ``main()`` and the notebook."""
    with working_directory(root or ROOT_DIR):
        py = _prepare(python_executable)
        print(f"Using Python interpreter: {py}")
        print(f"Working directory: {os.getcwd()}")
        run_preflight(root='.')
        for stage in STAGES:
            stage(py, root='.')
        report_precision(root='.')
        print("\n✅ Full data pipeline completed successfully.")


def _report_location_precision():
    import species_store as store

    # Read the enriched store, which is what the export and the app both see.
    df = store.load_all(store.ENRICHED_DIR)
    if df is None or 'location_precision' not in getattr(df, 'columns', []):
        # Nothing to say until a fetch has run that records the field.
        return
    counts = df['location_precision'].value_counts(dropna=False).to_dict()
    total = int(sum(counts.values()))
    if not total:
        return

    print("\n=== Location precision ===")
    for key in ('precise', 'coarse', 'obscured', 'unknown'):
        n = int(counts.get(key, 0))
        print(f"  {key:9s} {n:7,d}  {n / total * 100:5.1f}%")

    untrustworthy = int(counts.get('obscured', 0)) + int(counts.get('coarse', 0))
    if untrustworthy:
        print(
            f"  [!] {untrustworthy:,} of {total:,} rows ({untrustworthy / total * 100:.1f}%) "
            "carry terrain sampled at a point iNaturalist deliberately moved or "
            "could not pin down. Use the 'Precise coordinates only' filter before "
            "reading terrain relationships."
        )


def main():
    run_all()


if __name__ == "__main__":
    main()
