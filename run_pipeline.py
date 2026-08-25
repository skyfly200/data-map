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

    python_executable = _resolve_python()
    print(f"Using Python interpreter: {python_executable}")

    if should_skip_fetch(root):
        print("Using cached iNaturalist observations from mushroom_observations.csv; skipping network fetch.")
        observation_csv = 'mushroom_observations.csv'
    else:
        run_step("Fetch iNaturalist observations", python_executable, "iNat.py")
        observation_csv = latest_observation_csv(root)
        print(f"Using observation input: {observation_csv}")

    enriched_csv = stage_output_path(observation_csv, '_enriched')
    run_step("Download environmental layers", python_executable, "fetch.py")
    run_step("Process terrain DEM", python_executable, "terrain_pipeline.py")
    run_step("Enrich observations with rasters", python_executable, "enrich_with_rasters.py", "--input", observation_csv, "--output", enriched_csv)

    clustered_csv = stage_output_path(enriched_csv, '_clusters')
    run_step("Cluster observations", python_executable, "cluster.py", "--input", enriched_csv, "--output", clustered_csv)

    geojson_output = str(Path('public') / 'data' / f"{Path(clustered_csv).stem}.geojson")
    run_step("Export GeoJSON for map", python_executable, "export_geojson.py", "--input", clustered_csv, "--output", geojson_output)

    print("\n✅ Full data pipeline completed successfully.")


if __name__ == "__main__":
    main()
