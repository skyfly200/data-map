import os
import subprocess
import sys
from pathlib import Path


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


def run_step(label, python_executable, script_name):
    print(f"\n=== {label} ===")
    result = subprocess.run([python_executable, script_name], check=False)
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

    for label, script in [
        ("Fetch iNaturalist observations", "iNat.py"),
        ("Download environmental layers", "fetch.py"),
        ("Enrich observations with rasters", "enrich_with_rasters.py"),
        ("Cluster observations", "cluster.py"),
        ("Export GeoJSON for map", "export_geojson.py"),
    ]:
        run_step(label, python_executable, script)

    print("\n✅ Full data pipeline completed successfully.")


if __name__ == "__main__":
    main()
