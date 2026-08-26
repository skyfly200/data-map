"""Pre-flight credential check for the data pipeline.

Prints which data sources are ready and which will be skipped — with the env var
or file to set for each — so a run's gaps are predictable before it starts.

    python preflight.py            # standalone report

``run_pipeline.py`` calls :func:`print_preflight` at the start of every run.
"""

import os
from pathlib import Path


def _load_env(path=".env"):
    """Populate os.environ from a .env file (no-op if already set / missing)."""
    p = Path(os.getenv("ENV_FILE") or path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _earth_engine_ready():
    if os.environ.get("SKIP_EARTH_ENGINE") == "1":
        return False, "disabled by SKIP_EARTH_ENGINE=1 — unset it (or set 0) to enable NDVI"
    cred = Path.home() / ".config" / "earthengine" / "credentials"
    if os.environ.get("EARTHENGINE_PROJECT") or cred.exists():
        return True, ""
    return False, "run `earthengine authenticate` and set EARTHENGINE_PROJECT"


def _cds_ready():
    # Mirrors cdsapi's own resolution: CDSAPI_URL/KEY env, then CDSAPI_RC, then
    # ~/.cdsapirc. A repo-local .cdsapirc is NOT auto-read by cdsapi.
    if os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"):
        return True, ""
    rc = os.environ.get("CDSAPI_RC")
    if rc and Path(rc).expanduser().exists():
        return True, ""
    if (Path.home() / ".cdsapirc").exists():
        return True, ""
    if Path(".cdsapirc").exists():
        return False, ("found ./.cdsapirc, but cdsapi reads ~/.cdsapirc — move it to your home "
                       "directory or set CDSAPI_RC=$(pwd)/.cdsapirc")
    return False, "create ~/.cdsapirc (or set CDSAPI_URL / CDSAPI_KEY)"


def check_preflight():
    """Return [(source, enriches, ready, note)] for every pipeline data source."""
    _load_env()

    ee_ok, ee_note = _earth_engine_ready()
    cds_ok, cds_note = _cds_ready()
    otk = bool(os.environ.get("OPENTOPOGRAPHY_API_KEY"))

    return [
        ("iNaturalist observations", "the observations themselves", True, "public, no key"),
        ("Precipitation (CHIRPS)", "rain7 / prcp_d0..6", True, "public, no key"),
        ("Land cover (ESA WorldCover)", "land_cover / land_cover_label", True, "public, no key"),
        ("Temperature (Open-Meteo)", "tmin / tmax / tavg", True, "public, no key"),
        ("Elevation + terrain (OpenTopography DEM)",
         "elevation, slope, aspect, solar/wind exposure, water retention", otk,
         "" if otk else "set OPENTOPOGRAPHY_API_KEY (free at portal.opentopography.org)"),
        ("NDVI (Earth Engine)", "ndvi", ee_ok, ee_note),
        ("Soil moisture (ERA5-Land / CDS)", "soil_moisture", cds_ok, cds_note),
    ]


def print_preflight():
    checks = check_preflight()
    print("\n── Pipeline pre-flight ───────────────────────────────────────────────")
    ready = 0
    for name, enriches, ok, note in checks:
        mark = "✅" if ok else "⏭️ "
        print(f"  {mark} {name}")
        print(f"       → {enriches}")
        if note:
            print(f"       {'' if ok else '⚠  skipped: '}{note}")
        ready += 1 if ok else 0
    skipped = len(checks) - ready
    print(f"── {ready} ready, {skipped} will be skipped "
          f"{'(fields above stay empty until configured)' if skipped else ''} ──\n")
    return checks


if __name__ == "__main__":
    print_preflight()
