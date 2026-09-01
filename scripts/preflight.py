"""Pre-flight credential check for the data pipeline.

Prints which data sources are ready and which will be skipped — with the env var
or file to set for each — so a run's gaps are predictable before it starts.

    python preflight.py            # standalone report

``run_pipeline.py`` calls :func:`print_preflight` at the start of every run.
"""

import json
import os
import subprocess
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
        return False, ("disabled by SKIP_EARTH_ENGINE=1 — unset it (or set 0) to sample every "
                       "environmental layer from Earth Engine instead of downloading rasters")
    cred = Path.home() / ".config" / "earthengine" / "credentials"
    if os.environ.get("EARTHENGINE_PROJECT") or cred.exists():
        return True, ""
    return False, "run `earthengine authenticate` and set EARTHENGINE_PROJECT"


def earth_engine_ready():
    """(ready, note) for Earth Engine — the credential the whole pipeline hangs on."""
    return _earth_engine_ready()


# Where to look for a Google Cloud project id, in the order the pipeline trusts.
EE_PROJECT_HELP = """\
EARTHENGINE_PROJECT is the id of a Google Cloud project that is registered for
Earth Engine. It is the project id (e.g. "my-project-451208"), not the display
name and not the project number. To find yours:

  1. https://console.cloud.google.com/ — the project picker lists every project
     with its ID column. Create one if you have none.
  2. https://code.earthengine.google.com/ — the Earth Engine Code Editor shows
     the active project in the top-right; it also appears in the Assets tab.
  3. Not registered yet? https://code.earthengine.google.com/register attaches a
     Cloud project to Earth Engine (free for noncommercial use).
  4. Already using gcloud? `gcloud config get-value project` prints the current one.

Then either export it or put it in .env at the repo root:

    EARTHENGINE_PROJECT=your-project-id
"""


def resolve_earthengine_project():
    """(project_id, source) for the project Earth Engine will actually use.

    Returns (None, None) when nothing is configured, in which case
    ``EE_PROJECT_HELP`` explains where to find one.
    """
    _load_env()

    value = os.environ.get("EARTHENGINE_PROJECT")
    if value:
        return value, "EARTHENGINE_PROJECT"

    # The stored Earth Engine credential often carries the project it was
    # authorised against, under one of a couple of key names across versions.
    cred = Path.home() / ".config" / "earthengine" / "credentials"
    if cred.exists():
        try:
            data = json.loads(cred.read_text(encoding="utf-8"))
            for key in ("project", "project_id", "quota_project_id"):
                if data.get(key):
                    return data[key], f"{cred} ({key})"
        except (ValueError, OSError):
            pass

    # Finally the active gcloud project, which is usually the same one.
    try:
        out = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        candidate = (out.stdout or "").strip()
        if candidate and candidate != "(unset)":
            return candidate, "gcloud config get-value project"
    except (OSError, subprocess.SubprocessError):
        pass

    return None, None


def print_earthengine_project():
    """Report the resolved project id, or explain where to find one."""
    project, source = resolve_earthengine_project()
    if project:
        print(f"Earth Engine project: {project}   (from {source})")
        if source != "EARTHENGINE_PROJECT":
            print("  Set EARTHENGINE_PROJECT to pin it explicitly.")
    else:
        print("Earth Engine project: not configured\n")
        print(EE_PROJECT_HELP)
    return project


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
    """Return [(source, enriches, ready, note)] for every pipeline data source.

    Earth Engine now supplies every environmental column, so one credential
    covers the whole enrichment. The raster downloads it replaced are listed
    after it as the fallback path — they only matter when EE is unavailable, or
    when FETCH_RASTERS=1 asks for the local rasters that
    ``validate_wetness.py raster`` and the Coverage page read.
    """
    _load_env()

    ee_ok, ee_note = _earth_engine_ready()
    cds_ok, cds_note = _cds_ready()
    otk = bool(os.environ.get("OPENTOPOGRAPHY_API_KEY"))

    checks = [
        ("iNaturalist observations", "the observations themselves", True, "public, no key"),
        ("Earth Engine (all environmental layers)",
         "ndvi, soil_moisture, prcp_d0..6, tmax/tmin_d0..6, land_cover, "
         "elevation, slope, aspect, solar/wind exposure, water retention",
         ee_ok, ee_note),
    ]
    if ee_ok:
        return checks

    # Fallback path — only reachable when Earth Engine is not available.
    return checks + [
        ("Precipitation (CHIRPS download)", "prcp_d0..6", True, "fallback — public, no key"),
        ("Land cover (ESA WorldCover download)", "land_cover / land_cover_label", True,
         "fallback — public, no key"),
        ("Temperature (Open-Meteo)", "tmax/tmin_d0..6", True, "fallback — public, no key"),
        ("Elevation + terrain (OpenTopography DEM)",
         "elevation, slope, aspect, solar/wind exposure, water retention", otk,
         "fallback" if otk else "fallback — set OPENTOPOGRAPHY_API_KEY (free at portal.opentopography.org)"),
        ("Soil moisture (ERA5-Land / CDS)", "soil_moisture", cds_ok,
         "fallback" if cds_ok else f"fallback — {cds_note}"),
    ]


def print_preflight():
    checks = check_preflight()
    print("\n── Pipeline pre-flight ───────────────────────────────────────────────")
    print_earthengine_project()
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
    import sys

    # `python preflight.py --ee-project` answers just "which project id?"
    if "--ee-project" in sys.argv:
        sys.exit(0 if print_earthengine_project() else 1)
    print_preflight()
