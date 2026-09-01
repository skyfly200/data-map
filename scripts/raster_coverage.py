"""Summarize the raster cache: what layers exist, for which dates, over what area.

Scans the environmental-layer directories that ``fetch.py`` / ``terrain_pipeline.py``
populate (CHIRPS precip, ERA5-Land soil, NDVI, tree cover, DEM, WorldCover) and
writes ``public/data/coverage.json`` — a lightweight manifest the Nuxt "Coverage"
page reads to show, per layer, the file count, date range, footprint on disk, and
geographic extent, plus a date→layers index so gaps are visible at a glance.

    python raster_coverage.py                 # scan default dirs → public/data/coverage.json
    python raster_coverage.py --pretty        # human-readable JSON

Reading bounds needs rasterio; if it isn't available (or a file can't be opened,
e.g. a NetCDF subdataset), the file is still counted with its date and size — the
page degrades gracefully to counts and dates without an extent.
"""

import argparse
import glob
import json
import os
import re
from datetime import datetime, timezone

try:
    import rasterio
    from rasterio.warp import transform_bounds
except Exception:  # pragma: no cover - rasterio optional
    rasterio = None
    transform_bounds = None

# Layer registry: key, human label, directory, and glob. `dated` marks layers
# whose files carry a YYYY-MM-DD in the name (a time series) vs static coverage.
LAYERS = [
    {"key": "precip", "label": "CHIRPS precipitation", "dir": "precip", "glob": "*.tif", "dated": True},
    {"key": "soil", "label": "ERA5-Land soil moisture", "dir": "soil", "glob": "*.nc", "dated": True},
    {"key": "ndvi", "label": "NDVI", "dir": "ndvi", "glob": "*.tif", "dated": True},
    {"key": "treecover", "label": "Tree cover", "dir": "treecover", "glob": "*.tif", "dated": False},
    {"key": "dem", "label": "DEM / terrain", "dir": "dem", "glob": "*.tif", "dated": False},
    {"key": "terrain", "label": "Terrain exposure layers", "dir": os.path.join("data", "terrain"), "glob": "*.tif", "dated": False},
    {"key": "worldcover", "label": "ESA WorldCover", "dir": "world_cover", "glob": "*.tif", "dated": False},
]

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _date_of(name):
    m = _DATE_RE.search(name)
    return m.group(1) if m else None


def _bounds_wgs84(path):
    """(minLon, minLat, maxLon, maxLat, width, height) or Nones if unreadable."""
    if rasterio is None:
        return (None, None, None, None, None, None)
    try:
        with rasterio.open(path) as src:
            b = src.bounds
            if src.crs and str(src.crs) != "EPSG:4326":
                b = transform_bounds(src.crs, "EPSG:4326", *b, densify_pts=21)
            return (b[0], b[1], b[2], b[3], src.width, src.height)
    except Exception:
        return (None, None, None, None, None, None)


def _union(bbox, other):
    if other[0] is None:
        return bbox
    if bbox is None:
        return [other[0], other[1], other[2], other[3]]
    return [min(bbox[0], other[0]), min(bbox[1], other[1]),
            max(bbox[2], other[2]), max(bbox[3], other[3])]


def scan_layer(spec):
    paths = sorted(glob.glob(os.path.join(spec["dir"], spec["glob"])))
    files = []
    dates = []
    bbox = None
    total_bytes = 0
    for path in paths:
        name = os.path.basename(path)
        date = _date_of(name)
        size = os.path.getsize(path)
        total_bytes += size
        minlon, minlat, maxlon, maxlat, w, h = _bounds_wgs84(path)
        entry = {"name": name, "date": date, "bytes": size, "width": w, "height": h}
        if minlon is not None:
            entry["bbox"] = [round(minlon, 4), round(minlat, 4), round(maxlon, 4), round(maxlat, 4)]
            bbox = _union(bbox, (minlon, minlat, maxlon, maxlat))
        if date:
            dates.append(date)
        files.append(entry)

    dates = sorted(set(dates))
    return {
        "key": spec["key"],
        "label": spec["label"],
        "dir": spec["dir"],
        "dated": spec["dated"],
        "count": len(files),
        "total_bytes": total_bytes,
        "dates": dates,
        "date_range": [dates[0], dates[-1]] if dates else None,
        "bbox": [round(v, 4) for v in bbox] if bbox else None,
        "files": files,
    }


def build_coverage(layers=LAYERS):
    present = []
    date_index = {}
    for spec in layers:
        if not os.path.isdir(spec["dir"]):
            continue
        summary = scan_layer(spec)
        if summary["count"] == 0:
            continue
        present.append(summary)
        for d in summary["dates"]:
            date_index.setdefault(d, [])
            if summary["key"] not in date_index[d]:
                date_index[d].append(summary["key"])

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_bytes": sum(l["total_bytes"] for l in present),
        "layers": present,
        "date_index": dict(sorted(date_index.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize the raster cache into coverage.json")
    parser.add_argument("--output", default=os.path.join("public", "data", "coverage.json"))
    parser.add_argument("--pretty", action="store_true", help="indent the JSON for reading")
    args = parser.parse_args()

    coverage = build_coverage()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(coverage, f, indent=2 if args.pretty else None)

    n_layers = len(coverage["layers"])
    n_dates = len(coverage["date_index"])
    mb = coverage["total_bytes"] / 1e6
    print(f"✅ Coverage: {n_layers} layers, {n_dates} dated snapshots, "
          f"{mb:.1f} MB → {args.output}")
    for l in coverage["layers"]:
        rng = f"{l['date_range'][0]}…{l['date_range'][1]}" if l["date_range"] else "static"
        print(f"   • {l['label']}: {l['count']} files, {rng}, {l['total_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
