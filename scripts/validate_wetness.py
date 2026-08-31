"""Validate the topographic wetness index against observed moisture.

``water_retention`` (TWI, from ``terrain_pipeline.py``) is a *static* prediction
of where water should accumulate based on terrain alone. This script measures how
well it agrees with an *independent, observed* moisture signal.

Because TWI is a potential (not an instantaneous state) the two are expected to
correlate positively but imperfectly — most strongly right after rain, weakest in
drought. TWI is log-scaled, so agreement is assessed with **Spearman rank**
correlation rather than Pearson.

Two modes:

    points   Correlate the columns already in the enriched CSV at observation
             points — a quick check against soil_moisture (ERA5-Land) and ndvi
             (Sentinel-2). ERA5 is ~9 km, so treat it as a coarse sanity check.

    raster   Correlate two rasters pixel-by-pixel over the whole DEM footprint:
             dem/derived/water_retention.tif against a satellite moisture raster
             you provide (Sentinel-1 VV, Sentinel-2 NDMI, SMAP, ...). This gives
             far more statistical power and a spatial residual map.

Examples
--------
    python validate_wetness.py points
    python validate_wetness.py points --refs soil_moisture ndvi
    python validate_wetness.py raster --satellite ndmi.tif
    python validate_wetness.py raster --satellite s1_vv.tif --landcover world_cover/tile.tif
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


TWI_COL = "water_retention"
TWI_RASTER = "dem/derived/water_retention.tif"

# ESA WorldCover codes to drop from a raster comparison (open water, built-up,
# snow/ice) — terrain wetness is not meaningful there.
_MASK_LANDCOVER = {50, 70, 80}


# ─── Point mode ───────────────────────────────────────────────────────────────

def compare_points(csv_path, refs, group_col="land_cover_label"):
    """Correlate the TWI column against reference columns at observation points."""
    df = pd.read_csv(csv_path)

    if TWI_COL not in df.columns:
        raise SystemExit(
            f"'{TWI_COL}' not found in {csv_path}. Re-run enrich_with_rasters.py "
            "so the terrain layers are sampled into the enriched CSV first."
        )

    print(f"\nTWI vs. observed moisture at {len(df)} observation points")
    print("=" * 60)
    for ref in refs:
        if ref not in df.columns:
            print(f"  {ref:16s}  (column not present — skipped)")
            continue
        _report_pair(df[TWI_COL], df[ref], ref)

    # Break the strongest reference down by land cover: TWI should track moisture
    # more cleanly within a single cover type than across mixed vegetation.
    if group_col in df.columns and refs:
        ref = next((r for r in refs if r in df.columns), None)
        if ref:
            print(f"\nBy {group_col} (vs {ref}):")
            for label, sub in df.groupby(group_col):
                pair = sub[[TWI_COL, ref]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(pair) >= 5:
                    rho, _ = spearmanr(pair[TWI_COL], pair[ref])
                    print(f"  {str(label):28s} n={len(pair):4d}  Spearman ρ={rho:+.3f}")


def _report_pair(a, b, label):
    pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"),
                         "b": pd.to_numeric(b, errors="coerce")}).dropna()
    n = len(pair)
    if n < 5:
        print(f"  {label:16s}  n={n:4d}  (too few paired values)")
        return
    rho, p_s = spearmanr(pair["a"], pair["b"])
    r, p_p = pearsonr(pair["a"], pair["b"])
    flag = "" if p_s < 0.05 else "   (not significant)"
    print(f"  {label:16s}  n={n:4d}  Spearman ρ={rho:+.3f} (p={p_s:.1e})"
          f"  Pearson r={r:+.3f}{flag}")


# ─── Raster mode ──────────────────────────────────────────────────────────────

def compare_rasters(twi_path, sat_path, landcover_path=None, max_samples=200_000,
                    out_scatter=None):
    """Correlate TWI against a satellite moisture raster over the DEM footprint.

    The satellite raster is reprojected onto the TWI grid (average resampling),
    so the two can differ in resolution, extent, and CRS.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    with rasterio.open(twi_path) as twi_src:
        twi = twi_src.read(1).astype("float64")
        twi_nodata = twi_src.nodata
        dst_profile = twi_src.profile
        dst_transform = twi_src.transform
        dst_crs = twi_src.crs
        dst_shape = (twi_src.height, twi_src.width)

    # Bring the satellite layer onto the TWI grid.
    sat_on_grid = np.full(dst_shape, np.nan, dtype="float64")
    with rasterio.open(sat_path) as sat_src:
        reproject(
            source=rasterio.band(sat_src, 1),
            destination=sat_on_grid,
            src_transform=sat_src.transform,
            src_crs=sat_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            dst_nodata=np.nan,
        )

    valid = np.isfinite(twi) & np.isfinite(sat_on_grid)
    if twi_nodata is not None:
        valid &= twi != twi_nodata

    # Optionally drop water / built-up / snow using the land-cover raster.
    if landcover_path and os.path.exists(landcover_path):
        lc_on_grid = np.full(dst_shape, np.nan, dtype="float64")
        with rasterio.open(landcover_path) as lc_src:
            reproject(
                source=rasterio.band(lc_src, 1),
                destination=lc_on_grid,
                src_transform=lc_src.transform,
                src_crs=lc_src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan,
            )
        masked = np.isin(np.rint(lc_on_grid), list(_MASK_LANDCOVER))
        valid &= ~masked

    twi_v = twi[valid]
    sat_v = sat_on_grid[valid]
    n = twi_v.size
    if n < 20:
        raise SystemExit(f"Only {n} overlapping valid pixels — check the rasters align.")

    # Subsample for the correlation/plot when the overlap is large.
    if n > max_samples:
        idx = np.random.default_rng(0).choice(n, max_samples, replace=False)
        twi_s, sat_s = twi_v[idx], sat_v[idx]
    else:
        twi_s, sat_s = twi_v, sat_v

    rho, p_s = spearmanr(twi_s, sat_s)
    r, p_p = pearsonr(twi_s, sat_s)

    print(f"\nPixel-wise: {os.path.basename(twi_path)} vs {os.path.basename(sat_path)}")
    print("=" * 60)
    print(f"  overlapping valid pixels : {n:,}")
    print(f"  correlated on            : {twi_s.size:,} (subsampled)" if n > max_samples
          else f"  correlated on            : {n:,}")
    print(f"  Spearman ρ               : {rho:+.3f}  (p={p_s:.1e})")
    print(f"  Pearson  r               : {r:+.3f}  (p={p_p:.1e})")
    print(_interpret(rho))

    if out_scatter:
        _save_scatter(twi_s, sat_s, rho, twi_path, sat_path, out_scatter)


def _interpret(rho):
    if rho >= 0.5:
        return "  → Strong agreement: terrain explains much of the observed moisture pattern."
    if rho >= 0.3:
        return "  → Moderate agreement: terrain is one of several drivers (vegetation, recent rain)."
    if rho >= 0.1:
        return "  → Weak agreement: try a wetter date, a finer satellite layer, or masking by land cover."
    return ("  → Little/negative agreement: check units/date, scale mismatch (e.g. ERA5 too coarse), "
            "or that a drought is masking the terrain signal.")


def _save_scatter(twi, sat, rho, twi_path, sat_path, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib not installed — skipping scatter plot)")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hexbin(twi, sat, gridsize=40, cmap="viridis", mincnt=1)
    ax.set_xlabel("Topographic wetness index (predicted)")
    ax.set_ylabel(f"Satellite moisture: {os.path.basename(sat_path)}")
    ax.set_title(f"Spearman ρ = {rho:+.3f}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"  ✓ scatter written to {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_pts = sub.add_parser("points", help="Correlate enriched-CSV columns at observation points")
    p_pts.add_argument("--csv", default="mushroom_observations_enriched.csv")
    p_pts.add_argument("--refs", nargs="+", default=["soil_moisture", "ndvi"],
                       help="Reference moisture columns to compare against")

    p_ras = sub.add_parser("raster", help="Correlate TWI against a satellite moisture raster")
    p_ras.add_argument("--twi", default=TWI_RASTER)
    p_ras.add_argument("--satellite", required=True,
                       help="Satellite moisture GeoTIFF (Sentinel-1 VV, NDMI, SMAP, ...)")
    p_ras.add_argument("--landcover", default=None,
                       help="Optional WorldCover raster to mask water/built-up/snow")
    p_ras.add_argument("--scatter", default=None, help="Optional output PNG for a hexbin scatter")

    args = parser.parse_args()
    if args.mode == "points":
        compare_points(args.csv, args.refs)
    else:
        compare_rasters(args.twi, args.satellite, args.landcover, out_scatter=args.scatter)


if __name__ == "__main__":
    main()
