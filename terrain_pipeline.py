"""Topographic processing pipeline.

Turns a raw Digital Elevation Model (DEM) into a set of terrain-derived
raster layers that describe *microclimate exposure* at each location:

    * slope / aspect           – the geometric building blocks
    * solar_exposure           – potential incoming solar radiation (0-1)
    * wind_exposure            – topographic wind exposure / shelter (0-1)
    * water_retention          – topographic water-retention index (0-1)

The DEM itself is fetched by ``fetch.py`` (``download_srtm_dem``) and the
derived layers written here are sampled at observation points by
``enrich_with_rasters.py`` (``enrich_with_terrain``).

Design note
-----------
The numerical core (``compute_slope_aspect``, ``solar_exposure_index``,
``wind_exposure_index``, ``water_retention_index``) operates purely on
NumPy arrays plus pixel sizes in metres so it can be reasoned about and
unit-tested without any raster I/O.  The rasterio wrappers at the bottom
handle reading the DEM and writing GeoTIFFs of each derived layer.
"""

import math
import os
import sys
import re

import numpy as np

try:  # rasterio is only needed for the file-based entry points
    import rasterio
    from rasterio.warp import transform as warp_transform
except Exception:  # pragma: no cover - allows importing the math core alone
    rasterio = None


# ─── Pixel geometry ───────────────────────────────────────────────────────────

def pixel_size_metres(transform, height, mean_lat):
    """Approximate pixel size (dx, dy) in metres for a geographic raster.

    ``transform`` is an affine transform in degrees (EPSG:4326).  One degree
    of latitude is ~110540 m; one degree of longitude shrinks with latitude.
    """
    deg_x = abs(transform.a)
    deg_y = abs(transform.e)
    dx = deg_x * 111320.0 * math.cos(math.radians(mean_lat))
    dy = deg_y * 110540.0
    return dx, dy


# ─── Slope & aspect ───────────────────────────────────────────────────────────

def compute_slope_aspect(dem, dx, dy):
    """Return (slope_rad, aspect_rad) from a DEM grid.

    Row index increases *southward* (row 0 = north edge), the standard
    orientation of a north-up raster.  ``aspect_rad`` is the compass bearing
    (clockwise from north, 0..2pi) that the downhill slope faces; it is set to
    a nominal value where the surface is flat.
    """
    dem = np.asarray(dem, dtype="float64")

    # np.gradient returns d/d(row) then d/d(col).
    dz_drow, dz_dcol = np.gradient(dem, dy, dx)

    # Convert to east/north partial derivatives.  Column increases eastward,
    # row increases southward, so d/d(north) = -d/d(row).
    dz_de = dz_dcol
    dz_dn = -dz_drow

    slope_rad = np.arctan(np.hypot(dz_de, dz_dn))

    # Downhill direction is the negative gradient; as a compass bearing that is
    # atan2(east-component, north-component) of the steepest-descent vector.
    aspect_rad = np.arctan2(-dz_de, -dz_dn)
    aspect_rad = np.mod(aspect_rad, 2 * math.pi)

    # Flat cells have an undefined aspect – park them at 0.
    flat = slope_rad < 1e-9
    aspect_rad = np.where(flat, 0.0, aspect_rad)

    return slope_rad, aspect_rad


# ─── Solar exposure ───────────────────────────────────────────────────────────

def _sun_positions(latitude, declinations, n_hours=13):
    """Yield (altitude, azimuth) sun positions in radians over daylight hours.

    Averaging over several solar declinations (seasons) yields an annual
    potential-radiation proxy that is a stable property of the terrain.
    """
    lat = math.radians(latitude)
    positions = []
    for dec_deg in declinations:
        dec = math.radians(dec_deg)
        # Hour angle from sunrise to sunset in even steps (H = 0 at solar noon).
        for H_deg in np.linspace(-90, 90, n_hours):
            H = math.radians(H_deg)
            sin_alt = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(H)
            if sin_alt <= 0:  # sun below the horizon
                continue
            alt = math.asin(sin_alt)
            # Solar azimuth measured clockwise from north.
            cos_az = (math.sin(dec) - math.sin(alt) * math.sin(lat)) / (
                math.cos(alt) * math.cos(lat) + 1e-9
            )
            cos_az = max(-1.0, min(1.0, cos_az))
            az = math.acos(cos_az)
            if H > 0:  # afternoon -> sun in the west
                az = 2 * math.pi - az
            positions.append((alt, az))
    return positions


def solar_exposure_index(slope_rad, aspect_rad, latitude, declinations=None, n_hours=13):
    """Potential incoming solar radiation, normalised to 0..1.

    For each modelled sun position the cosine of the solar incidence angle on
    the sloped surface is accumulated (self-shadowed faces contribute zero),
    weighted by ``sin(altitude)`` to approximate air-mass/flux.  The result is
    min-max normalised so 1 = the sunniest terrain in the scene.
    """
    if declinations is None:
        # Winter solstice, equinox, summer solstice -> annual average.
        declinations = [-23.44, 0.0, 23.44]

    total = np.zeros_like(slope_rad, dtype="float64")
    weight = 0.0
    for alt, az in _sun_positions(latitude, declinations, n_hours):
        zenith = math.pi / 2 - alt
        cos_inc = (
            math.cos(zenith) * np.cos(slope_rad)
            + math.sin(zenith) * np.sin(slope_rad) * np.cos(az - aspect_rad)
        )
        cos_inc = np.clip(cos_inc, 0.0, None)  # ignore faces turned away from the sun
        flux = math.sin(alt)  # brighter when the sun is high
        total += cos_inc * flux
        weight += flux

    if weight > 0:
        total /= weight
    return _normalise(total)


# ─── Wind exposure ────────────────────────────────────────────────────────────

def _multiscale_tpi(dem, dx, dy, radii_m=(150, 500, 1500)):
    """Mean Topographic Position Index over several neighbourhood radii.

    TPI = cell elevation - mean elevation of a surrounding window.  Positive on
    ridges/spurs (exposed), negative in valleys (sheltered).  Averaging radii
    captures both fine spurs and broad landforms.
    """
    from scipy.ndimage import uniform_filter

    dem = np.asarray(dem, dtype="float64")
    acc = np.zeros_like(dem)
    for r_m in radii_m:
        win_x = max(1, int(round(r_m / dx)))
        win_y = max(1, int(round(r_m / dy)))
        size = (2 * win_y + 1, 2 * win_x + 1)
        local_mean = uniform_filter(dem, size=size, mode="nearest")
        acc += dem - local_mean
    return acc / len(radii_m)


def wind_exposure_index(dem, slope_rad, aspect_rad, dx, dy,
                        prevailing_wind_deg=270.0, radii_m=(150, 500, 1500)):
    """Topographic wind exposure, normalised to 0..1.

    Combines two effects:

    * **Openness / shelter** – multi-scale TPI: ridges and exposed spurs sit
      above their surroundings and catch more wind; valleys are sheltered.
    * **Windward aspect** – slopes that face into the prevailing wind are more
      exposed than lee slopes of equal steepness.

    ``prevailing_wind_deg`` is the compass direction the wind blows *from*
    (270 = westerly, the default for the US Mountain West).
    """
    tpi = _multiscale_tpi(dem, dx, dy, radii_m)
    openness = _normalise(tpi)  # 0 = deep valley, 1 = high ridge

    wind_from = math.radians(prevailing_wind_deg)
    # A slope whose aspect points toward the wind source faces into the wind.
    windward = 0.5 * (1.0 + np.cos(aspect_rad - wind_from)) * np.sin(slope_rad)
    windward = _normalise(windward)

    exposure = 0.7 * openness + 0.3 * windward
    return _normalise(exposure)


# ─── Water retention (Topographic Wetness Index) ──────────────────────────────

def _d8_flow_accumulation(dem, dx, dy, max_cells=4_000_000):
    """D8 flow accumulation (number of cells draining through each cell).

    Every cell drains to its single steepest-descent neighbour.  Accumulation
    is computed in one descending-elevation pass so each cell's count reaches
    all of its downslope receivers.  Very large grids are coarsened first (and
    the result upsampled) so the pure-Python routing pass stays tractable.
    """
    dem = np.asarray(dem, dtype="float64")
    rows, cols = dem.shape

    # Coarsen oversized DEMs for the routing step only.
    step = 1
    if rows * cols > max_cells:
        step = int(math.ceil(math.sqrt(rows * cols / max_cells)))
        dem_c = dem[::step, ::step]
    else:
        dem_c = dem
    r, c = dem_c.shape
    n = r * c

    # Neighbour offsets and their planar distances.
    neigh = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1), (1, 0), (1, 1)]
    dist = [math.hypot(dx * step * ox, dy * step * oy) for oy, ox in neigh]

    z = dem_c.ravel()
    receiver = np.arange(n)  # default: cell is its own sink
    best_slope = np.zeros(n)

    for k, (oy, ox) in enumerate(neigh):
        shifted = np.full((r, c), np.inf)
        ys = slice(max(0, oy), r + min(0, oy))
        xs = slice(max(0, ox), c + min(0, ox))
        yt = slice(max(0, -oy), r + min(0, -oy))
        xt = slice(max(0, -ox), c + min(0, -ox))
        shifted[yt, xt] = dem_c[ys, xs]
        drop = (dem_c - shifted).ravel() / dist[k]  # positive = downhill
        neighbour_idx = (np.arange(n).reshape(r, c))
        nb = np.full((r, c), -1)
        nb[yt, xt] = neighbour_idx[ys, xs]
        nb = nb.ravel()
        better = (drop > best_slope) & (nb >= 0)
        best_slope = np.where(better, drop, best_slope)
        receiver = np.where(better, nb, receiver)

    # One accumulation pass from highest to lowest cell.
    acc = np.ones(n, dtype="float64")
    order = np.argsort(z)[::-1]
    for idx in order:
        rcv = receiver[idx]
        if rcv != idx:
            acc[rcv] += acc[idx]

    acc = acc.reshape(r, c)
    if step > 1:  # upsample back to the native grid
        acc = np.repeat(np.repeat(acc, step, axis=0), step, axis=1)[:rows, :cols]
    return acc


def water_retention_index(dem, slope_rad, dx, dy, max_cells=4_000_000):
    """Topographic Wetness Index, normalised to 0..1 (higher = wetter).

    TWI = ln(a / tan(slope)) where ``a`` is the specific catchment area
    (upslope contributing area per unit contour width).  Flat, converging,
    valley-bottom terrain retains water (high index); steep ridges shed it.
    """
    acc = _d8_flow_accumulation(dem, dx, dy, max_cells=max_cells)
    cell = math.sqrt(dx * dy)
    specific_area = acc * cell  # a = (acc * cell^2) / cell

    tan_slope = np.tan(slope_rad)
    tan_slope = np.maximum(tan_slope, 0.001)  # avoid divide-by-zero on flats

    twi = np.log(specific_area / tan_slope)
    return _normalise(twi)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _normalise(arr):
    """Min-max scale a finite array to 0..1 (constant arrays map to 0.5)."""
    arr = np.asarray(arr, dtype="float64")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr)
    lo = arr[finite].min()
    hi = arr[finite].max()
    if hi - lo < 1e-12:
        out = np.full_like(arr, 0.5)
    else:
        out = (arr - lo) / (hi - lo)
    return np.where(finite, out, np.nan)


# ─── Raster I/O entry points ──────────────────────────────────────────────────

def _write_raster(path, data, profile):
    profile = profile.copy()
    profile.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)

    try:
        from compress_rasters import convert_raster_to_cog
        converted = convert_raster_to_cog(path, delete_original=True, verify=True)
        if converted and os.path.exists(converted):
            print(f"   ↳ compressed to {converted}")
    except Exception as exc:
        print(f"   [!] Raster compression failed for {path}: {exc}")


def process_dem(dem_path, out_dir="dem/derived/", prevailing_wind_deg=270.0):
    """Read a DEM GeoTIFF, derive every terrain layer, write them as GeoTIFFs.

    Returns a dict mapping layer name -> output path.  Layers written:
    ``slope``, ``aspect``, ``solar_exposure``, ``wind_exposure``,
    ``water_retention``.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is required to process DEM files")

    os.makedirs(out_dir, exist_ok=True)
    print(f"🏔  Processing DEM {dem_path} ...")

    # Extract bounding box from the input filename (e.g., N46.3_N37.0_W124.6_W102.0)
    match = re.search(r'([NS]\d+\.\d+_[NS]\d+\.\d+_[EW]\d+\.\d+_[EW]\d+\.\d+)', os.path.basename(dem_path))
    box_suffix = f"_{match.group(1)}" if match else ""

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype("float64")
        profile = src.profile
        transform = src.transform
        height = src.height
        # Reproject the raster centre to lon/lat to get a representative latitude.
        cx = transform.c + transform.a * (src.width / 2)
        cy = transform.f + transform.e * (src.height / 2)
        if src.crs and src.crs.to_epsg() != 4326:
            lons, lats = warp_transform(src.crs, "EPSG:4326", [cx], [cy])
            mean_lat = lats[0]
        else:
            mean_lat = cy
        nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    # Fill nodata gaps with the scene mean so gradients stay finite.
    if np.isnan(dem).any():
        dem = np.where(np.isnan(dem), np.nanmean(dem), dem)

    dx, dy = pixel_size_metres(transform, height, mean_lat)
    print(f"   pixel size ≈ {dx:.1f} m × {dy:.1f} m at lat {mean_lat:.3f}")

    slope_rad, aspect_rad = compute_slope_aspect(dem, dx, dy)
    solar = solar_exposure_index(slope_rad, aspect_rad, mean_lat)
    wind = wind_exposure_index(dem, slope_rad, aspect_rad, dx, dy,
                               prevailing_wind_deg=prevailing_wind_deg)
    water = water_retention_index(dem, slope_rad, dx, dy)

    layers = {
        "slope": np.degrees(slope_rad),
        "aspect": np.degrees(aspect_rad),
        "solar_exposure": solar,
        "wind_exposure": wind,
        "water_retention": water,
    }

    paths = {}
    for name, data in layers.items():
        out_path = os.path.join(out_dir, f"{name}{box_suffix}.tif")
        _write_raster(out_path, data, profile)
        paths[name] = out_path
        print(f"   ✓ wrote {out_path}")

    print("✅ Terrain layers ready.")
    return paths


def _find_dem(dem_dir="dem/"):
    dem_dir = str(dem_dir)
    if os.path.isdir(dem_dir):
        tifs = sorted(f for f in os.listdir(dem_dir) if f.lower().endswith((".tif", ".tiff")))
        if tifs:
            return os.path.join(dem_dir, tifs[0])
    if os.path.isdir(os.path.join('.', 'dem')):
        tifs = sorted(f for f in os.listdir(os.path.join('.', 'dem')) if f.lower().endswith((".tif", ".tiff")))
        if tifs:
            return os.path.join('.', 'dem', tifs[0])
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Derive terrain exposure layers from a DEM")
    parser.add_argument("--dem", default=None, help="Path to a DEM GeoTIFF (default: first file in dem/)")
    parser.add_argument("--out", default="dem/derived/", help="Output directory for derived layers")
    parser.add_argument("--wind-dir", type=float, default=270.0,
                        help="Prevailing wind direction in degrees (direction wind comes FROM)")
    args = parser.parse_args()

    dem_path = args.dem or _find_dem()
    if not dem_path or not os.path.exists(dem_path):
        try:
            from fetch import download_srtm_dem
            downloaded = download_srtm_dem()
            if downloaded and os.path.exists(downloaded):
                dem_path = downloaded
        except Exception as exc:
            print(f"[!] DEM download unavailable: {exc}")

    if not dem_path or not os.path.exists(dem_path):
        # Skip gracefully (exit 0) so the full pipeline still completes without
        # terrain layers, rather than aborting every later stage. Terrain
        # enrichment is optional; the map just won't have solar/wind/water.
        print("[!] No DEM found — skipping terrain layers. "
              "Set OPENTOPOGRAPHY_API_KEY (free at https://portal.opentopography.org/login) "
              "or pass --dem to enable them.")
        sys.exit(0)

    process_dem(dem_path, out_dir=args.out, prevailing_wind_deg=args.wind_dir)
