"""Earth Engine point sampling for every environmental column.

The original pipeline downloaded bulk rasters from four different servers and
sampled them locally: ERA5-Land netCDFs from the CDS queue, a CHIRPS GeoTIFF per
day, multi-gigabyte ESA WorldCover tiles, and an SRTM DEM from OpenTopography
that then had to be run through ``terrain_pipeline``. All of those datasets are
already hosted in Earth Engine, so this module asks EE for the values *at the
observation points* instead — no bulk download, no local raster cache, no API
keys beyond an EE project.

The sampling pattern throughout is the one proven by the NDVI stage: group the
points that share a query (a date, or "all of them" for a static layer), build
ONE image carrying every band that group needs, and issue ONE ``reduceRegions``
per chunk of points. A 7-day weather history therefore costs one round trip per
observation *date*, not seven downloads or — as with the old Open-Meteo
temperature stage — one HTTP request per observation.

What comes from where:

    ndvi                     COPERNICUS/S2_SR_HARMONIZED   (unchanged)
    soil_moisture            ECMWF/ERA5_LAND/DAILY_AGGR
    prcp_d0..d6              UCSB-CHG/CHIRPS/DAILY
    tmax_d0..d6, tmin_d0..d6 ECMWF/ERA5_LAND/DAILY_AGGR
    land_cover               ESA/WorldCover
    elevation, slope, aspect USGS/SRTMGL1_003
    solar/wind exposure,     derived from the sampled terrain with the same
    water_retention          formulas terrain_pipeline uses on rasters

Every stage resumes: it only samples rows whose column is still empty, so an
interrupted run picks up where it stopped, and a stage whose data is already
present costs nothing.
"""

import concurrent.futures
import math
import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd

# The derived-index formulas live with the raster pipeline; sampling from Earth
# Engine changes where the inputs come from, not the physics, so they are reused
# rather than reimplemented.
from terrain_pipeline import (
    _normalise,
    solar_exposure_index,
)

# ─── Dataset ids and native resolutions ───────────────────────────────────────
SRTM = 'USGS/SRTMGL1_003'
MERIT_HYDRO = 'MERIT/Hydro/v1_0_1'
WORLDCOVER = 'ESA/WorldCover/v200'
ERA5_DAILY = 'ECMWF/ERA5_LAND/DAILY_AGGR'
CHIRPS_DAILY = 'UCSB-CHG/CHIRPS/DAILY'
S2_SR = 'COPERNICUS/S2_SR_HARMONIZED'

SCALE_SRTM = 30
SCALE_MERIT = 90
SCALE_WORLDCOVER = 10
SCALE_ERA5 = 11132
SCALE_CHIRPS = 5566

# reduceRegions payload bound — points per request within a single query group.
CHUNK_SIZE = 500

_ee = None


def _log(msg):
    print(f"  {msg}", flush=True)


def _progress(label, i, total, start):
    """Throttled [i/total] progress line, matching the raster stages' output."""
    stride = max(1, total // 40)
    if i != total and i % stride:
        return
    pct = i / total * 100 if total else 100.0
    elapsed = time.monotonic() - start
    eta = (elapsed / i) * (total - i) if i else 0.0
    print(f"  [{i}/{total}] {pct:5.1f}% {label} · {elapsed:4.0f}s elapsed, ~{eta:4.0f}s left", flush=True)


def init_ee():
    """Initialise Earth Engine once, returning the module or None if unavailable.

    Never raises: a missing credential or a blocked network degrades the run to
    the raster fallback rather than aborting enrichment.
    """
    global _ee
    if _ee is not None:
        return _ee
    if os.environ.get('SKIP_EARTH_ENGINE') == '1':
        return None
    try:
        import ee
        project = os.environ.get('EARTHENGINE_PROJECT')
        try:
            ee.Initialize(project=project)
        except Exception:
            ee.Authenticate(quiet=True)
            ee.Initialize(project=project)
        _ee = ee
        return ee
    except Exception as exc:
        print(f"[!] Earth Engine unavailable — falling back to cached rasters: {exc}")
        return None


# ─── Core sampling primitive ──────────────────────────────────────────────────

def _sample_points(ee, image, points, scale, reducer=None):
    """Sample every band of ``image`` at ``points`` → {position: {band: value}}.

    ``points`` is a list of ``(position, lon, lat)``. The position is an opaque
    caller-side key (not a DataFrame label), so callers stay free of assumptions
    about the frame's index type. Points are chunked to bound request size.
    """
    reducer = reducer or ee.Reducer.first()
    out = {}
    for c in range(0, len(points), CHUNK_SIZE):
        chunk = points[c:c + CHUNK_SIZE]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([lon, lat]), {'pidx': int(pos)})
            for pos, lon, lat in chunk
        ])
        reduced = image.reduceRegions(collection=fc, reducer=reducer, scale=scale).getInfo()
        for feat in reduced.get('features', []):
            props = dict(feat.get('properties', {}))
            pos = props.pop('pidx', None)
            if pos is not None:
                out[pos] = props
    return out


def _daily_band(ee, collection, band, day, out_name):
    """One calendar day of a daily collection as a single named band.

    A day with no image yields a fully-masked band rather than an error, so one
    gap in a 7-day window leaves that day null instead of failing the whole
    request (and never silently reads as zero rainfall).
    """
    start = ee.Date(day.strftime('%Y-%m-%d'))
    filtered = collection.filterDate(start, start.advance(1, 'day')).select([band])
    return ee.Image(
        ee.Algorithms.If(filtered.size().gt(0),
                         filtered.first(),
                         ee.Image.constant(0).selfMask())
    ).rename([out_name])


def _pending_rows(df, column):
    """Rows with usable coordinates whose ``column`` is still empty (resume)."""
    if column not in df.columns:
        df[column] = None
    return [
        (idx, float(row['lon']), float(row['lat']))
        for idx, row in df.iterrows()
        if not (pd.isna(row.get('lat')) or pd.isna(row.get('lon')))
        and pd.isna(row.get(column))
    ]


def _pending_dated_rows(df, column):
    """As ``_pending_rows`` but also requiring a date, grouped by that date."""
    if column not in df.columns:
        df[column] = None
    by_date = {}
    total = 0
    for idx, row in df.iterrows():
        if pd.isna(row.get('lat')) or pd.isna(row.get('lon')) or pd.isna(row.get('date')):
            continue
        if not pd.isna(row.get(column)):
            continue
        day = pd.Timestamp(row['date']).normalize()
        by_date.setdefault(day, []).append((idx, float(row['lon']), float(row['lat'])))
        total += 1
    return by_date, total


def _run_date_batches(by_date, worker, label, max_workers, checkpoint=None):
    """Run one worker per observation date concurrently, collecting results.

    Returns ``(results, failed_dates)`` where results is a list of
    ``(row_index, column, value)`` triples applied by the caller.
    """
    results = []
    failed = 0
    t0 = time.monotonic()
    total = len(by_date)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, item) for item in by_date.items()]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            got, err = fut.result()
            if err:
                failed += 1
            results.extend(got)
            _progress(f"{label} ({len(results)} values)", i, total, t0)
            if checkpoint:
                checkpoint()
    return results, failed


# ─── Terrain (static) ─────────────────────────────────────────────────────────

def _tpi_image(ee, dem, radius_m):
    """Topographic Position Index at one neighbourhood radius.

    The DEM is reprojected so the circular kernel spans roughly eight pixels
    regardless of radius. Without that, a 1.5 km kernel over 30 m SRTM would
    cover ~8,000 pixels per output cell and dominate the request's cost.
    """
    scale = max(SCALE_SRTM, radius_m / 8.0)
    d = dem.resample('bilinear').reproject(crs='EPSG:3857', scale=scale)
    local_mean = d.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.circle(radius_m, 'meters'),
    )
    return d.subtract(local_mean)


def enrich_terrain_ee(df, radii_m=(150, 500, 1500), prevailing_wind_deg=270.0,
                      checkpoint=None):
    """Elevation, slope, aspect and the three exposure indices, from EE.

    Replaces the OpenTopography DEM download plus the local ``terrain_pipeline``
    run. Elevation, slope, aspect and the two neighbourhood quantities (a
    multi-scale TPI, and MERIT Hydro's upstream drainage area) are sampled at the
    points; the exposure indices are then derived with terrain_pipeline's own
    formulas.

    One deliberate difference from the raster path: those indices end in a
    min-max normalisation, which here spans the observation points rather than
    the whole DEM scene. The indices stay comparable *between observations* —
    which is what the map, charts and clustering use them for — but an absolute
    value is not comparable across two runs over different point sets.
    """
    ee = init_ee()
    if ee is None:
        return df

    for col in ('elevation', 'slope', 'aspect', 'solar_exposure',
                'wind_exposure', 'water_retention'):
        if col not in df.columns:
            df[col] = None

    # 'slope' gates the stage: it is the first column this stage owns outright
    # (elevation may already have arrived from the iNaturalist fetch).
    pending = _pending_rows(df, 'slope')
    if not pending:
        print("Terrain already complete — skipping.")
        return df
    print(f"Sampling terrain from Earth Engine ({SRTM} + {MERIT_HYDRO}) "
          f"— {len(pending)} points...")

    dem = ee.Image(SRTM).select('elevation')
    terrain = ee.Terrain.products(dem)  # slope + aspect in degrees

    tpi = None
    for r in radii_m:
        band = _tpi_image(ee, dem, r)
        tpi = band if tpi is None else tpi.add(band)
    tpi = tpi.divide(len(radii_m)).rename('tpi')

    # MERIT Hydro's upstream drainage area (km²) stands in for the local D8 flow
    # accumulation — a hydrologically conditioned global product, and far better
    # conditioned than routing a raw SRTM tile ourselves.
    upa = ee.Image(MERIT_HYDRO).select('upa').rename('upa')

    image = (terrain.select(['elevation', 'slope', 'aspect'])
             .addBands(tpi)
             .addBands(upa))

    points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pending)]
    t0 = time.monotonic()
    sampled = _sample_points(ee, image, points, SCALE_SRTM)
    _progress("terrain", len(points), len(points), t0)

    idx_of = [idx for idx, _lon, _lat in pending]
    lats = np.array([lat for _idx, _lon, lat in pending], dtype='float64')

    def column(name):
        return np.array([sampled.get(pos, {}).get(name, np.nan) for pos in range(len(points))],
                        dtype='float64')

    elevation = column('elevation')
    slope_deg = column('slope')
    aspect_deg = column('aspect')
    tpi_vals = column('tpi')
    upa_km2 = column('upa')

    slope_rad = np.radians(slope_deg)
    aspect_rad = np.radians(aspect_deg)

    # Solar exposure is pointwise, so terrain_pipeline's model applies directly.
    # It takes a single scene latitude; the observations span a narrow band, so
    # their mean latitude is used for the sun-position geometry.
    mean_lat = float(np.nanmean(lats)) if np.isfinite(lats).any() else 0.0
    solar = solar_exposure_index(slope_rad, aspect_rad, mean_lat)

    # Wind exposure: the same 0.7 openness + 0.3 windward blend, with openness
    # from the sampled multi-scale TPI.
    openness = _normalise(tpi_vals)
    wind_from = math.radians(prevailing_wind_deg)
    windward = _normalise(0.5 * (1.0 + np.cos(aspect_rad - wind_from)) * np.sin(slope_rad))
    wind = _normalise(0.7 * openness + 0.3 * windward)

    # TWI = ln(a / tan(slope)); a = upslope area per unit contour width. MERIT is
    # 3 arc-second (~90 m), so a = upa_m2 / cell_width.
    specific_area = np.where(np.isfinite(upa_km2), upa_km2 * 1e6 / SCALE_MERIT, np.nan)
    tan_slope = np.maximum(np.tan(slope_rad), 0.001)
    with np.errstate(divide='ignore', invalid='ignore'):
        twi = np.log(specific_area / tan_slope)
    water = _normalise(twi)

    filled = 0
    for pos, idx in enumerate(idx_of):
        if not np.isfinite(slope_deg[pos]):
            continue
        if np.isfinite(elevation[pos]) and pd.isna(df.at[idx, 'elevation']):
            df.at[idx, 'elevation'] = float(elevation[pos])
        df.at[idx, 'slope'] = float(slope_deg[pos])
        df.at[idx, 'aspect'] = float(aspect_deg[pos])
        for col, arr in (('solar_exposure', solar), ('wind_exposure', wind),
                         ('water_retention', water)):
            if np.isfinite(arr[pos]):
                df.at[idx, col] = float(arr[pos])
        filled += 1

    if checkpoint:
        checkpoint()
    print(f"✅ Terrain sampled for {filled}/{len(pending)} points.")
    return df


# ─── Land cover (static) ──────────────────────────────────────────────────────

def enrich_landcover_ee(df, checkpoint=None):
    """ESA WorldCover class code at each point, replacing the 3°×3° tile downloads.

    Uses the ``first`` reducer: the values are class codes, so averaging pixels
    would invent classes that do not exist.
    """
    ee = init_ee()
    if ee is None:
        return df

    pending = _pending_rows(df, 'land_cover')
    if not pending:
        print("Land cover already complete — skipping.")
        return df
    print(f"Sampling {WORLDCOVER} land cover from Earth Engine — {len(pending)} points...")

    image = ee.ImageCollection(WORLDCOVER).first().select('Map').rename('land_cover')
    points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pending)]

    t0 = time.monotonic()
    sampled = _sample_points(ee, image, points, SCALE_WORLDCOVER)
    _progress("land cover", len(points), len(points), t0)

    filled = 0
    for pos, (idx, _lon, _lat) in enumerate(pending):
        val = sampled.get(pos, {}).get('land_cover')
        if val is not None:
            df.at[idx, 'land_cover'] = val
            filled += 1

    if checkpoint:
        checkpoint()
    print(f"✅ Land cover sampled for {filled}/{len(pending)} points.")
    return df


# ─── Soil moisture (per observation date) ─────────────────────────────────────

def enrich_soil_moisture_ee(df, max_workers=8, checkpoint=None):
    """ERA5-Land volumetric soil water (layer 1, 0–7 cm) on the observation day.

    Replaces the CDS API stage, which queued one netCDF request per date and was
    reliably the slowest thing in the pipeline.
    """
    ee = init_ee()
    if ee is None:
        return df

    by_date, total = _pending_dated_rows(df, 'soil_moisture')
    if not total:
        print("Soil moisture already complete — skipping.")
        return df
    print(f"Sampling ERA5-Land soil moisture from Earth Engine — {total} points "
          f"across {len(by_date)} date(s), {max_workers} parallel...")

    era5 = ee.ImageCollection(ERA5_DAILY)

    def worker(item):
        day, pts = item
        try:
            image = _daily_band(ee, era5, 'volumetric_soil_water_layer_1', day, 'soil_moisture')
            points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pts)]
            sampled = _sample_points(ee, image, points, SCALE_ERA5, reducer=ee.Reducer.mean())
            out = []
            for pos, (idx, _lon, _lat) in enumerate(pts):
                val = sampled.get(pos, {}).get('soil_moisture')
                if val is not None:
                    out.append((idx, 'soil_moisture', val))
            return out, None
        except Exception as exc:
            return [], str(exc)

    results, failed = _run_date_batches(by_date, worker, "soil moisture", max_workers, checkpoint)
    for idx, col, val in results:
        df.at[idx, col] = val

    if failed:
        print(f"[!] {failed}/{len(by_date)} soil-moisture date batch(es) failed.")
    print(f"✅ Soil moisture sampled for {len(results)}/{total} points.")
    return df


# ─── Wind (per observation date) ──────────────────────────────────────────────

def enrich_wind_ee(df, max_workers=8, checkpoint=None):
    """ERA5-Land 10 m wind as its eastward/northward components.

    Stored as vector components rather than a speed and a bearing on purpose:
    directions are circular, so `wind_u`/`wind_v` can be averaged over a map
    cell by summing, while averaging bearings would put the mean of 350° and 10°
    at 180° — pointing exactly the wrong way. The map's wind overlay reads these
    and falls back to terrain aspect where they are missing.
    """
    ee = init_ee()
    if ee is None:
        return df

    for col in ('wind_u', 'wind_v'):
        if col not in df.columns:
            df[col] = None

    by_date, total = _pending_dated_rows(df, 'wind_u')
    if not total:
        print("Wind already complete — skipping.")
        return df
    print(f"Sampling ERA5-Land 10 m wind from Earth Engine — {total} points "
          f"across {len(by_date)} date(s), {max_workers} parallel...")

    era5 = ee.ImageCollection(ERA5_DAILY)

    def worker(item):
        day, pts = item
        try:
            image = ee.Image.cat([
                _daily_band(ee, era5, 'u_component_of_wind_10m', day, 'wind_u'),
                _daily_band(ee, era5, 'v_component_of_wind_10m', day, 'wind_v'),
            ])
            points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pts)]
            sampled = _sample_points(ee, image, points, SCALE_ERA5, reducer=ee.Reducer.mean())
            out = []
            for pos, (idx, _lon, _lat) in enumerate(pts):
                props = sampled.get(pos, {})
                for col in ('wind_u', 'wind_v'):
                    val = props.get(col)
                    if val is not None:
                        out.append((idx, col, val))
            return out, None
        except Exception as exc:
            return [], str(exc)

    results, failed = _run_date_batches(by_date, worker, "wind", max_workers, checkpoint)
    for idx, col, val in results:
        df.at[idx, col] = val

    if failed:
        print(f"[!] {failed}/{len(by_date)} wind date batch(es) failed.")
    print(f"✅ Wind sampled for {total} points ({len(results)} values).")
    return df


# ─── Precipitation history (per observation date) ─────────────────────────────

def enrich_precip_ee(df, days=7, max_workers=8, checkpoint=None):
    """CHIRPS daily rainfall for the ``days`` up to each observation.

    All seven days become bands of one image, so an observation date costs a
    single request instead of seven GeoTIFF downloads.
    """
    ee = init_ee()
    if ee is None:
        return df

    for d in range(days):
        if f'prcp_d{d}' not in df.columns:
            df[f'prcp_d{d}'] = None

    by_date, total = _pending_dated_rows(df, 'prcp_d0')
    if not total:
        print("Precipitation history already complete — skipping.")
        return df
    print(f"Sampling CHIRPS {days}-day rainfall from Earth Engine — {total} points "
          f"across {len(by_date)} date(s), {max_workers} parallel...")

    chirps = ee.ImageCollection(CHIRPS_DAILY)

    def worker(item):
        day, pts = item
        try:
            image = ee.Image.cat([
                _daily_band(ee, chirps, 'precipitation', day - timedelta(days=d), f'prcp_d{d}')
                for d in range(days)
            ])
            points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pts)]
            sampled = _sample_points(ee, image, points, SCALE_CHIRPS, reducer=ee.Reducer.mean())
            out = []
            for pos, (idx, _lon, _lat) in enumerate(pts):
                props = sampled.get(pos, {})
                for d in range(days):
                    val = props.get(f'prcp_d{d}')
                    if val is not None:
                        out.append((idx, f'prcp_d{d}', val))
            return out, None
        except Exception as exc:
            return [], str(exc)

    results, failed = _run_date_batches(by_date, worker, "precipitation", max_workers, checkpoint)
    for idx, col, val in results:
        df.at[idx, col] = val

    if failed:
        print(f"[!] {failed}/{len(by_date)} precipitation date batch(es) failed.")
    print(f"✅ Precipitation history sampled for {total} points ({len(results)} values).")
    return df


# ─── Temperature history (per observation date) ───────────────────────────────

def enrich_temperature_ee(df, days=7, max_workers=8, checkpoint=None):
    """ERA5-Land daily max/min air temperature for the ``days`` up to each observation.

    Replaces the Open-Meteo stage, which issued one HTTP request *per
    observation* — thousands of round trips for a few thousand finds. Here the
    cost is one request per observation date.

    ERA5 reports kelvin; the column contract (and the UI's °C/°F conversion) is
    celsius, so values are converted on the way in.
    """
    ee = init_ee()
    if ee is None:
        return df

    for d in range(days):
        for col in (f'tmax_d{d}', f'tmin_d{d}'):
            if col not in df.columns:
                df[col] = None

    by_date, total = _pending_dated_rows(df, 'tmax_d0')
    if not total:
        print("Temperature history already complete — skipping.")
        return df
    print(f"Sampling ERA5-Land {days}-day temperature from Earth Engine — {total} points "
          f"across {len(by_date)} date(s), {max_workers} parallel...")

    era5 = ee.ImageCollection(ERA5_DAILY)

    def worker(item):
        day, pts = item
        try:
            bands = []
            for d in range(days):
                target = day - timedelta(days=d)
                bands.append(_daily_band(ee, era5, 'temperature_2m_max', target, f'tmax_d{d}'))
                bands.append(_daily_band(ee, era5, 'temperature_2m_min', target, f'tmin_d{d}'))
            image = ee.Image.cat(bands)
            points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pts)]
            sampled = _sample_points(ee, image, points, SCALE_ERA5, reducer=ee.Reducer.mean())
            out = []
            for pos, (idx, _lon, _lat) in enumerate(pts):
                props = sampled.get(pos, {})
                for d in range(days):
                    for col in (f'tmax_d{d}', f'tmin_d{d}'):
                        val = props.get(col)
                        if val is not None:
                            out.append((idx, col, val - 273.15))  # kelvin → celsius
            return out, None
        except Exception as exc:
            return [], str(exc)

    results, failed = _run_date_batches(by_date, worker, "temperature", max_workers, checkpoint)
    for idx, col, val in results:
        df.at[idx, col] = val

    if failed:
        print(f"[!] {failed}/{len(by_date)} temperature date batch(es) failed.")
    print(f"✅ Temperature history sampled for {total} points ({len(results)} values).")
    return df


# ─── NDVI (per observation date) ──────────────────────────────────────────────

def enrich_ndvi_ee(df, buffer_days=15, scale=10, cloud_pct=60, max_workers=8, checkpoint=None):
    """Sentinel-2 NDVI at each point, from a ±``buffer_days`` median composite.

    Points sharing a date share the window, so one composite and one
    reduceRegions serve the whole group.
    """
    ee = init_ee()
    if ee is None:
        return df

    by_date, total = _pending_dated_rows(df, 'ndvi')
    if not total:
        print("NDVI already complete — skipping.")
        return df
    print(f"Sampling Sentinel-2 NDVI (±{buffer_days}d, {scale} m) from Earth Engine — "
          f"{total} points across {len(by_date)} date(s), {max_workers} parallel...")

    def worker(item):
        day, pts = item
        try:
            start = (day - timedelta(days=buffer_days)).strftime('%Y-%m-%d')
            end = (day + timedelta(days=buffer_days)).strftime('%Y-%m-%d')
            lons = [p[1] for p in pts]
            lats = [p[2] for p in pts]
            region = ee.Geometry.Rectangle([min(lons) - 0.05, min(lats) - 0.05,
                                            max(lons) + 0.05, max(lats) + 0.05])
            image = (ee.ImageCollection(S2_SR)
                     .filterDate(start, end)
                     .filterBounds(region)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
                     .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('ndvi'))
                     .median())
            points = [(pos, lon, lat) for pos, (_idx, lon, lat) in enumerate(pts)]
            sampled = _sample_points(ee, image, points, scale, reducer=ee.Reducer.mean())
            out = []
            for pos, (idx, _lon, _lat) in enumerate(pts):
                val = sampled.get(pos, {}).get('ndvi')
                if val is not None:
                    out.append((idx, 'ndvi', val))
            return out, None
        except Exception as exc:
            return [], str(exc)

    results, failed = _run_date_batches(by_date, worker, "NDVI", max_workers, checkpoint)
    for idx, col, val in results:
        df.at[idx, col] = val

    if failed:
        print(f"[!] {failed}/{len(by_date)} NDVI date batch(es) failed.")
    print(f"✅ NDVI sampled for {len(results)}/{total} points.")
    return df


# ─── Orchestration ────────────────────────────────────────────────────────────

# (label, function) in the order the enrichment runs them.
EE_STAGES = [
    ("Terrain (Earth Engine)", enrich_terrain_ee),
    ("Land cover (Earth Engine)", enrich_landcover_ee),
    ("Soil moisture (Earth Engine)", enrich_soil_moisture_ee),
    ("Wind (Earth Engine)", enrich_wind_ee),
    ("Precipitation history (Earth Engine)", enrich_precip_ee),
    ("Temperature history (Earth Engine)", enrich_temperature_ee),
    ("NDVI (Earth Engine)", enrich_ndvi_ee),
]


def earth_engine_enabled():
    """True when EE sampling should be attempted for this run.

    Off when ``SKIP_EARTH_ENGINE=1`` or ``USE_EARTH_ENGINE=0``; otherwise on, so
    a configured project gets the fast path without extra opt-in.
    """
    if os.environ.get('SKIP_EARTH_ENGINE') == '1':
        return False
    return os.environ.get('USE_EARTH_ENGINE', '1').strip().lower() not in {'0', 'false', 'no', 'off'}
