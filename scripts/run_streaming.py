"""A streaming (parallel) variant of the pipeline: fetch and enrich at the same time.

The default pipeline runs the stages one after another — the whole iNaturalist
fetch, then the whole enrichment, then clustering and export. On a large run the
fetch is bound by iNaturalist's rate limit and the enrichment is bound by Earth
Engine, two *different* services, so running them back to back leaves each idle
while the other works.

``run_streaming`` overlaps them with a small producer/consumer setup:

    producer thread   → fetches one location at a time, writes each species to
                        the store, and drops the location's rows on a queue
        │
        ▼  queue of (label, DataFrame)
    enrichment workers → pull a location's rows and enrich them (Earth Engine
                        first, cached rasters as fallback), write the enriched
                        store, and grow the map's GeoJSON

    then, once every location is enriched:  cluster  →  export (global, once)

The unit that flows through the queue is one **location** (a configured
INAT_LOCATIONS / plus-code entry), not one species — so all of a location's
species still enrich together and share one Earth Engine request per date, the
batching that keeps enrichment cheap. Enrichment of the last location overlaps
the fetch of the next.

Everything is resumable exactly like the sequential pipeline: the fetch skips
observation ids already on disk, and each enrichment stage only fills rows still
missing its column, so a re-run continues rather than restarting.

Tunables (env):
    STREAM_ENRICH_WORKERS   how many locations to enrich concurrently (default 2).
                            Each worker issues its own Earth Engine requests, so
                            raise it only if EE is keeping up; lower to 1 if you
                            see EE quota errors.
    EXPORT_EACH_AREA=0      turn off the grow-the-GeoJSON-as-you-go export.
    plus every variable the normal fetch/enrichment read (SPECIES, INAT_LOCATIONS,
    REFRESH_ALL, USE_EARTH_ENGINE, AREA_BIN_PLUS_LENGTH, …).
"""

import os
import queue
import sys
import threading
import time

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
for _p in (SCRIPTS_DIR, ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd

import iNat
import enrich_with_rasters as enrich
import species_store as store
import run_pipeline


_SENTINEL = ('__done__', None)


def _stream_enrich_workers():
    try:
        n = int(os.getenv('STREAM_ENRICH_WORKERS', '2'))
    except (TypeError, ValueError):
        n = 2
    return max(1, n)


def _location_label(loc):
    if 'nelat' in loc:
        return (f"box {loc.get('label', '')} "
                f"[{loc['swlat']:.3f},{loc['swlng']:.3f}→{loc['nelat']:.3f},{loc['nelng']:.3f}]")
    return f"{loc['lat']},{loc['lng']} r{loc['radius']}km"


def _resolve_locations(default_radius):
    locations = iNat._parse_locations_with_radius(os.getenv('INAT_LOCATIONS'), default_radius)
    locations.extend(iNat.parse_plus_codes(os.getenv('INAT_PLUS_CODES'), default_radius))
    locations.extend(iNat.parse_plus_code_ranges(
        os.getenv('INAT_PLUS_CODE_RANGES') or os.getenv('INAT_PLUS_CODES_RANGES')))
    if not locations:
        lat, lng = iNat._resolve_location_from_env(default_lat=40.0, default_lng=-105.0)
        locations = [{'lat': lat, 'lng': lng, 'radius': default_radius}]
    return locations


def _fetch_one_location(loc, species_list, quality_grade, per_page, max_observations,
                        existing_ids, refresh_all, group_by):
    """Fetch every species for one location, write each to the species store as it
    lands, and return the location's combined DataFrame (or None if it added
    nothing)."""
    bounds = None
    lat = loc.get('lat')
    lng = loc.get('lng')
    radius = loc.get('radius')
    if 'nelat' in loc:
        bounds = (loc['swlat'], loc['swlng'], loc['nelat'], loc['nelng'])

    frames = []
    for species_name in species_list:
        total = iNat.get_species_observation_total(
            taxon_name=species_name, quality_grade=quality_grade,
            lat=lat, lng=lng, radius=radius, bounds=bounds)
        if max_observations and total > max_observations:
            total = max_observations
        df_species = iNat.fetch_inat_data(
            taxon_name=species_name, quality_grade=quality_grade,
            lat=lat, lng=lng, radius=radius, bounds=bounds,
            per_page=per_page, max_observations=max_observations or None,
            total_count=total, existing_ids=existing_ids)
        if df_species is None or df_species.empty:
            print(f"  [fetch] {species_name}: 0", flush=True)
            continue
        if not refresh_all and 'inat_id' in df_species.columns:
            df_species = df_species[~df_species['inat_id'].astype(str).isin(existing_ids)]
        if df_species.empty:
            print(f"  [fetch] {species_name}: 0 new", flush=True)
            continue
        # Persist immediately so the fetch survives an interruption, and record
        # the ids so a later location doesn't re-emit the same observation.
        store.write_split(df_species, base=store.SPECIES_DIR, key=group_by, merge=True)
        if 'inat_id' in df_species.columns:
            existing_ids.update(df_species['inat_id'].astype(str).tolist())
        print(f"  [fetch] {species_name}: {len(df_species)} new", flush=True)
        frames.append(df_species)

    if not frames:
        return None
    frames = [f for f in frames if f is not None and not f.empty and not f.isna().all(axis=None)]
    return pd.concat(frames, ignore_index=True) if frames else None


def _enrich_location(df, label, group_by, store_lock):
    """Enrich one location's rows and merge them into the enriched store."""
    # Earth Engine first, cached rasters as fallback (the shared value-filling
    # stages), then the whole-frame finishers on this location's slice.
    df = enrich._fill_stages(df)
    try:
        df = enrich.fill_missing_ndvi(df, max_days_gap=7)
    except Exception as exc:  # noqa: BLE001
        print(f"  [enrich {label}] NDVI gap-fill skipped ({exc})", flush=True)
    try:
        df = enrich._postprocess_landcover(df)
    except Exception as exc:  # noqa: BLE001
        print(f"  [enrich {label}] land-cover post-process skipped ({exc})", flush=True)

    with store_lock:
        store.write_split(df, base=store.ENRICHED_DIR, key=group_by, merge=True)
        if enrich._incremental_export_enabled():
            try:
                import export_geojson
                export_geojson.export_all(store.load_all(store.ENRICHED_DIR), group_by=group_by)
                print(f"  🗺️  [enrich {label}] GeoJSON updated.", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"  [enrich {label}] GeoJSON export skipped ({exc})", flush=True)
    return len(df)


def run_streaming(python_executable=None, root=None):
    with run_pipeline.working_directory(root or ROOT_DIR):
        run_pipeline._prepare(python_executable)
        _run(python_executable)


def _run(python_executable=None):
    env_file = os.getenv('ENV_FILE') or '.env'
    species_value = iNat.getenv_with_file(
        'INAT_TAXON_NAME',
        default=iNat.getenv_with_file('SPECIES', default='morchella', env_file=env_file),
        env_file=env_file)
    species_list = iNat.parse_species_list(species_value) or ['morchella']
    quality_grade = iNat.getenv_with_file(
        'INAT_QUALITY_GRADE',
        default=iNat.getenv_with_file('QUALITY_GRADE', default='research', env_file=env_file),
        env_file=env_file)
    default_radius = iNat._read_float_env('INAT_RADIUS', 'RADIUS', default=500.0)
    locations = _resolve_locations(default_radius)
    per_page = iNat.resolve_inat_page_size({**iNat.load_env_file(env_file), **os.environ})
    max_observations = int(iNat.getenv_with_file(
        'INAT_MAX_OBSERVATIONS_PER_SPECIES',
        default=iNat.getenv_with_file('MAX_OBSERVATIONS_PER_SPECIES', default='0', env_file=env_file),
        env_file=env_file) or 0)
    refresh_all = iNat.should_refresh_all()
    group_by = os.getenv('GROUP_BY', 'genus')
    workers = _stream_enrich_workers()

    # Full refresh clears both stores so incremental merges start clean.
    if refresh_all:
        for base in (store.SPECIES_DIR, store.ENRICHED_DIR):
            for path in store.list_species_files(base):
                try:
                    os.remove(path)
                except OSError:
                    pass
        if os.path.exists(store.ENRICHED_DONE):
            try:
                os.remove(store.ENRICHED_DONE)
            except OSError:
                pass

    existing_ids = set()
    if not refresh_all:
        existing = store.load_all(store.SPECIES_DIR)
        if 'inat_id' in existing.columns:
            existing_ids = {str(v) for v in existing['inat_id'].dropna().tolist()}

    print(f"\n=== Streaming pipeline ===\n"
          f"  {len(species_list)} species × {len(locations)} location(s), "
          f"{workers} enrichment worker(s), refresh_all={refresh_all}", flush=True)

    work = queue.Queue()
    store_lock = threading.Lock()
    counters = {'enriched_rows': 0, 'enriched_locations': 0}
    counters_lock = threading.Lock()

    def producer():
        for i, loc in enumerate(locations, 1):
            label = _location_label(loc)
            print(f"\n[fetch {i}/{len(locations)}] {label}", flush=True)
            try:
                df = _fetch_one_location(loc, species_list, quality_grade, per_page,
                                         max_observations, existing_ids, refresh_all, group_by)
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch {i}/{len(locations)}] failed: {exc}", flush=True)
                df = None
            if df is not None and not df.empty:
                work.put((f"{i}/{len(locations)} {label}", df))
        for _ in range(workers):
            work.put(_SENTINEL)

    def consumer(worker_id):
        while True:
            label, df = work.get()
            try:
                if df is None:  # sentinel
                    return
                print(f"\n[enrich w{worker_id}] {label} — {len(df)} point(s)", flush=True)
                n = _enrich_location(df, label, group_by, store_lock)
                with counters_lock:
                    counters['enriched_rows'] += n
                    counters['enriched_locations'] += 1
            except Exception as exc:  # noqa: BLE001 — one bad location must not sink the run
                print(f"[enrich w{worker_id}] {label} failed: {exc}", flush=True)
            finally:
                work.task_done()

    t0 = time.monotonic()
    producer_thread = threading.Thread(target=producer, name='fetch', daemon=True)
    consumer_threads = [threading.Thread(target=consumer, args=(w + 1,), name=f'enrich-{w+1}', daemon=True)
                        for w in range(workers)]
    producer_thread.start()
    for t in consumer_threads:
        t.start()
    producer_thread.join()
    for t in consumer_threads:
        t.join()

    print(f"\n=== Fetch + enrichment done in {time.monotonic() - t0:.0f}s "
          f"({counters['enriched_locations']} location(s), {counters['enriched_rows']} rows) ===", flush=True)

    if store.load_all(store.ENRICHED_DIR).empty:
        print("No enriched observations produced; skipping clustering and export.")
        return

    # Mark enrichment complete, then the two global stages that need all the data.
    try:
        open(store.ENRICHED_DONE, 'w').close()
    except OSError:
        pass
    run_pipeline.run_clustering(python_executable, root='.')
    run_pipeline.run_export(python_executable, root='.')
    run_pipeline.run_coverage(python_executable, root='.')
    run_pipeline.report_precision(root='.')
    print("\n✅ Streaming pipeline completed successfully.")


def main():
    run_streaming()


if __name__ == "__main__":
    main()
