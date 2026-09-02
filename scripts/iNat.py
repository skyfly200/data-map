import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pyinaturalist import get_observations
import pandas as pd
from meteostat import Point, stations, daily
from datetime import datetime
import requests

import sys
from pathlib import Path

# Add repository root (parent of scripts/) to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import species_store as store
import utils.olc as olc_utils

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

_ELEVATION_CACHE = {}
_WEATHER_CACHE = {}

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

def getenv_with_file(key, default=None, *, env_file=None):
    values = load_env_file(env_file)
    if key in values and values[key] not in ('', None):
        return values[key]
    return os.getenv(key, default)

def get_elevation(lat, lon):
    if lat is None or lon is None:
        return None
    try:
        flat = round(float(lat), 4)
        flon = round(float(lon), 4)
    except (ValueError, TypeError):
        return None
    key = (flat, flon)
    if key in _ELEVATION_CACHE:
        return _ELEVATION_CACHE[key]

    url = f"https://api.open-elevation.com/api/v1/lookup?locations={flat},{flon}"
    try:
        r = requests.get(url, timeout=15)
        if r.ok:
            value = r.json()['results'][0]['elevation']
            _ELEVATION_CACHE[key] = value
            return value
    except requests.RequestException:
        pass
    _ELEVATION_CACHE[key] = None
    return None

def get_weather(lat, lon, date_str):
    if not date_str:
        return {'station_id': None}

    key = (round(float(lat), 4), round(float(lon), 4), str(date_str))
    if key in _WEATHER_CACHE:
        return _WEATHER_CACHE[key]

    try:
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        point = Point(lat, lon)
        nearby = stations.nearby(point, radius=50000, limit=5)

        for station_id in nearby.index:
            ts = daily(station_id, date, date)
            df = ts.fetch() if hasattr(ts, 'fetch') else ts
            if df is not None and not df.empty:
                row = df.iloc[0].to_dict()
                row['station_used'] = station_id
                _WEATHER_CACHE[key] = row
                return row
    except Exception:
        pass

    result = {'station_id': None}
    _WEATHER_CACHE[key] = result
    return result

def _read_float_env(*names, default):
    for name in names:
        value = os.getenv(name)
        if value is None or value == '':
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return default

def _resolve_location_from_env(default_lat=40.0, default_lng=-105.0):
    location_value = (
        os.getenv('INAT_LOCATION')
        or os.getenv('LOCATION')
        or os.getenv('INAT_LAT_LNG')
    )
    if location_value:
        match = re.search(r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*[,\s]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', location_value)
        if match:
            return float(match.group(1)), float(match.group(2))

    lat = _read_float_env('INAT_LAT', 'LAT', default=default_lat)
    lng = _read_float_env('INAT_LNG', 'LNG', 'LON', default=default_lng)
    return lat, lng

def _parse_locations_with_radius(env_value: str, default_radius: float) -> list[dict]:
    locations = []
    if not env_value:
        return locations
    for entry in env_value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        try:
            lat = float(parts[0])
            lng = float(parts[1])
            rad = float(parts[2]) if len(parts) > 2 else default_radius
            locations.append({"lat": lat, "lng": lng, "radius": rad})
        except Exception:
            print(f"[!] Invalid location entry ignored: {entry}")
    return locations

def parse_plus_codes(env_value: str, default_radius: float) -> list[dict]:
    locations = []
    if not env_value:
        return locations
    for entry in env_value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(",")
        try:
            plus = parts[0]
            lat, lng = olc_utils.decode_olc(plus)
            rad = float(parts[1]) if len(parts) > 1 else default_radius
            locations.append({"lat": lat, "lng": lng, "radius": rad})
        except Exception as exc:
            print(f"[!] Invalid plus-code entry ignored: {entry} ({exc})")
    return locations

def render_progress_bar(current, total, width=20):
    if total <= 0:
        return '[' + ('=' * width) + '] 0/0'
    filled = max(0, min(width, int(round((current / total) * width))))
    bar = '#' * filled + '-' * (width - filled)
    return f'[{bar}] {current}/{total}'

def format_observation_progress(species, current, total, width=20):
    return f'{species} {render_progress_bar(current, total, width=width)}'

def parse_species_list(species_value):
    if species_value is None:
        return []
    if isinstance(species_value, (list, tuple, set)):
        values = species_value
    else:
        values = str(species_value).split(',')

    cleaned = []
    seen = set()
    for item in values:
        value = str(item).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned

def _coerce_int(value, default, *, minimum=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default

def resolve_inat_page_size(env=None):
    values = {**os.environ, **(env or {})}
    for key in ('INAT_PER_PAGE', 'PER_PAGE'):
        raw = values.get(key)
        if raw is None or str(raw).strip() == '':
            continue
        parsed = _coerce_int(raw, 200)
        if parsed > 0:
            return parsed
    return 200

def get_parallel_fetch_workers(env=None):
    values = {**os.environ, **(env or {})}
    for key in ('INAT_PARALLEL_FETCHES', 'PARALLEL_FETCHES', 'FETCH_WORKERS'):
        raw = values.get(key)
        if raw is None or str(raw).strip() == '':
            continue
        parsed = _coerce_int(raw, 3, minimum=1)
        if parsed > 0:
            return parsed
    return 3

def should_refresh_all(env=None):
    values = {**(env or {})}
    keys = ('REFRESH_ALL', 'INAT_REFRESH_ALL', 'FULL_REFRESH')

    for key in keys:
        raw = values.get(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if value in ('0', 'false', 'no', 'n', 'off', ''):
            return False

    for key in keys:
        raw = os.getenv(key)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if value in ('0', 'false', 'no', 'n', 'off', ''):
            return False
    return False

def _existing_observation_ids(path):
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    ids = []
    for column in ('inat_id', 'id'):
        if column in df.columns:
            ids.extend(df[column].dropna().astype(str).tolist())
    return {str(item) for item in ids}

def filter_new_observations(fresh_rows, existing_rows=None):
    seen = set()
    if isinstance(existing_rows, list):
        seen = {str(item.get('inat_id') if isinstance(item, dict) else item) for item in existing_rows if item is not None}
    elif isinstance(existing_rows, pd.DataFrame):
        if 'inat_id' in existing_rows.columns:
            seen = {str(item) for item in existing_rows['inat_id'].dropna().tolist()}
    if not isinstance(fresh_rows, list):
        return []
    new_rows = []
    for item in fresh_rows:
        if not isinstance(item, dict):
            continue
        inat_id = item.get('inat_id')
        key = str(inat_id) if inat_id is not None else str(item.get('uuid')) if item.get('uuid') is not None else None
        if key is None:
            new_rows.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        new_rows.append(item)
    return new_rows

# Above this, a coordinate is too vague to sample terrain at. A 0.2° obscuring
# cell is ~22km across, so iNaturalist reports tens of kilometres of accuracy for
# an obscured record; a GPS fix under a canopy is tens or hundreds of metres.
# 1km sits well clear of honest GPS error and well below any obscured record.
COARSE_ACCURACY_M = 1000

def _coerce_accuracy(value):
    """Accuracy in metres, or None. iNaturalist leaves it unset more often than not."""
    if value is None:
        return None
    try:
        metres = int(float(value))
    except (TypeError, ValueError):
        return None
    return metres if metres >= 0 else None

def classify_location_precision(obscured, geoprivacy, taxon_geoprivacy, public_accuracy):
    """
    One word for how much the coordinates can be trusted.

    'obscured'  the point is randomised inside a ~20km cell — iNaturalist is
                deliberately not telling us where this was, either because the
                observer asked or because the taxon is threatened.
    'coarse'    an honest but vague location: no obscuring, but an accuracy
                radius too wide to sample terrain at.
    'precise'   good enough to read the ground under it.
    'unknown'   iNaturalist reported no accuracy at all, which is common. Not
                the same as precise, and worth keeping separate so a filter can
                decide what to do about it.

    Obscuring wins over accuracy: a record can be obscured and still carry a
    small accuracy number, and that number describes the observer's GPS, not the
    published point.
    """
    if obscured or geoprivacy == 'obscured' or taxon_geoprivacy == 'obscured':
        return 'obscured'
    if geoprivacy == 'private' or taxon_geoprivacy == 'private':
        return 'obscured'
    if public_accuracy is None:
        return 'unknown'
    return 'coarse' if public_accuracy > COARSE_ACCURACY_M else 'precise'

def get_species_observation_total(taxon_name='morchella', quality_grade='research', lat=40.0, lng=-105.0, radius=500.0):
    try:
        response = get_observations(
            taxon_name=taxon_name,
            lat=lat,
            lng=lng,
            quality_grade=quality_grade,
            radius=radius,
            captive=False,
            geo=True,
            per_page=1,
            page=1,
        )
        if not isinstance(response, dict):
            return 0
        total = response.get('total_results')
        return int(total) if total is not None else 0
    except Exception:
        return 0

def _get_observations_with_retry(max_retries=3, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return get_observations(**kwargs)
        except Exception as err:
            if attempt == max_retries:
                taxon = kwargs.get('taxon_name')
                page = kwargs.get('page', 1)
                print(f"\n[!] iNaturalist query failed for taxon '{taxon}' on page {page}: {err}")
                return None
            time.sleep(1.0 * attempt)
    return None

def fetch_inat_data(taxon_name='morchella', quality_grade='research', lat=40.0, lng=-105.0, radius=500.0, per_page=200, max_observations=None, progress_callback=None, total_count=None, existing_ids=None):
    observations = []
    page = 1
    per_page = max(1, int(per_page or resolve_inat_page_size()))
    max_allowed = max_observations if isinstance(max_observations, int) and max_observations > 0 else None
    target_total = total_count if isinstance(total_count, int) and total_count > 0 else None
    if max_allowed is not None and (target_total is None or target_total > max_allowed):
        target_total = max_allowed
    seen_ids = set(str(item) for item in (existing_ids or []))

    while True:
        results = _get_observations_with_retry(
            max_retries=3,
            taxon_name=taxon_name,
            lat=lat,
            lng=lng,
            quality_grade=quality_grade,
            radius=radius,
            captive=False,
            geo=True,
            per_page=per_page,
            page=page,
        )
        if not results:
            break

        raw_results = results.get('results', []) if isinstance(results, dict) else []
        if not raw_results:
            break

        for obs in raw_results:
            if not isinstance(obs, dict):
                continue
            if max_allowed is not None and len(observations) >= max_allowed:
                return pd.DataFrame(observations)

            obs_id = obs.get('id')
            obs_uuid = obs.get('uuid')
            key = str(obs_id) if obs_id is not None else str(obs_uuid) if obs_uuid is not None else None
            if key is not None and key in seen_ids:
                continue
            if key is not None:
                seen_ids.add(key)

            if progress_callback:
                progress_callback(len(observations) + 1, target_total or (len(observations) + 1))

            timestamp = obs.get('observed_on')
            if isinstance(timestamp, datetime):
                date = timestamp.strftime('%Y-%m-%d')
            elif timestamp:
                date = str(timestamp)
            else:
                date = None

            geojson = obs.get('geojson') or {}
            coords = geojson.get('coordinates') if isinstance(geojson, dict) else None
            if not coords or not isinstance(coords, (list, tuple)) or len(coords) < 2:
                coords = [None, None]

            lon_val, lat_val = coords[0], coords[1]
            has_coords = lon_val is not None and lat_val is not None

            elevation = get_elevation(lat_val, lon_val) if has_coords else None
            weather = get_weather(lat_val, lon_val, date) if has_coords and date else {}
            if not isinstance(weather, dict):
                weather = {}

            species_name_found = (obs.get('taxon') or {}).get('name', '') if isinstance(obs.get('taxon'), dict) else ''

            genus_name_found = ''
            if species_name_found:
                genus_name_found = species_name_found.split()[0] if species_name_found else ''

            # How much to trust the coordinates.
            #
            # iNaturalist obscures a location when the observer asks for it, or
            # automatically for a threatened taxon. An obscured record is not
            # merely vague: the published point is randomised inside a 0.2°
            # cell, roughly 20km across, so it can sit on the wrong side of a
            # ridge or in a different watershed entirely. Every terrain and
            # weather value this pipeline samples is read AT the point, so for
            # these records the enrichment describes a place the mushroom was
            # probably not.
            #
            # All of this rides along on the response already being fetched —
            # no extra request — but none of it was being read.
            obscured = bool(obs.get('obscured'))
            geoprivacy = obs.get('geoprivacy')
            taxon_geoprivacy = obs.get('taxon_geoprivacy')
            # public_positional_accuracy is what a stranger sees, and is what
            # matters here: for an obscured record it reflects the whole cell,
            # not the observer's GPS.
            public_accuracy = _coerce_accuracy(obs.get('public_positional_accuracy'))
            accuracy = _coerce_accuracy(obs.get('positional_accuracy'))

            observations.append({
                'uuid': obs.get('uuid'),
                'inat_id': obs.get('id'),
                'obscured': obscured,
                'geoprivacy': geoprivacy,
                'taxon_geoprivacy': taxon_geoprivacy,
                'positional_accuracy': accuracy,
                'public_positional_accuracy': public_accuracy,
                'location_precision': classify_location_precision(
                    obscured, geoprivacy, taxon_geoprivacy, public_accuracy),
                'timestamp': timestamp,
                'date': date,
                'lon': lon_val,
                'lat': lat_val,
                'olc': olc_utils.encode_olc(lat_val, lon_val) if has_coords else None,
                'elevation': elevation,
                'tavg': weather.get('tavg', None),
                'tmin': weather.get('tmin', None),
                'tmax': weather.get('tmax', None),
                'precipitation': weather.get('prcp', None),
                'windspeed': weather.get('wspd', None),
                'winddirection': weather.get('wdir', None),
                'presure': weather.get('pres', None),
                'species': species_name_found or taxon_name,
                'genus': genus_name_found or (taxon_name.split()[0] if taxon_name else ''),
                'location': obs.get('place_guess', ''),
                'num_identification_agreements': obs.get('num_identification_agreements', 0),
            })

        if max_allowed is not None and len(observations) >= max_allowed:
            break
        if len(raw_results) < per_page:
            break
        page += 1

    return pd.DataFrame(observations)

def main():
    env_file = os.getenv('ENV_FILE') or '.env'
    species_value = getenv_with_file('INAT_TAXON_NAME', default=(getenv_with_file('SPECIES', default='morchella', env_file=env_file)), env_file=env_file)
    species_list = parse_species_list(species_value)
    if not species_list:
        species_list = ['morchella']

    quality_grade = getenv_with_file('INAT_QUALITY_GRADE', default=(getenv_with_file('QUALITY_GRADE', default='research', env_file=env_file)), env_file=env_file)

    default_radius = _read_float_env('INAT_RADIUS', 'RADIUS', default=500.0)
    locations = _parse_locations_with_radius(os.getenv('INAT_LOCATIONS'), default_radius)
    plus_locations = parse_plus_codes(os.getenv('INAT_PLUS_CODES'), default_radius)
    if plus_locations:
        locations.extend(plus_locations)
    if not locations:
        lat, lng = _resolve_location_from_env(default_lat=40.0, default_lng=-105.0)
        locations = [{'lat': lat, 'lng': lng, 'radius': default_radius}]

    per_page = resolve_inat_page_size({**load_env_file(env_file), **os.environ})
    max_observations = int(getenv_with_file('INAT_MAX_OBSERVATIONS_PER_SPECIES', default=(getenv_with_file('MAX_OBSERVATIONS_PER_SPECIES', default='0', env_file=env_file)), env_file=env_file) or 0)
    parallel_fetches = get_parallel_fetch_workers({**load_env_file(env_file), **os.environ})
    refresh_all = should_refresh_all()
    existing_inat_ids = set()
    if not refresh_all:
        existing_df = store.load_all(store.SPECIES_DIR)
        if 'inat_id' in existing_df.columns:
            existing_inat_ids = {str(v) for v in existing_df['inat_id'].dropna().tolist()}
        print(f"Incremental refresh enabled. Skipping {len(existing_inat_ids)} existing records by inat_id unless REFRESH_ALL=1.")
    else:
        print('Full refresh enabled: REFRESH_ALL=1, reloading all observations.')

    frames = []
    for loc in locations:
        lat = loc['lat']
        lng = loc['lng']
        radius = loc['radius']
        print(f"Fetching iNaturalist data for {', '.join(species_list)} near {lat}, {lng} within {radius}km (per_page={per_page}, max_per_species={max_observations or 'unlimited'}, parallel_workers={parallel_fetches}, refresh_all={refresh_all})...")
        
        def fetch_single_species(species_name, target_lat=lat, target_lng=lng, target_radius=radius):
            species_total = get_species_observation_total(
                taxon_name=species_name,
                quality_grade=quality_grade,
                lat=target_lat,
                lng=target_lng,
                radius=target_radius,
            )
            if max_observations and species_total > max_observations:
                species_total = max_observations

            progress_callback = lambda current, total, species_name=species_name: None
            df_species = fetch_inat_data(
                taxon_name=species_name,
                quality_grade=quality_grade,
                lat=target_lat,
                lng=target_lng,
                radius=target_radius,
                per_page=per_page,
                max_observations=max_observations or None,
                progress_callback=progress_callback,
                total_count=species_total,
                existing_ids=existing_inat_ids,
            )
            
            new_count = 0 if df_species is None else len(df_species)
            if not refresh_all and df_species is not None and not df_species.empty:
                df_species = df_species[~df_species['inat_id'].astype(str).isin(existing_inat_ids)] if 'inat_id' in df_species.columns else df_species
                new_count = len(df_species)
            print(f"{species_name}: {new_count} new observations fetched")
            return df_species

        with ThreadPoolExecutor(max_workers=max(1, parallel_fetches)) as executor:
            future_map = {executor.submit(fetch_single_species, species): species for species in species_list}
            for future in as_completed(future_map):
                species_name = future_map[future]
                try:
                    df_species = future.result()
                    if df_species is not None and not df_species.empty:
                        frames.append(df_species)
                except Exception as e:
                    print(f"[!] Error fetching taxon '{species_name}': {e}")

    if not frames:
        print('No new observations found; leaving the existing canonical dataset unchanged.')
        return

    df_inat = pd.concat(frames, ignore_index=True)
    print("Data fetched successfully.")

    # Check GROUP_BY env var to determine how to split files
    group_by = os.getenv('GROUP_BY', 'genus')
    key_column = group_by if group_by in df_inat.columns else 'species'

    written = store.write_split(df_inat, base=store.SPECIES_DIR, key=key_column, merge=not refresh_all)

    store.write_geojson_tiles(df_inat)
    total = sum(written.values())
    print(f"Saved {len(df_inat)} fetched observation(s) into {len(written)} {key_column} file(s) "
          f"under {store.SPECIES_DIR}/ ({total} rows on disk after merge/dedup).")

if __name__ == "__main__":
    main()