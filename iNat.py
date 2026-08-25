import os
import re
from pathlib import Path

from pyinaturalist import get_observations
import pandas as pd
from meteostat import Point, stations, daily
from datetime import datetime
import requests

# https://www.inaturalist.org/observations?subview=map

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
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url)
    if r.ok:
        return r.json()['results'][0]['elevation']
    return None

def get_weather(lat, lon, date_str):
    if not date_str:
        return {'station_id': None}

    date = datetime.strptime(date_str, '%Y-%m-%d').date()
    point = Point(lat, lon)
    nearby = stations.nearby(point, radius=50000, limit=5)

    for station_id in nearby.index:
        ts = daily(station_id, date, date)
        df = ts.fetch() if hasattr(ts, 'fetch') else ts
        if df is not None and not df.empty:
            row = df.iloc[0].to_dict()
            row['station_used'] = station_id
            return row
    return {'station_id': None}

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

def _slugify(value):
    slug = re.sub(r'[^a-zA-Z0-9]+', '-', str(value).strip().lower()).strip('-')
    return slug or 'mushroom'


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


def should_refresh_all(env=None):
    values = {**(env or {})}
    for key in ('REFRESH_ALL', 'INAT_REFRESH_ALL', 'FULL_REFRESH'):
        raw = values.get(key)
        if raw is None:
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


def _unique_output_base(prefix='mushroom_observations', species='morchella', lat=40.0, lng=-105.0, radius=500.0):
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    species_list = parse_species_list(species)
    slug = '-'.join(_slugify(s) for s in species_list) if species_list else 'mushroom'
    return f"{prefix}_{slug}_{lat}_{lng}_{radius}_{timestamp}"

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


def fetch_inat_data(taxon_name='morchella', quality_grade='research', lat=40.0, lng=-105.0, radius=500.0, per_page=100, max_observations=None, progress_callback=None, total_count=None):
    observations = []
    page = 1
    max_allowed = max_observations if isinstance(max_observations, int) and max_observations > 0 else None
    target_total = total_count if isinstance(total_count, int) and total_count > 0 else None
    if max_allowed is not None and (target_total is None or target_total > max_allowed):
        target_total = max_allowed

    while True:
        results = get_observations(
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

        raw_results = results.get('results', []) if isinstance(results, dict) else []
        if not raw_results:
            break

        for obs in raw_results:
            if max_allowed is not None and len(observations) >= max_allowed:
                return pd.DataFrame(observations)

            if progress_callback:
                progress_callback(len(observations) + 1, target_total or (len(observations) + 1))

            timestamp = obs.get('observed_on')
            if isinstance(timestamp, datetime):
                date = timestamp.strftime('%Y-%m-%d')
            elif not timestamp:
                date = None

            coords = obs['geojson']['coordinates'] if 'geojson' in obs else [None, None]
            elevation = get_elevation(coords[1], coords[0])
            weather = get_weather(coords[1], coords[0], date) if coords[0] and coords[1] and date else {}
            if not isinstance(weather, dict):
                weather = {}
                print(f"Unexpected weather data type: {type(weather)}")

            observations.append({
                'uuid': obs.get('uuid'),
                'inat_id': obs.get('id'),
                'timestamp': timestamp,
                'date': date,
                'lon': coords[0],
                'lat': coords[1],
                'elevation': elevation,
                'tavg': weather.get('tavg', None),
                'tmin': weather.get('tmin', None),
                'tmax': weather.get('tmax', None),
                'precipitation': weather.get('prcp', None),
                'windspeed': weather.get('wspd', None),
                'winddirection': weather.get('wdir', None),
                'presure': weather.get('pres', None),
                'species': obs.get('taxon', {}).get('name', ''),
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
    lat, lng = _resolve_location_from_env(default_lat=40.0, default_lng=-105.0)
    radius = _read_float_env('INAT_RADIUS', 'RADIUS', default=500.0)
    per_page = int(getenv_with_file('INAT_PER_PAGE', default=(getenv_with_file('PER_PAGE', default='100', env_file=env_file)), env_file=env_file) or 100)
    max_observations = int(getenv_with_file('INAT_MAX_OBSERVATIONS_PER_SPECIES', default=(getenv_with_file('MAX_OBSERVATIONS_PER_SPECIES', default='0', env_file=env_file)), env_file=env_file) or 0)
    refresh_all = should_refresh_all()
    output_prefix = getenv_with_file('OUTPUT_PREFIX', default='mushroom_observations', env_file=env_file)
    output_dir = getenv_with_file('OUTPUT_DIR', default='.', env_file=env_file)
    os.makedirs(output_dir, exist_ok=True)

    canonical_csv = os.path.join(output_dir, 'mushroom_observations.csv')
    canonical_geojson = os.path.join(output_dir, 'mushroom_observations.geojson')
    existing_inat_ids = set()
    if not refresh_all:
        existing_inat_ids = _existing_observation_ids(canonical_csv)
        print(f"Incremental refresh enabled. Skipping {len(existing_inat_ids)} existing records by inat_id unless REFRESH_ALL=1.")
    else:
        print('Full refresh enabled: REFRESH_ALL=1, reloading all observations.')

    print(f"Fetching iNaturalist data for {', '.join(species_list)} near {lat}, {lng} within {radius}km (per_page={per_page}, max_per_species={max_observations or 'unlimited'}, refresh_all={refresh_all})...")
    frames = []
    total_species = len(species_list)
    for index, species in enumerate(species_list, start=1):
        print(f"\n[{index}/{total_species}] {species} {render_progress_bar(index, total_species)}")

        species_total = get_species_observation_total(
            taxon_name=species,
            quality_grade=quality_grade,
            lat=lat,
            lng=lng,
            radius=radius,
        )
        if max_observations and species_total > max_observations:
            species_total = max_observations

        def progress_callback(current, total, species_name=species):
            if total <= 0:
                return
            print(f"\r  {format_observation_progress(species_name, current, total, width=20)}", end='', flush=True)

        df_species = fetch_inat_data(
            taxon_name=species,
            quality_grade=quality_grade,
            lat=lat,
            lng=lng,
            radius=radius,
            per_page=per_page,
            max_observations=max_observations or None,
            progress_callback=progress_callback,
            total_count=species_total,
        )
        print()
        count = len(df_species) if df_species is not None else 0
        if not refresh_all and df_species is not None and not df_species.empty:
            df_species = df_species[~df_species['inat_id'].astype(str).isin(existing_inat_ids)] if 'inat_id' in df_species.columns else df_species
            count = len(df_species)
        print(f"  -> {species}: {count} new observations fetched")
        if df_species is not None and not df_species.empty:
            frames.append(df_species)
    if not frames:
        print('No new observations found; leaving the existing canonical dataset unchanged.')
        return

    df_inat = pd.concat(frames, ignore_index=True)
    print("Data fetched successfully.")

    if refresh_all:
        base_name = _unique_output_base(output_prefix, species_list, lat, lng, radius)
        csv_path = os.path.join(output_dir, f'{base_name}.csv')
        geojson_path = os.path.join(output_dir, f'{base_name}.geojson')
        print(f"Saving CSV to {csv_path}...")
        df_inat.to_csv(csv_path, index=False)
        print(f"Saving GeoJSON to {geojson_path}...")
        df_inat.to_json(geojson_path, orient='records')

    print(f"Saving canonical CSV to {canonical_csv}...")
    df_inat.to_csv(canonical_csv, index=False)
    print(f"Saving canonical GeoJSON to {canonical_geojson}...")
    df_inat.to_json(canonical_geojson, orient='records')
    print("Data saved successfully.")


if __name__ == "__main__":
    main()