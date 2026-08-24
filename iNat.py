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

    date = datetime.strptime(date_str, '%Y-%m-%d')
    point = Point(lat, lon)
    nearby = stations.nearby(point, radius=50000, limit=5)

    for station_id in nearby.index:
        df = daily(station_id, date, date)
        if not df.empty:
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

def _unique_output_base(prefix='mushroom_observations', species='morchella', lat=40.0, lng=-105.0, radius=500.0):
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    slug = _slugify(species)
    return f"{prefix}_{slug}_{lat}_{lng}_{radius}_{timestamp}"

def fetch_inat_data(taxon_name='morchella', quality_grade='research', lat=40.0, lng=-105.0, radius=500.0, per_page=100):
    results = get_observations(
        taxon_name=taxon_name,
        lat=lat,
        lng=lng,
        quality_grade=quality_grade,
        radius=radius,
        captive=False,
        geo=True,
        per_page=per_page,
    )

    observations = []
    for obs in results['results']:
        timestamp = obs.get('observed_on')
        if isinstance(timestamp, datetime):
            date = timestamp.strftime('%Y-%m-%d')
        elif not timestamp:  # Handle missing or None dates
            date = None

        coords = obs['geojson']['coordinates'] if 'geojson' in obs else [None, None]
        elevation = get_elevation(coords[1], coords[0])
        weather = get_weather(coords[1], coords[0], date) if coords[0] and coords[1] and date else {}
        if not isinstance(weather, dict):  # Safeguard against unexpected types
            weather = {}
            print(f"Unexpected weather data type: {type(weather)}")

        observations.append({
            'uuid': obs.get('uuid'),
            'inat_id': obs.get('id'),  # numeric id for the canonical iNat URL
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

    df = pd.DataFrame(observations)
    return df

def main():
    env_file = os.getenv('ENV_FILE') or '.env'
    species = getenv_with_file('INAT_TAXON_NAME', default=(getenv_with_file('SPECIES', default='morchella', env_file=env_file)), env_file=env_file)
    quality_grade = getenv_with_file('INAT_QUALITY_GRADE', default=(getenv_with_file('QUALITY_GRADE', default='research', env_file=env_file)), env_file=env_file)
    lat, lng = _resolve_location_from_env(default_lat=40.0, default_lng=-105.0)
    radius = _read_float_env('INAT_RADIUS', 'RADIUS', default=500.0)
    per_page = int(getenv_with_file('INAT_PER_PAGE', default=(getenv_with_file('PER_PAGE', default='100', env_file=env_file)), env_file=env_file) or 100)
    output_prefix = getenv_with_file('OUTPUT_PREFIX', default='mushroom_observations', env_file=env_file)
    output_dir = getenv_with_file('OUTPUT_DIR', default='.', env_file=env_file)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching iNaturalist data for {species} near {lat}, {lng} within {radius}km...")
    df_inat = fetch_inat_data(
        taxon_name=species,
        quality_grade=quality_grade,
        lat=lat,
        lng=lng,
        radius=radius,
        per_page=per_page,
    )
    print("Data fetched successfully.")

    base_name = _unique_output_base(output_prefix, species, lat, lng, radius)
    csv_path = os.path.join(output_dir, f'{base_name}.csv')
    geojson_path = os.path.join(output_dir, f'{base_name}.geojson')

    print(f"Saving CSV to {csv_path}...")
    df_inat.to_csv(csv_path, index=False)
    print(f"Saving GeoJSON to {geojson_path}...")
    df_inat.to_json(geojson_path, orient='records')

    canonical_csv = os.path.join(output_dir, 'mushroom_observations.csv')
    canonical_geojson = os.path.join(output_dir, 'mushroom_observations.geojson')
    print(f"Saving canonical CSV to {canonical_csv}...")
    df_inat.to_csv(canonical_csv, index=False)
    print(f"Saving canonical GeoJSON to {canonical_geojson}...")
    df_inat.to_json(canonical_geojson, orient='records')
    print("Data saved successfully.")


if __name__ == "__main__":
    main()