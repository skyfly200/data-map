from pyinaturalist import get_observations
import pandas as pd
from meteostat import Stations, Point, Daily
from datetime import datetime
import requests

# https://www.inaturalist.org/observations?subview=map

def get_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url)
    if r.ok:
        return r.json()['results'][0]['elevation']
    return None

def get_weather(lat, lon, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    stations = Stations().nearby(lat,lon).fetch(5)

    for station_id in stations.index:
        df = Daily(station_id, date, date).fetch()
        if not df.empty:
            return df.iloc[0].to_dict() | {'station_used': station_id}
    return {'station_id': None}

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
            
        print(weather)

        observations.append({
            'uuid': obs.get('uuid'),
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

print("Fetching iNaturalist data...")
df_inat = fetch_inat_data()
print("Data fetched successfully.")
# print(df_inat.head())
print("Saving data to CSV...")
df_inat.to_csv('mushroom_observations.csv', index=False)
print("Saving data to GeoJSON...")
df_inat.to_json('mushroom_observations.geojson', orient='records')
print("Data saved successfully.")