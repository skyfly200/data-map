from pyinaturalist import get_observations
import pandas as pd
from meteostat import Point, Daily
from datetime import datetime
import requests

def get_elevation(lat, lon):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    r = requests.get(url)
    if r.ok:
        return r.json()['results'][0]['elevation']
    return None

def get_weather(lat, lon, date_str):
    date = datetime.strptime(date_str, '%Y-%m-%d')
    location = Point(lat, lon)
    data = Daily(location, date, date)
    data = data.fetch()
    if not data.empty:
        return data.iloc[0].to_dict()
    return {}  # Ensure an empty dictionary is returned if no data is available

def fetch_inat_data(taxon_name='Pleurotus ostreatus', quality_grade='research', lat=40.0, lng=-105.0, radius=50.0, per_page=10):
    results = get_observations(
        taxon_name=taxon_name,
        lat=lat,
        lng=lng,
        quality_grade=quality_grade,
        radius=radius,
        geo=True,
        per_page=per_page,
        fields='observed_on,geojson,place_guess,taxon'
    )

    observations = []
    for obs in results['results']:
        date = obs.get('observed_on')
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        elif not date:  # Handle missing or None dates
            date = None

        coords = obs['geojson']['coordinates'] if 'geojson' in obs else [None, None]
        elevation = get_elevation(coords[1], coords[0])
        weather = get_weather(coords[1], coords[0], date) if coords[0] and coords[1] and date else {}
        if not isinstance(weather, dict):  # Safeguard against unexpected types
            weather = {}
            
        observations.append({
            'date': date,
            'lon': coords[0],
            'lat': coords[1],
            'elevation': elevation,
            'temp': weather.get('tavg', None),
            'precipitation': weather.get('prcp', None),
            'humidity': weather.get('rh', None),
            'species': obs.get('taxon', {}).get('name', ''),
            'location': obs.get('place_guess', '')
        })

    df = pd.DataFrame(observations)
    return df

print("Fetching iNaturalist data...")
df_inat = fetch_inat_data()
print("Data fetched successfully.")
print(df_inat.head())
print("Saving data to CSV...")
df_inat.to_csv('mushroom_observations.csv', index=False)
df_inat.to_json('mushroom_observations.geojson', orient='records')
print("Data saved successfully.")