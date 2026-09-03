import os
import sys
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pyinaturalist import get_observations
import pandas as pd
try:
    from meteostat import Point, stations, daily
    _METEOSTAT_AVAILABLE = True
except ImportError:
    # Weather enrichment is best-effort. If meteostat isn't installed
    # (e.g. a hosted notebook that doesn't install requirements.txt), the
    # pipeline still runs and simply skips weather lookups.
    Point = stations = daily = None
    _METEOSTAT_AVAILABLE = False
    print(
        "Warning: 'meteostat' not installed; weather enrichment will be "
        "skipped. Install it with `pip install meteostat` to enable.",
        file=sys.stderr,
    )
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

# One shared, rate-limited session for every iNaturalist call. Without it each
# parallel worker (and each retry) uses its own default session, so their
# per-session limiters don't coordinate and together they burst past
# iNaturalist's limit — which is what produced the 429 "normal_throttling"
# errors. A single session shared across threads throttles them as a group,
# staying under ~1 request/second and 60/minute (iNaturalist's published
# ceiling), and carries pyinaturalist's own 429-aware retry/backoff.
_INAT_SESSION = None
_INAT_SESSION_READY = False

def _inat_session():
    global _INAT_SESSION, _INAT_SESSION_READY
    if _INAT_SESSION_READY:
        return _INAT_SESSION
    _INAT_SESSION_READY = True
    try:
        from pyinaturalist import ClientSession
        _INAT_SESSION = ClientSession(
            per_second=1,        # iNaturalist asks for ~1 request/second
            per_minute=60,       # and no more than 60/minute
            per_day=10000,
            burst=1,             # no bursts — bursts are what tripped the throttle
            max_retries=5,
            backoff_factor=2.0,  # 429s back off exponentially inside the session
        )
    except Exception as exc:
        # Older pyinaturalist, or a constructor change: fall back to the default
        # session rather than failing the fetch. Rate limiting is then weaker,
        # but the custom retry/backoff below still applies.
        print(f"[!] Could not build a shared rate-limited iNaturalist session "
              f"({exc}); using the default session.", file=sys.stderr)
        _INAT_SESSION = None
    return _INAT_SESSION

def _observation_kwargs(**kwargs):
    """Common get_observations kwargs, adding the shared session when available."""
    session = _inat_session()
    if session is not None:
        kwargs['session'] = session
    return kwargs

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
    if not date_str or not _METEOSTAT_AVAILABLE:
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
    """Turn ``INAT_PLUS_CODES`` into bounding-box locations.

    A plus code names a rectangular cell, so we fetch that box directly rather
    than reusing the circular ``radius`` model — the box is the footprint the
    code actually describes, and its own length sets the size. A shorter,
    zero-padded full code covers a larger box: ``84QW4600+`` is ~0.05° (~5.5km),
    ``84QW0000+`` is ~1° (~111km), ``85000000+`` is ~20°. Padding must drop whole
    pairs, so an odd-padded code like ``84Q00000+`` is not a valid full code and
    is skipped. Each entry is just a full plus code; any extra comma-separated
    field is ignored (radius does not apply to a box). ``default_radius`` is
    accepted for signature symmetry but unused.
    """
    locations = []
    if not env_value:
        return locations
    for entry in env_value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        plus = entry.split(",")[0].strip()
        try:
            swlat, swlng, nelat, nelng = olc_utils.decode_olc_bounds(plus)
            locations.append({
                "swlat": swlat, "swlng": swlng,
                "nelat": nelat, "nelng": nelng,
                "label": plus,
            })
        except Exception as exc:
            print(f"[!] Invalid plus-code entry ignored: {entry} ({exc})")
    return locations

def parse_plus_code_ranges(env_value: str) -> list[dict]:
    """Turn ``INAT_PLUS_CODE_RANGES`` into bounding-box locations.

    Where ``INAT_PLUS_CODES`` fetches one code's cell, a range fetches the box
    that *spans two* codes — the union of their cells — so two corner codes can
    describe an area of any size and aspect, not just the fixed square grid a
    single code offers. Each entry is two full plus codes separated by ``:`` (or
    whitespace), semicolon-separated between entries:

        INAT_PLUS_CODES_RANGES=84QWJF00+:85GRHM00+; 84QV0000+ 84QX0000+

    The two cells' corners are merged into one south-west / north-east box, so
    the order of the two codes does not matter.
    """
    locations = []
    if not env_value:
        return locations
    for entry in env_value.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = [p for p in re.split(r'[:\s]+', entry) if p]
        if len(parts) != 2:
            print(f"[!] Plus-code range needs exactly two codes, ignoring: {entry}")
            continue
        try:
            a = olc_utils.decode_olc_bounds(parts[0])
            b = olc_utils.decode_olc_bounds(parts[1])
            # a, b are (swlat, swlng, nelat, nelng); union the two cells.
            swlat = min(a[0], b[0])
            swlng = min(a[1], b[1])
            nelat = max(a[2], b[2])
            nelng = max(a[3], b[3])
            # Sorted label so the two orderings produce identical locations.
            lo, hi = sorted((parts[0], parts[1]))
            locations.append({
                "swlat": swlat, "swlng": swlng,
                "nelat": nelat, "nelng": nelng,
                "label": f"{lo}:{hi}",
            })
        except Exception as exc:
            print(f"[!] Invalid plus-code range ignored: {entry} ({exc})")
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

def _geo_query(lat=None, lng=None, radius=None, bounds=None):
    """iNaturalist geo params: a bounding box when ``bounds`` is given, else the
    point+radius model. ``bounds`` is (swlat, swlng, nelat, nelng).

    A bounding box is how a plus code's rectangular cell is fetched faithfully —
    the API's nelat/nelng/swlat/swlng, not a circle around the centroid.
    """
    if bounds is not None:
        swlat, swlng, nelat, nelng = bounds
        return {'nelat': nelat, 'nelng': nelng, 'swlat': swlat, 'swlng': swlng}
    return {'lat': lat, 'lng': lng, 'radius': radius}


def get_species_observation_total(taxon_name='morchella', quality_grade='research', lat=40.0, lng=-105.0, radius=500.0, bounds=None):
    try:
        response = get_observations(**_observation_kwargs(
            taxon_name=taxon_name,
            quality_grade=quality_grade,
            captive=False,
            geo=True,
            per_page=1,
            page=1,
            **_geo_query(lat, lng, radius, bounds),
        ))
        if not isinstance(response, dict):
            return 0
        total = response.get('total_results')
        return int(total) if total is not None else 0
    except Exception:
        return 0

def _retry_after_seconds(err, attempt):
    """How long to wait before retrying, in seconds.

    A 429 (throttled) is a rate limit, not a transient blip, so it needs a real
    pause: honour the server's ``Retry-After`` header when present, otherwise
    back off exponentially from a floor well above iNaturalist's ~1 req/s limit.
    Other errors keep the short linear backoff.
    """
    response = getattr(err, 'response', None)
    status = getattr(response, 'status_code', None)
    if status == 429:
        retry_after = (getattr(response, 'headers', None) or {}).get('Retry-After')
        if retry_after:
            try:
                return max(float(retry_after), 1.0)
            except (TypeError, ValueError):
                pass
        return min(60.0, 5.0 * (2 ** (attempt - 1)))  # 5s, 10s, 20s, ... capped
    return 1.0 * attempt

def _get_observations_with_retry(max_retries=5, **kwargs):
    for attempt in range(1, max_retries + 1):
        try:
            return get_observations(**_observation_kwargs(**kwargs))
        except Exception as err:
            if attempt == max_retries:
                taxon = kwargs.get('taxon_name')
                page = kwargs.get('page', 1)
                print(f"\n[!] iNaturalist query failed for taxon '{taxon}' on page {page}: {err}")
                return None
            delay = _retry_after_seconds(err, attempt)
            status = getattr(getattr(err, 'response', None), 'status_code', None)
            if status == 429:
                print(f"  [throttled] iNaturalist 429 for '{kwargs.get('taxon_name')}' "
                      f"page {kwargs.get('page', 1)}; waiting {delay:.0f}s "
                      f"(attempt {attempt}/{max_retries})...", flush=True)
            time.sleep(delay)
    return None

def fetch_inat_data(taxon_name='morchella', quality_grade='research', lat=40.0, lng=-105.0, radius=500.0, bounds=None, per_page=200, max_observations=None, progress_callback=None, total_count=None, existing_ids=None):
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
            quality_grade=quality_grade,
            captive=False,
            geo=True,
            per_page=per_page,
            page=page,
            **_geo_query(lat, lng, radius, bounds),
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
    range_locations = parse_plus_code_ranges(
        os.getenv('INAT_PLUS_CODE_RANGES') or os.getenv('INAT_PLUS_CODES_RANGES'))
    if range_locations:
        locations.extend(range_locations)
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
        # A location is either a bounding box (from a plus code) or the
        # point+radius model. Build the geo kwargs once and a label for the log.
        if 'nelat' in loc:
            target_bounds = (loc['swlat'], loc['swlng'], loc['nelat'], loc['nelng'])
            where = (f"plus-code box {loc.get('label', '')} "
                     f"[{loc['swlat']:.4f},{loc['swlng']:.4f} → {loc['nelat']:.4f},{loc['nelng']:.4f}]")
        else:
            target_bounds = None
            where = f"near {loc['lat']}, {loc['lng']} within {loc['radius']}km"
        print(f"Fetching iNaturalist data for {', '.join(species_list)} {where} "
              f"(per_page={per_page}, max_per_species={max_observations or 'unlimited'}, "
              f"parallel_workers={parallel_fetches}, refresh_all={refresh_all})...")

        def fetch_single_species(species_name, target_lat=loc.get('lat'), target_lng=loc.get('lng'),
                                 target_radius=loc.get('radius'), target_bounds=target_bounds):
            species_total = get_species_observation_total(
                taxon_name=species_name,
                quality_grade=quality_grade,
                lat=target_lat,
                lng=target_lng,
                radius=target_radius,
                bounds=target_bounds,
            )
            if max_observations and species_total > max_observations:
                species_total = max_observations

            print(f"  → {species_name}: {species_total or 'unknown'} observation(s) to fetch...", flush=True)

            # Throttled milestone progress so a long multi-species / multi-location
            # run shows steady movement instead of going silent between the start
            # line and the per-species total. Prints once per ~25% of the fetch;
            # species-prefixed so the parallel workers stay legible interleaved.
            last_milestone = {'value': -1}

            def progress_callback(current, total, species_name=species_name):
                if not total:
                    return
                milestone = int(current / total * 4)  # 0..4 → every ~25%
                if milestone == last_milestone['value']:
                    return
                last_milestone['value'] = milestone
                pct = min(100, int(current / total * 100))
                print(f"    {species_name}: {current}/{total} ({pct}%)", flush=True)

            df_species = fetch_inat_data(
                taxon_name=species_name,
                quality_grade=quality_grade,
                lat=target_lat,
                lng=target_lng,
                radius=target_radius,
                bounds=target_bounds,
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

    # Drop empty / all-NA frames before concat: pandas warns (and will change
    # dtype behavior) when these are mixed in, and they carry no rows anyway.
    frames = [f for f in frames if f is not None and not f.empty and not f.isna().all(axis=None)]
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