import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from pyinaturalist import get_observations


def normalize_observation_record(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Map a raw iNaturalist observation dict into the canonical Supabase row."""
    timestamp = obs.get('observed_on')
    if isinstance(timestamp, datetime):
        date_value = timestamp.strftime('%Y-%m-%d')
    else:
        date_value = timestamp if isinstance(timestamp, str) else None

    coords = obs.get('geojson', {}).get('coordinates') if isinstance(obs.get('geojson'), dict) else [None, None]
    lon = coords[0] if isinstance(coords, list) and len(coords) > 1 else None
    lat = coords[1] if isinstance(coords, list) and len(coords) > 1 else None

    return {
        'inat_id': obs.get('id'),
        'uuid': obs.get('uuid'),
        'date': date_value,
        'lon': lon,
        'lat': lat,
        'species': obs.get('taxon', {}).get('name', '') if isinstance(obs.get('taxon'), dict) else '',
        'location': obs.get('place_guess', ''),
        'num_identification_agreements': obs.get('num_identification_agreements', 0),
        'quality_grade': obs.get('quality_grade'),
        'raw_payload': json.dumps(obs, sort_keys=True),
        'updated_at': datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    }


def fetch_species_observations(
    taxon_name: str,
    lat: float,
    lng: float,
    radius: float,
    quality_grade: str = 'research',
    per_page: int = 200,
    page: int = 1,
) -> Dict[str, Any]:
    """Return a page of iNaturalist records for a species and location."""
    return get_observations(
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


def gather_observations_for_sync(
    species_list: Iterable[str],
    lat: float,
    lng: float,
    radius: float,
    quality_grade: str = 'research',
    per_page: int = 200,
    max_per_species: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch all observations for each species and normalize them for upsert."""
    all_rows = []
    for species in species_list:
        page = 1
        seen = 0
        while True:
            response = fetch_species_observations(species, lat, lng, radius, quality_grade, per_page, page)
            results = response.get('results', []) if isinstance(response, dict) else []
            if not results:
                break

            for obs in results:
                row = normalize_observation_record(obs)
                all_rows.append(row)
                seen += 1
                if max_per_species and seen >= max_per_species:
                    break
            if max_per_species and seen >= max_per_species:
                break
            if len(results) < per_page:
                break
            page += 1
    return all_rows


def sync_to_supabase(
    species_list: Iterable[str],
    lat: float,
    lng: float,
    radius: float,
    quality_grade: str = 'research',
    per_page: int = 200,
    max_per_species: Optional[int] = None,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
):
    """Minimal placeholder for a Supabase upsert workflow.

    This intentionally avoids hard dependencies on the client package so the
    project can use the same code without requiring first-party SDK setup.
    """
    url = supabase_url or os.getenv('SUPABASE_URL')
    key = supabase_key or os.getenv('SUPABASE_KEY')
    if not url or not key:
        raise RuntimeError('SUPABASE_URL and SUPABASE_KEY must be set to sync into Supabase.')

    rows = gather_observations_for_sync(
        species_list=species_list,
        lat=lat,
        lng=lng,
        radius=radius,
        quality_grade=quality_grade,
        per_page=per_page,
        max_per_species=max_per_species,
    )

    return {
        'url': url,
        'rows': len(rows),
        'message': 'Ready for Supabase upsert. Use the table schema below to insert.'
    }


if __name__ == '__main__':
    print('Supabase sync helper loaded. Set SUPABASE_URL and SUPABASE_KEY to enable sync.')
