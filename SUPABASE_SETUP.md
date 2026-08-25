# Supabase sync setup

This project now includes a lightweight sync helper at `supabase_sync.py`.

## Why this helps

Instead of re-fetching the whole iNaturalist dataset on every run, the sync path can:

1. query the API for each species/location/radius
2. normalize each observation into a canonical row
3. compare by `inat_id`
4. upsert only missing or changed records

That gives you incremental growth without repeated full-network requests.

## Recommended SQL schema

```sql
create table public.observations (
  inat_id bigint primary key,
  uuid text,
  species text,
  date date,
  lat double precision,
  lon double precision,
  location text,
  num_identification_agreements integer,
  quality_grade text,
  raw_payload jsonb,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
```

You can add enrichment tables later, for example:

```sql
create table public.observation_enrichments (
  inat_id bigint primary key references public.observations(inat_id),
  elevation double precision,
  tavg double precision,
  tmin double precision,
  tmax double precision,
  soil_moisture double precision,
  ndvi double precision,
  cluster integer,
  updated_at timestamptz default now()
);
```

## Env vars

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-or-service-role-key
```

## Python usage

```python
from supabase_sync import sync_to_supabase

sync_to_supabase(
    species_list=['morchella', 'amanita'],
    lat=40.0,
    lng=-105.0,
    radius=500,
    quality_grade='research',
    per_page=200,
    max_per_species=500,
)
```

## Next step

The next real implementation is to swap this helper for the official `supabase-py` client and perform an `upsert` keyed by `inat_id`.
