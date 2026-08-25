-- Canonical observation table for iNaturalist sync.
-- Store a single row per iNaturalist record keyed by inat_id.

create table if not exists public.observations (
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
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists observations_species_idx on public.observations (species);
create index if not exists observations_date_idx on public.observations (date);
create index if not exists observations_location_idx on public.observations using gist (st_geomfromtext('POINT(' || lon || ' ' || lat || ')', 4326));

-- Optional enrichment table for data appended by the Python pipeline.
create table if not exists public.observation_enrichments (
  inat_id bigint primary key references public.observations(inat_id) on delete cascade,
  elevation double precision,
  tavg double precision,
  tmin double precision,
  tmax double precision,
  soil_moisture double precision,
  ndvi double precision,
  precip_7d double precision,
  cluster integer,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists observation_enrichments_cluster_idx on public.observation_enrichments (cluster);

-- Trigger to update updated_at anytime a row changes.
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

create trigger observations_set_updated_at
before update on public.observations
for each row
execute function public.set_updated_at();

create trigger observation_enrichments_set_updated_at
before update on public.observation_enrichments
for each row
execute function public.set_updated_at();

-- Suggested upsert pattern for new data:
-- insert into public.observations (inat_id, uuid, species, date, lat, lon, location, num_identification_agreements, quality_grade, raw_payload)
-- values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
-- on conflict (inat_id)
-- do update set
--   uuid = excluded.uuid,
--   species = excluded.species,
--   date = excluded.date,
--   lat = excluded.lat,
--   lon = excluded.lon,
--   location = excluded.location,
--   num_identification_agreements = excluded.num_identification_agreements,
--   quality_grade = excluded.quality_grade,
--   raw_payload = excluded.raw_payload,
--   updated_at = now();
