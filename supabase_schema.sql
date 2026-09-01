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

-- ─── Per-user settings and saved charts ──────────────────────────────────────
-- Everything the app persists per viewer (appearance, chart layout, map overlay,
-- units) lives in one JSON blob rather than a column per preference: these are
-- display preferences that change shape as features are added, and a schema
-- migration for every new toggle is not worth it. Saved charts DO get their own
-- table — they are user-authored content, they are listed and reordered, and
-- they deserve to be queryable.

create table if not exists public.user_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  settings jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

create table if not exists public.saved_charts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  -- The chart builder's config, stored whole so an older client cannot lose
  -- fields it does not understand.
  config jsonb not null,
  title text,
  position integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists saved_charts_user_idx on public.saved_charts (user_id, position);

-- Row-level security: a signed-in user reaches only their own rows. Without
-- this, the anon key would expose every user's settings to every other user.
alter table public.user_settings enable row level security;
alter table public.saved_charts enable row level security;

drop policy if exists "own settings" on public.user_settings;
create policy "own settings" on public.user_settings
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own charts" on public.saved_charts;
create policy "own charts" on public.saved_charts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

create trigger user_settings_set_updated_at
before update on public.user_settings
for each row
execute function public.set_updated_at();

create trigger saved_charts_set_updated_at
before update on public.saved_charts
for each row
execute function public.set_updated_at();
