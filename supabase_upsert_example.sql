-- Example upsert for the canonical observations table.
-- Run this in the Supabase SQL editor after creating the table from supabase_schema.sql.

insert into public.observations (
  inat_id,
  uuid,
  species,
  date,
  lat,
  lon,
  location,
  num_identification_agreements,
  quality_grade,
  raw_payload
)
values (
  $1,
  $2,
  $3,
  $4,
  $5,
  $6,
  $7,
  $8,
  $9,
  $10
)
on conflict (inat_id)
do update set
  uuid = excluded.uuid,
  species = excluded.species,
  date = excluded.date,
  lat = excluded.lat,
  lon = excluded.lon,
  location = excluded.location,
  num_identification_agreements = excluded.num_identification_agreements,
  quality_grade = excluded.quality_grade,
  raw_payload = excluded.raw_payload,
  updated_at = now();

-- Example with a batch payload:
-- insert into public.observations (inat_id, uuid, species, date, lat, lon, location, num_identification_agreements, quality_grade, raw_payload)
-- values
--   (123, 'uuid-1', 'morchella', '2024-05-01', 40.1, -105.2, 'Fort Collins', 3, 'research', '{"source":"inaturalist"}'::jsonb),
--   (456, 'uuid-2', 'amanita', '2024-05-02', 40.2, -105.1, 'Boulder', 1, 'research', '{"source":"inaturalist"}'::jsonb)
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
