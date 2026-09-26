-- Drop the old Postgres observation tables.
-- The pipeline never writes to these; all observation data lives in Supabase
-- Storage as GeoJSON. Verify no rows exist before running on production:
--   SELECT count(*) FROM public.observations;
--   SELECT count(*) FROM public.observation_enrichments;

DROP TABLE IF EXISTS public.observation_enrichments CASCADE;
DROP TABLE IF EXISTS public.observations CASCADE;
