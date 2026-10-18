-- Where a saved dataset came from, for the ones no job produced.
--
-- A dataset saved from a job already has its settings: the job row keeps the
-- spec it ran (source, stages, dates) and what came back (bands, skipped
-- points). A fetched, uploaded or imported dataset had nothing — the query
-- that produced it was gone once the file was written, so its detail view
-- could say only "species/gbif-x.geojson".
--
-- Descriptive only. Nothing reads it to decide access or to run anything, so
-- it is written server-side from a whitelist of fields (see
-- netlify/lib/dataset-detail.mjs, cleanProvenance) and never from the request
-- verbatim.
--
--   { "origin": "inat" | "gbif" | "upload" | "gbif_archive" | "ee_asset" | "job",
--     "taxon", "date_from", "date_to", "lat", "lng", "radius_km", "quality",
--     "file_name", "format", "records", "skipped", "media", "dna",
--     "asset_path", "fetched_at" }

alter table public.saved_datasets add column if not exists provenance jsonb;

comment on column public.saved_datasets.provenance is
  'How a dataset no job produced was made: the fetch query, upload or import. See migration 019.';
