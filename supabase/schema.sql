-- ── Datasets ──────────────────────────────────────────────────────────────────
-- config shape: { taxon_name, quality_grade, lat, lng, radius, per_page, n_clusters }

CREATE TABLE datasets (
  id                uuid        DEFAULT gen_random_uuid() PRIMARY KEY,
  name              text        NOT NULL,
  description       text,
  config            jsonb       NOT NULL DEFAULT '{}',
  status            text        NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'complete', 'error')),
  stage             text        NOT NULL DEFAULT 'idle'
                    CHECK (stage IN ('idle', 'inat', 'enrich', 'cluster', 'done', 'error')),
  observation_count integer     DEFAULT 0,
  error_message     text,
  created_at        timestamptz DEFAULT now(),
  updated_at        timestamptz DEFAULT now()
);

-- ── Observations ──────────────────────────────────────────────────────────────

CREATE TABLE observations (
  id            uuid            DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id    uuid            NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  inat_uuid     text            NOT NULL,
  taxon_name    text,
  lat           double precision NOT NULL,
  lon           double precision NOT NULL,
  observed_on   date,
  quality_grade text,
  place_guess   text,
  -- Elevation (Open-Elevation)
  elevation     double precision,
  -- Weather (Open-Meteo historical archive)
  temp_avg      double precision,
  precip_total  double precision,
  wind_speed    double precision,
  -- 7-day precipitation lookback
  prcp_d0       double precision,
  prcp_d1       double precision,
  prcp_d2       double precision,
  prcp_d3       double precision,
  prcp_d4       double precision,
  prcp_d5       double precision,
  prcp_d6       double precision,
  -- Remote sensing (Google Earth Engine)
  ndvi          double precision,
  soil_moisture double precision,
  land_cover    integer,
  -- Clustering output
  cluster       integer,

  created_at    timestamptz     DEFAULT now(),

  UNIQUE (inat_uuid, dataset_id)
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX observations_dataset_id_idx ON observations(dataset_id);
CREATE INDEX observations_cluster_idx    ON observations(dataset_id, cluster);

-- ── Auto-update updated_at ────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER datasets_updated_at
  BEFORE UPDATE ON datasets
  FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── Row Level Security ────────────────────────────────────────────────────────

ALTER TABLE datasets     ENABLE ROW LEVEL SECURITY;
ALTER TABLE observations ENABLE ROW LEVEL SECURITY;

-- Public read (map viewer)
CREATE POLICY "public read datasets"
  ON datasets FOR SELECT TO anon USING (true);

CREATE POLICY "public read observations"
  ON observations FOR SELECT TO anon USING (true);

-- Anon can create datasets (personal tool — add auth later as needed)
CREATE POLICY "anon insert datasets"
  ON datasets FOR INSERT TO anon WITH CHECK (true);

-- Pipeline functions use service role key which bypasses RLS entirely.
-- No additional INSERT/UPDATE policies needed for observations.

-- ── Realtime ─────────────────────────────────────────────────────────────────
-- Run this after creating the table to enable realtime status updates:
--
--   ALTER PUBLICATION supabase_realtime ADD TABLE datasets;
--
-- (Do this in the Supabase dashboard SQL editor or via CLI)
