-- WANT-17: persist per-model variable contributions + preferred ranges, and
-- add tables for access data (PAD-US land status, OSM roads/trails).
-- Not applied automatically. Requires PostGIS (already used by observations).

ALTER TABLE model_results
  ADD COLUMN IF NOT EXISTS contributions JSONB,      -- { predictor: percent }, over the model's own predictors
  ADD COLUMN IF NOT EXISTS predictor_ranges JSONB;   -- { predictor: { p25, p75, min, max } } over presences

COMMENT ON COLUMN model_results.contributions IS
  'Per-predictor % contribution from the scout fit. NULL when the run was not autoOptimize.';

-- PAD-US: public land units with legal access class.
CREATE TABLE IF NOT EXISTS access_areas (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL DEFAULT 'padus',
  source_id TEXT NOT NULL,
  name TEXT,
  manager TEXT,
  designation TEXT,
  access_class TEXT NOT NULL CHECK (access_class IN ('open','restricted','closed','unknown')),
  geom geometry(MultiPolygon, 4326) NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (source, source_id)
);
CREATE INDEX IF NOT EXISTS access_areas_geom_idx ON access_areas USING GIST (geom);

-- OSM roads and trails: physical access.
CREATE TABLE IF NOT EXISTS access_lines (
  id BIGSERIAL PRIMARY KEY,
  osm_id BIGINT NOT NULL UNIQUE,
  kind TEXT NOT NULL CHECK (kind IN ('road','trail')),
  highway TEXT NOT NULL,
  name TEXT,
  geom geometry(LineString, 4326) NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS access_lines_geom_idx ON access_lines USING GIST (geom);

-- Reference data: readable by anyone, written only by the service role.
ALTER TABLE access_areas ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_lines ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "access_areas readable" ON access_areas;
CREATE POLICY "access_areas readable" ON access_areas FOR SELECT USING (true);
DROP POLICY IF EXISTS "access_lines readable" ON access_lines;
CREATE POLICY "access_lines readable" ON access_lines FOR SELECT USING (true);
