-- WANT-17: own queryable access database. Not applied automatically.
-- Classification columns are ESTIMATES unless *_source = 'ridb'.

ALTER TABLE access_areas
  ADD COLUMN IF NOT EXISTS manager_type TEXT,
  ADD COLUMN IF NOT EXISTS public_access TEXT NOT NULL DEFAULT 'unknown'
    CHECK (public_access IN ('open','restricted','closed','unknown')),
  ADD COLUMN IF NOT EXISTS fee_status TEXT NOT NULL DEFAULT 'unknown'
    CHECK (fee_status IN ('free','fee','unknown')),
  ADD COLUMN IF NOT EXISTS fee_source TEXT CHECK (fee_source IN ('estimated','ridb')),
  ADD COLUMN IF NOT EXISTS collecting TEXT NOT NULL DEFAULT 'unknown'
    CHECK (collecting IN ('allowed','restricted','prohibited','unknown')),
  ADD COLUMN IF NOT EXISTS collecting_source TEXT CHECK (collecting_source IN ('estimated')),
  ADD COLUMN IF NOT EXISTS region TEXT;

UPDATE access_areas SET public_access = access_class WHERE public_access = 'unknown' AND access_class <> 'unknown';
CREATE INDEX IF NOT EXISTS access_areas_region_idx ON access_areas (region);

ALTER TABLE access_lines ADD COLUMN IF NOT EXISTS region TEXT;
CREATE INDEX IF NOT EXISTS access_lines_region_idx ON access_lines (region);

-- Which regions have been loaded (so repeat loads are skipped).
CREATE TABLE IF NOT EXISTS access_regions (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  bbox JSONB NOT NULL,                       -- [west, south, east, north]
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','loading','loaded','partial','failed')),
  area_count INTEGER NOT NULL DEFAULT 0,
  line_count INTEGER NOT NULL DEFAULT 0,
  ridb_matched INTEGER NOT NULL DEFAULT 0,
  sources JSONB,
  error TEXT,
  requested_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  loaded_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE access_regions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "access_regions readable" ON access_regions;
CREATE POLICY "access_regions readable" ON access_regions FOR SELECT USING (true);
-- No insert/update/delete policies: service role only.

-- Batch upserts (service role). Keeps RIDB-sourced fee data when a later load has no RIDB key.
CREATE OR REPLACE FUNCTION access_upsert_areas(p_rows JSONB, p_region TEXT) RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE n INTEGER;
BEGIN
  INSERT INTO access_areas (source, source_id, name, manager, designation, access_class, public_access,
    manager_type, fee_status, fee_source, collecting, collecting_source, region, geom)
  SELECT DISTINCT ON (r->>'source', r->>'source_id')
    COALESCE(r->>'source','padus'), r->>'source_id', r->>'name', r->>'manager', r->>'designation',
    r->>'access_class', COALESCE(r->>'public_access', r->>'access_class'), r->>'manager_type',
    COALESCE(r->>'fee_status','unknown'), r->>'fee_source',
    COALESCE(r->>'collecting','unknown'), r->>'collecting_source', p_region,
    ST_Multi(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON((r->'geom')::text), 4326)))
  FROM jsonb_array_elements(p_rows) r
  ON CONFLICT (source, source_id) DO UPDATE SET
    name = EXCLUDED.name, manager = EXCLUDED.manager, designation = EXCLUDED.designation,
    access_class = EXCLUDED.access_class, public_access = EXCLUDED.public_access,
    manager_type = EXCLUDED.manager_type,
    fee_status = CASE WHEN access_areas.fee_source = 'ridb' AND EXCLUDED.fee_source IS DISTINCT FROM 'ridb'
                      THEN access_areas.fee_status ELSE EXCLUDED.fee_status END,
    fee_source = CASE WHEN access_areas.fee_source = 'ridb' AND EXCLUDED.fee_source IS DISTINCT FROM 'ridb'
                      THEN access_areas.fee_source ELSE EXCLUDED.fee_source END,
    collecting = EXCLUDED.collecting, collecting_source = EXCLUDED.collecting_source,
    region = EXCLUDED.region, geom = EXCLUDED.geom, ingested_at = now();
  GET DIAGNOSTICS n = ROW_COUNT;
  RETURN n;
END $$;

CREATE OR REPLACE FUNCTION access_upsert_lines(p_rows JSONB, p_region TEXT) RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE n INTEGER;
BEGIN
  INSERT INTO access_lines (osm_id, kind, highway, name, region, geom)
  SELECT DISTINCT ON ((r->>'osm_id')::bigint)
    (r->>'osm_id')::bigint, r->>'kind', r->>'highway', r->>'name', p_region,
    ST_SetSRID(ST_GeomFromGeoJSON((r->'geom')::text), 4326)
  FROM jsonb_array_elements(p_rows) r
  ON CONFLICT (osm_id) DO UPDATE SET kind = EXCLUDED.kind, highway = EXCLUDED.highway,
    name = EXCLUDED.name, region = EXCLUDED.region, geom = EXCLUDED.geom, ingested_at = now();
  GET DIAGNOSTICS n = ROW_COUNT;
  RETURN n;
END $$;

REVOKE EXECUTE ON FUNCTION access_upsert_areas(JSONB, TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION access_upsert_lines(JSONB, TEXT) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION access_upsert_areas(JSONB, TEXT) TO service_role;
GRANT EXECUTE ON FUNCTION access_upsert_lines(JSONB, TEXT) TO service_role;

-- Read helpers for netlify/functions/access.mjs (simplified GeoJSON geometry).
CREATE OR REPLACE FUNCTION access_areas_in_bbox(
  p_w FLOAT8, p_s FLOAT8, p_e FLOAT8, p_n FLOAT8,
  p_free BOOLEAN, p_public BOOLEAN, p_collecting TEXT, p_tol FLOAT8, p_limit INTEGER
) RETURNS JSONB LANGUAGE sql STABLE AS $$
  SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
    SELECT id, name, manager_type, public_access, fee_status, fee_source, collecting, collecting_source,
      ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, p_tol), 5)::jsonb AS geometry
    FROM access_areas
    WHERE geom && ST_MakeEnvelope(p_w, p_s, p_e, p_n, 4326)
      AND (NOT p_free OR fee_status = 'free')
      AND (NOT p_public OR public_access = 'open')
      AND (p_collecting IS NULL OR collecting = p_collecting)
    ORDER BY id LIMIT p_limit
  ) x
$$;

CREATE OR REPLACE FUNCTION access_lines_in_bbox(
  p_w FLOAT8, p_s FLOAT8, p_e FLOAT8, p_n FLOAT8, p_tol FLOAT8, p_limit INTEGER
) RETURNS JSONB LANGUAGE sql STABLE AS $$
  SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
    SELECT id, kind, highway, name,
      ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, p_tol), 5)::jsonb AS geometry
    FROM access_lines
    WHERE geom && ST_MakeEnvelope(p_w, p_s, p_e, p_n, 4326)
    ORDER BY id LIMIT p_limit
  ) x
$$;
