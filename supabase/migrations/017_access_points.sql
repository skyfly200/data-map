-- WANT-17: parking and trailheads for the foray planner's walk-in estimate.
-- Not applied automatically. Points come from OSM (amenity=parking nodes and
-- way centres, highway=trailhead nodes) in the same `access_ingest` job as
-- access_lines; access_nearest() answers "how far to park / to a road" for
-- the planner's grid cells.

CREATE TABLE IF NOT EXISTS access_points (
  id BIGSERIAL PRIMARY KEY,
  osm_type TEXT NOT NULL CHECK (osm_type IN ('node','way','relation')),
  osm_id BIGINT NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('parking','trailhead')),
  name TEXT,
  fee TEXT,                                  -- OSM fee tag: 'yes'|'no'|NULL
  wheelchair TEXT,                           -- OSM wheelchair tag: 'yes'|'limited'|'no'|NULL
  disabled_spaces INTEGER,                   -- OSM capacity:disabled, when numeric
  region TEXT,
  geom geometry(Point, 4326) NOT NULL,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (osm_type, osm_id)
);
CREATE INDEX IF NOT EXISTS access_points_geom_idx ON access_points USING GIST (geom);
CREATE INDEX IF NOT EXISTS access_points_region_idx ON access_points (region);

ALTER TABLE access_points ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "access_points readable" ON access_points;
CREATE POLICY "access_points readable" ON access_points FOR SELECT USING (true);
-- No insert/update/delete policies: service role only.

ALTER TABLE access_regions ADD COLUMN IF NOT EXISTS point_count INTEGER NOT NULL DEFAULT 0;

CREATE OR REPLACE FUNCTION access_upsert_points(p_rows JSONB, p_region TEXT) RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE n INTEGER;
BEGIN
  INSERT INTO access_points (osm_type, osm_id, kind, name, fee, wheelchair, disabled_spaces, region, geom)
  SELECT DISTINCT ON (r->>'osm_type', (r->>'osm_id')::bigint)
    r->>'osm_type', (r->>'osm_id')::bigint, r->>'kind', r->>'name', r->>'fee', r->>'wheelchair',
    NULLIF(r->>'disabled_spaces', '')::integer, p_region,
    ST_SetSRID(ST_MakePoint((r->>'lon')::float8, (r->>'lat')::float8), 4326)
  FROM jsonb_array_elements(p_rows) r
  ON CONFLICT (osm_type, osm_id) DO UPDATE SET kind = EXCLUDED.kind, name = EXCLUDED.name,
    fee = EXCLUDED.fee, wheelchair = EXCLUDED.wheelchair, disabled_spaces = EXCLUDED.disabled_spaces,
    region = EXCLUDED.region, geom = EXCLUDED.geom, ingested_at = now();
  GET DIAGNOSTICS n = ROW_COUNT;
  RETURN n;
END $$;

REVOKE EXECUTE ON FUNCTION access_upsert_points(JSONB, TEXT) FROM PUBLIC;
DO $$
DECLARE api_role TEXT;
BEGIN
  FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
      EXECUTE format('REVOKE EXECUTE ON FUNCTION access_upsert_points(JSONB, TEXT) FROM %I', api_role);
    END IF;
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    GRANT EXECUTE ON FUNCTION access_upsert_points(JSONB, TEXT) TO service_role;
  END IF;
END $$;

-- For each [lon, lat] in p_points (in order): the nearest parking, nearest
-- accessible parking (wheelchair=yes or disabled spaces), nearest trailhead and
-- nearest road, each within p_max_m metres (NULL beyond). Distances are metres.
CREATE OR REPLACE FUNCTION access_nearest(p_points JSONB, p_max_m FLOAT8) RETURNS JSONB
LANGUAGE sql STABLE AS $$
  WITH pts AS (
    SELECT (ord - 1)::int AS i,
      ST_SetSRID(ST_MakePoint((p->>0)::float8, (p->>1)::float8), 4326) AS g
    FROM jsonb_array_elements(p_points) WITH ORDINALITY AS t(p, ord)
  )
  SELECT COALESCE(jsonb_agg(jsonb_build_object(
    'i', pts.i,
    'parking', pk.j, 'accessible', ac.j, 'trailhead', th.j, 'road_m', rd.m
  ) ORDER BY pts.i), '[]'::jsonb)
  FROM pts
  LEFT JOIN LATERAL (
    SELECT jsonb_build_object('m', round(ST_Distance(a.geom::geography, pts.g::geography)), 'name', a.name,
      'fee', a.fee, 'lon', ST_X(a.geom), 'lat', ST_Y(a.geom)) AS j
    FROM access_points a
    WHERE a.kind = 'parking' AND ST_DWithin(a.geom::geography, pts.g::geography, p_max_m)
    ORDER BY a.geom <-> pts.g LIMIT 1
  ) pk ON true
  LEFT JOIN LATERAL (
    SELECT jsonb_build_object('m', round(ST_Distance(a.geom::geography, pts.g::geography)), 'name', a.name,
      'lon', ST_X(a.geom), 'lat', ST_Y(a.geom)) AS j
    FROM access_points a
    WHERE a.kind = 'parking' AND (a.wheelchair = 'yes' OR COALESCE(a.disabled_spaces, 0) > 0)
      AND ST_DWithin(a.geom::geography, pts.g::geography, p_max_m)
    ORDER BY a.geom <-> pts.g LIMIT 1
  ) ac ON true
  LEFT JOIN LATERAL (
    SELECT jsonb_build_object('m', round(ST_Distance(a.geom::geography, pts.g::geography)), 'name', a.name,
      'lon', ST_X(a.geom), 'lat', ST_Y(a.geom)) AS j
    FROM access_points a
    WHERE a.kind = 'trailhead' AND ST_DWithin(a.geom::geography, pts.g::geography, p_max_m)
    ORDER BY a.geom <-> pts.g LIMIT 1
  ) th ON true
  LEFT JOIN LATERAL (
    SELECT round(ST_Distance(l.geom::geography, pts.g::geography)) AS m
    FROM access_lines l
    WHERE l.kind = 'road' AND ST_DWithin(l.geom::geography, pts.g::geography, p_max_m)
    ORDER BY l.geom <-> pts.g LIMIT 1
  ) rd ON true
$$;
