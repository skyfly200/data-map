-- WANT-17: local collecting rules over the estimates. Not applied automatically.
-- access_areas.collecting_source gains 'rule' (a curated county/city/state rule,
-- netlify/lib/collecting-rules.mjs), and collecting_rule names which one.

DO $$
DECLARE c TEXT;
BEGIN
  FOR c IN SELECT conname FROM pg_constraint
    WHERE conrelid = 'access_areas'::regclass AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%collecting_source%'
  LOOP
    EXECUTE format('ALTER TABLE access_areas DROP CONSTRAINT %I', c);
  END LOOP;
END $$;
ALTER TABLE access_areas
  ADD CONSTRAINT access_areas_collecting_source_check
  CHECK (collecting_source IN ('estimated','rule'));
ALTER TABLE access_areas ADD COLUMN IF NOT EXISTS collecting_rule TEXT;

-- Same as 013, carrying collecting_rule.
CREATE OR REPLACE FUNCTION access_upsert_areas(p_rows JSONB, p_region TEXT) RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE n INTEGER;
BEGIN
  INSERT INTO access_areas (source, source_id, name, manager, designation, access_class, public_access,
    manager_type, fee_status, fee_source, collecting, collecting_source, collecting_rule, region, geom)
  SELECT DISTINCT ON (r->>'source', r->>'source_id')
    COALESCE(r->>'source','padus'), r->>'source_id', r->>'name', r->>'manager', r->>'designation',
    r->>'access_class', COALESCE(r->>'public_access', r->>'access_class'), r->>'manager_type',
    COALESCE(r->>'fee_status','unknown'), r->>'fee_source',
    COALESCE(r->>'collecting','unknown'), r->>'collecting_source', r->>'collecting_rule', p_region,
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
    collecting_rule = EXCLUDED.collecting_rule,
    region = EXCLUDED.region, geom = EXCLUDED.geom, ingested_at = now();
  GET DIAGNOSTICS n = ROW_COUNT;
  RETURN n;
END $$;

-- CREATE OR REPLACE keeps 013's grants; the guarded block repeats them in case
-- this runs where 013's did not take.
REVOKE EXECUTE ON FUNCTION access_upsert_areas(JSONB, TEXT) FROM PUBLIC;
DO $$
DECLARE api_role TEXT;
BEGIN
  FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
      EXECUTE format('REVOKE EXECUTE ON FUNCTION access_upsert_areas(JSONB, TEXT) FROM %I', api_role);
    END IF;
  END LOOP;
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    GRANT EXECUTE ON FUNCTION access_upsert_areas(JSONB, TEXT) TO service_role;
  END IF;
END $$;

CREATE OR REPLACE FUNCTION access_areas_in_bbox(
  p_w FLOAT8, p_s FLOAT8, p_e FLOAT8, p_n FLOAT8,
  p_free BOOLEAN, p_public BOOLEAN, p_collecting TEXT, p_tol FLOAT8, p_limit INTEGER
) RETURNS JSONB LANGUAGE sql STABLE AS $$
  SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
    SELECT id, name, manager_type, public_access, fee_status, fee_source, collecting, collecting_source,
      collecting_rule,
      ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom, p_tol), 5)::jsonb AS geometry
    FROM access_areas
    WHERE geom && ST_MakeEnvelope(p_w, p_s, p_e, p_n, 4326)
      AND (NOT p_free OR fee_status = 'free')
      AND (NOT p_public OR public_access = 'open')
      AND (p_collecting IS NULL OR collecting = p_collecting)
    ORDER BY id LIMIT p_limit
  ) x
$$;
