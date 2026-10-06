-- WANT-17: member-defined "allowed areas" as named sets (private or club-shared). Not applied automatically.
-- Layered over public PAD-US access_areas. collecting values are OWNER-ASSERTED, so plain 'allowed' is valid here.

-- No club/org concept existed before this migration: minimal tables.
CREATE TABLE IF NOT EXISTS clubs (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS club_members (
  club_id BIGINT NOT NULL REFERENCES clubs(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL DEFAULT 'member' CHECK (role IN ('owner','admin','member')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (club_id, user_id)
);
CREATE INDEX IF NOT EXISTS club_members_user_idx ON club_members (user_id);

CREATE TABLE IF NOT EXISTS access_area_sets (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  scope TEXT NOT NULL CHECK (scope IN ('user','club')),
  owner_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  club_id BIGINT REFERENCES clubs(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK ((scope = 'club') = (club_id IS NOT NULL))
);
CREATE INDEX IF NOT EXISTS access_area_sets_owner_idx ON access_area_sets (owner_id);
CREATE INDEX IF NOT EXISTS access_area_sets_club_idx ON access_area_sets (club_id);

CREATE TABLE IF NOT EXISTS access_set_areas (
  id BIGSERIAL PRIMARY KEY,
  set_id BIGINT NOT NULL REFERENCES access_area_sets(id) ON DELETE CASCADE,
  name TEXT,
  geom geometry(MultiPolygon, 4326) NOT NULL,
  fee_status TEXT NOT NULL DEFAULT 'unknown' CHECK (fee_status IN ('free','fee','unknown')),
  collecting TEXT NOT NULL DEFAULT 'unknown'
    CHECK (collecting IN ('allowed','likely_allowed','restricted','prohibited','unknown')),
  notes TEXT,
  created_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS access_set_areas_geom_idx ON access_set_areas USING GIST (geom);
CREATE INDEX IF NOT EXISTS access_set_areas_set_idx ON access_set_areas (set_id);

-- Membership helpers (SECURITY DEFINER avoids RLS recursion on club_members).
CREATE OR REPLACE FUNCTION is_club_member(p_club BIGINT) RETURNS BOOLEAN
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT EXISTS (SELECT 1 FROM club_members WHERE club_id = p_club AND user_id = auth.uid())
$$;
CREATE OR REPLACE FUNCTION is_club_admin(p_club BIGINT) RETURNS BOOLEAN
LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public AS $$
  SELECT EXISTS (SELECT 1 FROM club_members WHERE club_id = p_club AND user_id = auth.uid()
    AND role IN ('owner','admin'))
$$;

ALTER TABLE clubs ENABLE ROW LEVEL SECURITY;
ALTER TABLE club_members ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_area_sets ENABLE ROW LEVEL SECURITY;
ALTER TABLE access_set_areas ENABLE ROW LEVEL SECURITY;

-- Clubs and memberships: members read; writes go through the service role (netlify/functions/access-sets.mjs).
DROP POLICY IF EXISTS "clubs member read" ON clubs;
CREATE POLICY "clubs member read" ON clubs FOR SELECT USING (is_club_member(id));
DROP POLICY IF EXISTS "club_members member read" ON club_members;
CREATE POLICY "club_members member read" ON club_members FOR SELECT USING (is_club_member(club_id));

-- Sets. user scope: owner only. club scope: members read; owner/admin (and the set's creator while a member) write.
DROP POLICY IF EXISTS "access_area_sets read" ON access_area_sets;
CREATE POLICY "access_area_sets read" ON access_area_sets FOR SELECT USING (
  (scope = 'user' AND owner_id = auth.uid()) OR (scope = 'club' AND is_club_member(club_id)));
DROP POLICY IF EXISTS "access_area_sets write" ON access_area_sets;
CREATE POLICY "access_area_sets write" ON access_area_sets FOR ALL
  USING ((scope = 'user' AND owner_id = auth.uid())
    OR (scope = 'club' AND (is_club_admin(club_id) OR (owner_id = auth.uid() AND is_club_member(club_id)))))
  WITH CHECK ((scope = 'user' AND owner_id = auth.uid())
    OR (scope = 'club' AND (is_club_admin(club_id) OR (owner_id = auth.uid() AND is_club_member(club_id)))));

DROP POLICY IF EXISTS "access_set_areas read" ON access_set_areas;
CREATE POLICY "access_set_areas read" ON access_set_areas FOR SELECT USING (EXISTS (
  SELECT 1 FROM access_area_sets s WHERE s.id = set_id
    AND ((s.scope = 'user' AND s.owner_id = auth.uid()) OR (s.scope = 'club' AND is_club_member(s.club_id)))));
DROP POLICY IF EXISTS "access_set_areas write" ON access_set_areas;
CREATE POLICY "access_set_areas write" ON access_set_areas FOR ALL
  USING (EXISTS (SELECT 1 FROM access_area_sets s WHERE s.id = set_id AND (
    (s.scope = 'user' AND s.owner_id = auth.uid())
    OR (s.scope = 'club' AND (is_club_admin(s.club_id)
      OR ((s.owner_id = auth.uid() OR created_by = auth.uid()) AND is_club_member(s.club_id)))))))
  WITH CHECK (EXISTS (SELECT 1 FROM access_area_sets s WHERE s.id = set_id AND (
    (s.scope = 'user' AND s.owner_id = auth.uid())
    OR (s.scope = 'club' AND (is_club_admin(s.club_id)
      OR ((s.owner_id = auth.uid() OR created_by = auth.uid()) AND is_club_member(s.club_id)))))));

-- Insert a validated GeoJSON geometry (service role; the endpoint authorizes first).
CREATE OR REPLACE FUNCTION access_set_area_insert(
  p_set_id BIGINT, p_name TEXT, p_geom JSONB, p_fee TEXT, p_collecting TEXT, p_notes TEXT, p_user UUID
) RETURNS BIGINT LANGUAGE sql AS $$
  INSERT INTO access_set_areas (set_id, name, geom, fee_status, collecting, notes, created_by)
  VALUES (p_set_id, p_name,
    ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(p_geom::text), 4326)), 3)),
    p_fee, p_collecting, p_notes, p_user)
  RETURNING id
$$;
CREATE OR REPLACE FUNCTION access_set_area_update_geom(p_id BIGINT, p_geom JSONB) RETURNS VOID
LANGUAGE sql AS $$
  UPDATE access_set_areas
  SET geom = ST_Multi(ST_CollectionExtract(ST_MakeValid(ST_SetSRID(ST_GeomFromGeoJSON(p_geom::text), 4326)), 3))
  WHERE id = p_id
$$;

-- Bbox read for ONE caller's visible sets (service role; filters by p_user itself).
CREATE OR REPLACE FUNCTION access_set_areas_in_bbox(
  p_user UUID, p_w FLOAT8, p_s FLOAT8, p_e FLOAT8, p_n FLOAT8, p_tol FLOAT8, p_limit INTEGER
) RETURNS JSONB LANGUAGE sql STABLE AS $$
  SELECT COALESCE(jsonb_agg(x), '[]'::jsonb) FROM (
    SELECT a.id, a.name, a.fee_status, a.collecting, a.notes, a.set_id, s.name AS set_name, s.scope,
      ST_AsGeoJSON(ST_SimplifyPreserveTopology(a.geom, p_tol), 5)::jsonb AS geometry
    FROM access_set_areas a
    JOIN access_area_sets s ON s.id = a.set_id
    WHERE a.geom && ST_MakeEnvelope(p_w, p_s, p_e, p_n, 4326)
      AND ((s.scope = 'user' AND s.owner_id = p_user)
        OR (s.scope = 'club' AND EXISTS (SELECT 1 FROM club_members m WHERE m.club_id = s.club_id AND m.user_id = p_user)))
    ORDER BY a.id LIMIT p_limit
  ) x
$$;

-- All areas of ONE set as GeoJSON (service role; the endpoint authorizes the set first).
CREATE OR REPLACE FUNCTION access_set_areas_geojson(p_set_id BIGINT) RETURNS JSONB
LANGUAGE sql STABLE AS $$
  SELECT COALESCE(jsonb_agg(x ORDER BY x.id), '[]'::jsonb) FROM (
    SELECT a.id, a.name, a.fee_status, a.collecting, a.notes, a.created_by,
      ST_AsGeoJSON(a.geom, 6)::jsonb AS geometry
    FROM access_set_areas a WHERE a.set_id = p_set_id
  ) x
$$;

-- PUBLIC is a keyword, not a role; anon/authenticated/service_role may not exist on a
-- plain Postgres, so they are checked first (same pattern as 002).
REVOKE EXECUTE ON FUNCTION access_set_area_insert(BIGINT, TEXT, JSONB, TEXT, TEXT, TEXT, UUID) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION access_set_area_update_geom(BIGINT, JSONB) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION access_set_areas_in_bbox(UUID, FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8, INTEGER) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION access_set_areas_geojson(BIGINT) FROM PUBLIC;
DO $$
DECLARE fn TEXT; api_role TEXT;
BEGIN
  FOREACH fn IN ARRAY ARRAY['access_set_area_insert(BIGINT, TEXT, JSONB, TEXT, TEXT, TEXT, UUID)', 'access_set_area_update_geom(BIGINT, JSONB)', 'access_set_areas_in_bbox(UUID, FLOAT8, FLOAT8, FLOAT8, FLOAT8, FLOAT8, INTEGER)', 'access_set_areas_geojson(BIGINT)'] LOOP
    FOREACH api_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
      IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = api_role) THEN
        EXECUTE format('REVOKE EXECUTE ON FUNCTION %s FROM %I', fn, api_role);
      END IF;
    END LOOP;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
      EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO service_role', fn);
    END IF;
  END LOOP;
END $$;
