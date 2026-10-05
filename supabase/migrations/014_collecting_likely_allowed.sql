-- WANT-17: allow estimated 'likely_allowed' (BLM/USFS only) in access_areas.collecting.
DO $$
DECLARE c TEXT;
BEGIN
  FOR c IN SELECT conname FROM pg_constraint
    WHERE conrelid = 'access_areas'::regclass AND contype = 'c'
      AND pg_get_constraintdef(oid) ILIKE '%collecting%' AND pg_get_constraintdef(oid) NOT ILIKE '%collecting_source%'
  LOOP
    EXECUTE format('ALTER TABLE access_areas DROP CONSTRAINT %I', c);
  END LOOP;
END $$;
ALTER TABLE access_areas
  ADD CONSTRAINT access_areas_collecting_check
  CHECK (collecting IN ('allowed','likely_allowed','restricted','prohibited','unknown'));
