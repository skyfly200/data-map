-- Add accuracy_m to observations so location_precision can be computed server-side
-- without parsing raw_payload in every query. Backfill from raw_payload for
-- existing iNat rows; GBIF rows carry coordinateUncertaintyInMeters in their
-- GeoJSON properties (stored in the datasets table, not here) so no backfill
-- is needed for them.
ALTER TABLE observations
  ADD COLUMN IF NOT EXISTS accuracy_m double precision;

UPDATE observations
SET accuracy_m = (raw_payload->>'public_positional_accuracy')::double precision
WHERE accuracy_m IS NULL
  AND raw_payload->>'public_positional_accuracy' IS NOT NULL;

-- Fallback to positional_accuracy (observer-visible) when public value absent
UPDATE observations
SET accuracy_m = (raw_payload->>'positional_accuracy')::double precision
WHERE accuracy_m IS NULL
  AND raw_payload->>'positional_accuracy' IS NOT NULL;

COMMENT ON COLUMN observations.accuracy_m IS
  'Coordinate uncertainty radius in metres. NULL means unknown, not precise.';
