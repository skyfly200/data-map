-- Rename saved_datasets → datasets.
-- The old name was a legacy holdover; "datasets" is what the product calls them.

ALTER TABLE public.saved_datasets RENAME TO datasets;

-- Rename dependent indexes.
ALTER INDEX IF EXISTS saved_datasets_visibility_idx RENAME TO datasets_visibility_idx;
ALTER INDEX IF EXISTS saved_datasets_owner_idx      RENAME TO datasets_owner_idx;

-- Rename the trigger so it stays traceable.
ALTER TRIGGER saved_datasets_set_updated_at ON public.datasets
  RENAME TO datasets_set_updated_at;

-- Rename the path ownership trigger added in migration 005.
ALTER TRIGGER saved_datasets_path_owned ON public.datasets
  RENAME TO datasets_path_owned;

-- Update the FK on model_configs that points at saved_datasets(id).
-- The FK constraint itself targets the table by OID so the rename is
-- enough; just rename the constraint to match.
ALTER TABLE public.model_configs
  RENAME CONSTRAINT model_configs_source_dataset_id_fkey
                 TO model_configs_source_dataset_id_datasets_fkey;
