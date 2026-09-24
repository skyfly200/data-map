-- Add precise_only flag to model_configs.
-- When true the EE worker discards observations whose location_precision is
-- not 'precise' before fitting the model. Defaults to true so new runs are
-- safe by default and existing configs (which were always trained on all
-- points) stay at false to preserve their documented behaviour.
ALTER TABLE model_configs
  ADD COLUMN IF NOT EXISTS precise_only BOOLEAN NOT NULL DEFAULT false;
