-- Add archive support to ee_jobs.
-- Archived jobs stay in the database but are hidden from the main list.
-- The default list query filters to archived_at IS NULL.

ALTER TABLE ee_jobs
  ADD COLUMN IF NOT EXISTS archived_at timestamptz;

-- Sparse index: only unarchived rows are indexed, so the main-list query stays fast.
CREATE INDEX IF NOT EXISTS ee_jobs_active_idx
  ON ee_jobs (user_id, created_at DESC)
  WHERE archived_at IS NULL;
