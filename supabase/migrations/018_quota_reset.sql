-- An admin's "reset quota" button. Usage is summed from ee_jobs rather than kept
-- in a counter (see ee_usage_month in 002), so resetting it means moving the
-- start of the window, not zeroing a number: jobs created before this moment no
-- longer count against the member's monthly units, monthly job count or daily
-- job count. The job rows themselves are left alone as the record of what ran.
-- Not applied automatically.

alter table public.profiles
  add column if not exists quota_reset_at timestamptz;
