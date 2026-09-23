-- Migration date: 2026-09-22
-- Cron job run log. Each scheduled function writes one row per invocation
-- (or per claimed job for ee-worker, which runs every minute). Kept for 90
-- days; older rows are pruned by the admin endpoint on read.

create table if not exists cron_logs (
  id          bigserial primary key,
  job_id      text        not null,
  fired_at    timestamptz not null default now(),
  duration_ms integer,
  status      text        not null check (status in ('ok', 'error')),
  details     jsonb       not null default '{}'
);

create index if not exists cron_logs_job_fired
  on cron_logs (job_id, fired_at desc);

-- Only the service role may write; anyone authenticated may read (admin check
-- is enforced at the function layer, not here).
alter table cron_logs enable row level security;

create policy "service role full access"
  on cron_logs for all
  using (auth.role() = 'service_role')
  with check (auth.role() = 'service_role');

create policy "authenticated read"
  on cron_logs for select
  using (auth.role() = 'authenticated');
