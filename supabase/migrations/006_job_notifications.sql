-- Migration date: 2026-09-15
-- Notifications for long-running jobs (V12-PERF-1).
--
-- A pipeline or model job runs on the server and can take minutes, so a member
-- typically leaves the page. When it settles the worker reaches them two ways:
--   * email  — one message to the address on the account (Resend);
--   * push   — a Web Push notification to any browser they've subscribed with.
-- Both are opt-out and on by default; both no-op when their env is unset.
--
-- Idempotent: safe to run on a database that already has migration 002.

-- ── Per-member preferences ──────────────────────────────────────────────────
alter table public.profiles
  add column if not exists notify_job_email boolean not null default true;
alter table public.profiles
  add column if not exists notify_job_push boolean not null default true;

comment on column public.profiles.notify_job_email is
  'Email this member when one of their jobs finishes or fails. Opt-out; on by default.';
comment on column public.profiles.notify_job_push is
  'Send a Web Push notification when one of their jobs settles. Opt-out; on by default.';

-- ── Web Push subscriptions ──────────────────────────────────────────────────
-- One row per browser/device a member has granted notification permission on.
-- The endpoint is the push service's URL for that browser; keys encrypt the
-- payload. A member can hold several (laptop, phone), so endpoint is the key.
create table if not exists public.push_subscriptions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  endpoint text not null unique,
  p256dh text not null,
  auth text not null,
  user_agent text,
  created_at timestamptz not null default now()
);

create index if not exists push_subscriptions_user_idx
  on public.push_subscriptions (user_id);

alter table public.push_subscriptions enable row level security;

-- A member manages exactly their own subscriptions from the browser; the worker
-- reads them past RLS with the service role to send.
drop policy if exists "manage own push subscriptions" on public.push_subscriptions;
create policy "manage own push subscriptions" on public.push_subscriptions
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);
