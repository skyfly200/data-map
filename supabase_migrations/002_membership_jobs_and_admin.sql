-- Membership tiers, the Earth Engine job queue, and the admin surface.
--
-- Run after 001_user_settings_and_charts.sql. Safe to run more than once.
--
--   psql "$DATABASE_URL" -f supabase_migrations/002_membership_jobs_and_admin.sql
--
-- Three things are set up here:
--
--   profiles        one row per account: tier, when membership lapses, and the
--                   quotas and rate limits an admin sets per member.
--   ee_jobs         the queue. Members submit, a worker claims and reports
--                   progress, and the output is read back when it finishes.
--   saved_datasets  results promoted to something reusable, with a visibility
--                   an admin controls.
--
-- The security rule running through all of it: a member's browser holds the
-- anon key, so anything the browser can write, a member can write. Tier,
-- quotas, and job results are therefore writable only by the service role. The
-- browser reads them and never sets them.

-- ─────────────────────────────────────────────────────────────────────────────
-- Profiles
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  -- free: signed in, reads the shipped data. member: dues paid, may run jobs.
  -- admin: may also manage other people's quotas and datasets.
  tier text not null default 'free' check (tier in ('free', 'member', 'admin')),
  display_name text,
  -- Null means "no expiry". A date in the past is treated as 'free' by the
  -- token hook below, so a lapsed membership needs no sweep job to enforce it.
  member_until timestamptz,

  -- Quotas and rate limits, per member, set by an admin. Earth Engine bills the
  -- project rather than the caller, so without these one member's heavy query
  -- degrades the platform for everyone.
  ee_quota_monthly integer not null default 500,   -- cost units per calendar month
  ee_jobs_per_day integer not null default 20,
  ee_max_points integer not null default 5000,     -- points in a single job
  ee_max_concurrent integer not null default 1,

  notes text,                                      -- admin-only scratch
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Every account gets a profile the moment it exists, so nothing downstream has
-- to cope with a signed-in user who has no row.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (user_id, display_name)
  values (new.id, new.raw_user_meta_data->>'full_name')
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

-- Backfill anyone who signed up before this migration ran.
insert into public.profiles (user_id)
select id from auth.users
on conflict (user_id) do nothing;

-- ─────────────────────────────────────────────────────────────────────────────
-- The custom access token hook: tier travels in the JWT
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Supabase calls this while minting an access token. Stamping the tier into the
-- claims means the API can authorize a request by reading the token it already
-- has to verify, instead of a database round trip on every call.
--
-- The cost is staleness: claims are fixed until the token refreshes, so a tier
-- change takes up to one refresh interval (an hour by default) to be felt. That
-- is fine for showing and hiding UI and for gating cheap reads. It is NOT fine
-- for anything that spends money, so the job submission path re-reads this
-- table — it has to look up the quota anyway, which makes the check free.
--
-- Enable it afterwards in Dashboard → Authentication → Hooks → Custom Access
-- Token, pointing at public.custom_access_token_hook. Until it is enabled the
-- claim is simply absent and everything falls back to 'free'.

create or replace function public.custom_access_token_hook(event jsonb)
returns jsonb
language plpgsql
stable
as $$
declare
  claims jsonb;
  found_tier text;
  lapses timestamptz;
begin
  select tier, member_until into found_tier, lapses
  from public.profiles
  where user_id = (event->>'user_id')::uuid;

  -- No row yet (the trigger races a first sign-in), or a membership that has
  -- run out: both are a free account.
  if found_tier is null then
    found_tier := 'free';
  elsif lapses is not null and lapses < now() then
    found_tier := 'free';
  end if;

  claims := coalesce(event->'claims', '{}'::jsonb);

  -- app_metadata is the right home for it: PostgREST exposes it to RLS, and
  -- unlike user_metadata it is not writable by the account holder.
  if claims->'app_metadata' is null or jsonb_typeof(claims->'app_metadata') <> 'object' then
    claims := jsonb_set(claims, '{app_metadata}', '{}'::jsonb);
  end if;
  claims := jsonb_set(claims, '{app_metadata,tier}', to_jsonb(found_tier));

  return jsonb_set(event, '{claims}', claims);
end;
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- Reading the tier back out, inside the database
-- ─────────────────────────────────────────────────────────────────────────────
--
-- These come BEFORE the grants below, deliberately. Everything downstream
-- depends on them — the row-level security policies in this file, and the whole
-- of migration 003 — while the grants depend on a role that is not present on
-- every deployment. Putting the fragile statement first meant one failed grant
-- aborted the script and left the database with tables but no is_member(),
-- which surfaced later as "function public.is_member() does not exist" from a
-- migration that looked unrelated.

-- The tier carried by the current request's token. Used by the policies below
-- so they never have to join back to profiles, which would recurse: reading a
-- profile is exactly what the policy is deciding about.
create or replace function public.current_tier()
returns text
language sql
stable
as $$
  select coalesce(
    nullif(current_setting('request.jwt.claims', true), '')::jsonb
      -> 'app_metadata' ->> 'tier',
    'free'
  );
$$;

create or replace function public.is_admin()
returns boolean
language sql
stable
as $$
  select public.current_tier() = 'admin';
$$;

create or replace function public.is_member()
returns boolean
language sql
stable
as $$
  select public.current_tier() in ('member', 'admin');
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- Letting the auth service call the hook
-- ─────────────────────────────────────────────────────────────────────────────
--
-- The hook runs as supabase_auth_admin, a role hosted Supabase provides and a
-- self-hosted or older project may not. Guarded rather than assumed: a missing
-- role should cost the token hook, which can be fixed later, not the entire
-- migration and everything defined after it.
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'supabase_auth_admin') then
    grant usage on schema public to supabase_auth_admin;
    grant execute on function public.custom_access_token_hook(jsonb) to supabase_auth_admin;
    grant select on public.profiles to supabase_auth_admin;
  else
    raise notice 'No supabase_auth_admin role: skipping the access token hook grants. %',
      'Tiers will read as free until the hook can be enabled.';
  end if;
exception
  when insufficient_privilege then
    raise notice 'Not permitted to grant to supabase_auth_admin; run those grants as a superuser.';
end
$$;

-- Nobody else may call the hook: it is not a secret, but an API role that can
-- invoke it gains nothing and the smaller surface is free.
--
-- Guarded for the same reason as the block above. authenticated and anon are
-- PostgREST's roles, present on hosted Supabase and absent on a plain Postgres,
-- and naming an absent role in a revoke is an error like any other.
do $$
declare
  api_role text;
begin
  foreach api_role in array array['authenticated', 'anon'] loop
    if exists (select 1 from pg_roles where rolname = api_role) then
      execute format(
        'revoke execute on function public.custom_access_token_hook(jsonb) from %I', api_role);
    end if;
  end loop;
  -- PUBLIC is not a role but a keyword, so it needs no guard and covers any
  -- role the deployment has that the two above do not.
  revoke execute on function public.custom_access_token_hook(jsonb) from public;
end
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- The job queue
-- ─────────────────────────────────────────────────────────────────────────────

do $$
begin
  if not exists (select 1 from pg_type where typname = 'ee_job_status') then
    create type public.ee_job_status as enum
      ('queued', 'running', 'succeeded', 'failed', 'cancelled');
  end if;
end
$$;

create table if not exists public.ee_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,

  -- Which pipeline to run. Checked against an allowlist in the API, because a
  -- job kind chooses which Earth Engine assets get touched.
  kind text not null,
  -- The parameters, already validated and normalised by the API. Members never
  -- send Earth Engine expressions, only values that select from an allowlist:
  -- arbitrary EE code from a browser is arbitrary compute on our billing.
  params jsonb not null default '{}'::jsonb,
  title text,

  status public.ee_job_status not null default 'queued',
  -- 0..1. Written by the worker as it finishes each stage, so a member watching
  -- a ten-minute job sees it move rather than guessing whether it is stuck.
  progress real not null default 0 check (progress >= 0 and progress <= 1),
  stage text,
  message text,

  -- What it cost, counted against the monthly quota. Written when the job
  -- finishes; estimated up front to refuse a job that cannot fit.
  cost_units integer not null default 0,
  estimated_units integer not null default 0,

  result_path text,                        -- storage path of the output GeoJSON
  result_meta jsonb,                       -- feature count, bands, bounds
  error text,
  attempts integer not null default 0,

  -- Held by a worker while it runs, so two workers cannot claim one job.
  locked_by text,
  locked_at timestamptz,

  created_at timestamptz not null default now(),
  started_at timestamptz,
  finished_at timestamptz,
  updated_at timestamptz not null default now()
);

create index if not exists ee_jobs_user_idx on public.ee_jobs (user_id, created_at desc);
-- The worker's claim query: oldest queued job first.
create index if not exists ee_jobs_queue_idx on public.ee_jobs (status, created_at)
  where status in ('queued', 'running');

-- Usage per member per month, which is what the quota is actually about.
-- A view rather than a counter column: a counter drifts the first time a job
-- fails halfway, and this cannot.
create or replace view public.ee_usage_month
with (security_invoker = true) as
  select user_id,
         date_trunc('month', created_at) as month,
         count(*) as jobs,
         coalesce(sum(cost_units), 0) as units
  from public.ee_jobs
  where status in ('running', 'succeeded')
  group by user_id, date_trunc('month', created_at);

-- ─────────────────────────────────────────────────────────────────────────────
-- Saved datasets
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.saved_datasets (
  id uuid primary key default gen_random_uuid(),
  -- Kept when the owner's account goes away: a society's shared dataset should
  -- outlive the membership of whoever happened to generate it.
  owner_id uuid references auth.users(id) on delete set null,
  job_id uuid references public.ee_jobs(id) on delete set null,

  slug text not null unique,
  title text not null,
  description text,
  path text not null,                      -- storage path of the GeoJSON

  -- private: the owner and admins. members: anyone with dues paid. public: the
  -- open web, for the society's published work.
  visibility text not null default 'private'
    check (visibility in ('private', 'members', 'public')),

  feature_count integer,
  bytes bigint,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists saved_datasets_visibility_idx on public.saved_datasets (visibility);
create index if not exists saved_datasets_owner_idx on public.saved_datasets (owner_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Row-level security
-- ─────────────────────────────────────────────────────────────────────────────

alter table public.profiles enable row level security;
alter table public.ee_jobs enable row level security;
alter table public.saved_datasets enable row level security;

-- Profiles: you may read yours; an admin may read and change anyone's.
-- There is deliberately no self-update policy. A self-writable tier column is a
-- free membership button, and a self-writable quota column is an unmetered one.
drop policy if exists "read own profile" on public.profiles;
create policy "read own profile" on public.profiles
  for select using (auth.uid() = user_id or public.is_admin());

drop policy if exists "admins manage profiles" on public.profiles;
create policy "admins manage profiles" on public.profiles
  for update using (public.is_admin()) with check (public.is_admin());

-- The token hook reads this table as supabase_auth_admin, which RLS would
-- otherwise block, leaving every token stamped 'free'. Guarded like the grants
-- above: naming a role that does not exist aborts the script, and everything
-- below this line is more important than the hook.
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'supabase_auth_admin') then
    drop policy if exists "auth admin reads profiles" on public.profiles;
    create policy "auth admin reads profiles" on public.profiles
      as permissive for select to supabase_auth_admin using (true);
  end if;
end
$$;

-- Jobs: you see your own, an admin sees all. Nobody inserts from the browser —
-- submission goes through the API, which checks quota and normalises params
-- first, and an insert policy here would be a way around both.
drop policy if exists "read own jobs" on public.ee_jobs;
create policy "read own jobs" on public.ee_jobs
  for select using (auth.uid() = user_id or public.is_admin());

-- Cancelling is the one thing a member may write, and only on a job of theirs
-- that has not finished.
drop policy if exists "cancel own jobs" on public.ee_jobs;
create policy "cancel own jobs" on public.ee_jobs
  for update using (auth.uid() = user_id and status in ('queued', 'running'))
  with check (auth.uid() = user_id and status = 'cancelled');

-- Datasets: your own, plus whatever your tier entitles you to see.
drop policy if exists "read visible datasets" on public.saved_datasets;
create policy "read visible datasets" on public.saved_datasets
  for select using (
    visibility = 'public'
    or (visibility = 'members' and public.is_member())
    or owner_id = auth.uid()
    or public.is_admin()
  );

drop policy if exists "admins manage datasets" on public.saved_datasets;
create policy "admins manage datasets" on public.saved_datasets
  for all using (public.is_admin()) with check (public.is_admin());

-- ─────────────────────────────────────────────────────────────────────────────
-- updated_at triggers
-- ─────────────────────────────────────────────────────────────────────────────

create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

drop trigger if exists profiles_set_updated_at on public.profiles;
create trigger profiles_set_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

drop trigger if exists ee_jobs_set_updated_at on public.ee_jobs;
create trigger ee_jobs_set_updated_at
before update on public.ee_jobs
for each row execute function public.set_updated_at();

drop trigger if exists saved_datasets_set_updated_at on public.saved_datasets;
create trigger saved_datasets_set_updated_at
before update on public.saved_datasets
for each row execute function public.set_updated_at();

-- ─────────────────────────────────────────────────────────────────────────────
-- Making the first admin
-- ─────────────────────────────────────────────────────────────────────────────
-- There is no way to do this from the app, by design: the admin surface is
-- guarded by the admin tier, so the first one has to be set here. Run this once
-- with your own address, then sign out and back in to pick up the new claim.
--
--   update public.profiles set tier = 'admin'
--   where user_id = (select id from auth.users where email = 'you@example.org');
