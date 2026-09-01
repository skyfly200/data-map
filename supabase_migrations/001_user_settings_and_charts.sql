-- Adds the two tables the app's account sync needs, and nothing else.
--
-- Run this against a database that already has the observation tables. It is
-- safe to run more than once.
--
-- Paste it into the Supabase dashboard → SQL Editor → Run, or:
--   psql "$DATABASE_URL" -f supabase_migrations/001_user_settings_and_charts.sql
--
-- Without these tables every sync call fails and the header shows "Sync failed":
-- PostgREST answers an unknown table with PGRST205 / "Could not find the table
-- 'public.user_settings' in the schema cache".

-- Shared by both tables' updated_at triggers. Already present in
-- supabase_schema.sql; repeated here so this file stands alone.
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;

-- Everything the app persists per viewer (appearance, chart layout, map overlay,
-- units, saved filter subsets) in one JSON blob: these are display preferences
-- that change shape as features are added, and a schema migration per toggle is
-- not worth it.
create table if not exists public.user_settings (
  user_id uuid primary key references auth.users(id) on delete cascade,
  settings jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

-- Saved charts DO get their own table — they are user-authored content that is
-- listed and reordered, so they deserve to be queryable.
create table if not exists public.saved_charts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  -- The builder's config, stored whole so an older client cannot drop fields it
  -- does not understand.
  config jsonb not null,
  title text,
  position integer not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists saved_charts_user_idx on public.saved_charts (user_id, position);

-- Row-level security: a signed-in user reaches only their own rows. Without
-- this, the public anon key would expose every user's settings to every other
-- user.
alter table public.user_settings enable row level security;
alter table public.saved_charts enable row level security;

drop policy if exists "own settings" on public.user_settings;
create policy "own settings" on public.user_settings
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

drop policy if exists "own charts" on public.saved_charts;
create policy "own charts" on public.saved_charts
  for all using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- PostgreSQL has no `create trigger if not exists`, so drop first to keep this
-- file re-runnable.
drop trigger if exists user_settings_set_updated_at on public.user_settings;
create trigger user_settings_set_updated_at
before update on public.user_settings
for each row
execute function public.set_updated_at();

drop trigger if exists saved_charts_set_updated_at on public.saved_charts;
create trigger saved_charts_set_updated_at
before update on public.saved_charts
for each row
execute function public.set_updated_at();
