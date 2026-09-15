-- Membership granted from outside the app, by an automation.
--
-- Run after 003_custom_ee_layers.sql. Safe to run more than once.
--
--   psql "$DATABASE_URL" -f supabase_migrations/004_membership_api.sql
--
-- The caller is a webhook: a PayPal payment on the FRMS site fires an
-- automation, which calls netlify/functions/membership.mjs, which lands here.
-- Two facts about that caller drive this whole file.
--
-- It retries. Payment processors redeliver a webhook until they get a 2xx, and
-- a delivery that timed out after succeeding looks exactly like one that
-- failed. So a grant carries the processor's own reference, that reference is
-- unique, and a repeat returns the first outcome instead of selling a second
-- year for one payment.
--
-- It arrives early. Somebody pays and then makes an account, often days later
-- and sometimes never. A grant for an address with no account is therefore
-- recorded rather than refused, and applied when that address signs up.

do $$
begin
  if to_regprocedure('public.is_admin()') is null then
    raise exception using
      message = 'Migration 002 has not been applied in full: is_admin() missing.',
      hint = 'Run supabase_migrations/002_membership_jobs_and_admin.sql, then 003, then this file.';
  end if;
end
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- The grants themselves
-- ─────────────────────────────────────────────────────────────────────────────

create table if not exists public.membership_grants (
  id uuid primary key default gen_random_uuid(),

  -- Lower-cased by the API before it gets here. This is the only link between
  -- a payment and an account, so it is the one field that must match exactly.
  email text not null,

  -- 'free' is absent on purpose: removing a membership is a revocation, which
  -- has its own path and leaves no grant behind.
  tier text not null default 'member' check (tier in ('member', 'perpetual', 'admin')),
  -- One of these, never both: a duration to add, or an end date decided
  -- elsewhere. The API refuses a grant carrying both.
  months integer check (months is null or (months >= 1 and months <= 120)),
  until timestamptz,

  -- The payment processor's own transaction id. UNIQUE is what makes a
  -- redelivered webhook harmless, so it is a constraint rather than a check in
  -- application code that a concurrent retry could race past.
  ref text unique,
  source text not null default 'api',
  note text,

  -- Filled in when the grant reaches an account. Null user_id with a null
  -- applied_at is a grant still waiting for its owner to sign up.
  user_id uuid references auth.users(id) on delete set null,
  applied_at timestamptz,
  member_until timestamptz,

  created_at timestamptz not null default now()
);

-- Re-applied for a database where this file already ran: "create table if not
-- exists" leaves an existing table, and its constraint, exactly as it was.
alter table public.membership_grants drop constraint if exists membership_grants_tier_check;
alter table public.membership_grants add constraint membership_grants_tier_check
  check (tier in ('member', 'perpetual', 'admin'));

create index if not exists membership_grants_email_idx
  on public.membership_grants (lower(email));
-- The pending queue: what the signup trigger below looks through.
create index if not exists membership_grants_pending_idx
  on public.membership_grants (lower(email)) where applied_at is null;

alter table public.membership_grants enable row level security;

-- Only an administrator reads these from the browser. The API reaches them with
-- the service role, which bypasses RLS — so there is deliberately no policy
-- letting a member see what anybody paid.
drop policy if exists "admins read grants" on public.membership_grants;
create policy "admins read grants" on public.membership_grants
  for select using (public.is_admin());

-- ─────────────────────────────────────────────────────────────────────────────
-- Email → account
-- ─────────────────────────────────────────────────────────────────────────────
--
-- profiles has no email; auth.users does, and that schema is not reachable
-- through PostgREST. A security-definer function is the narrow way across:
-- it answers exactly one question and returns nothing else about the user.
--
-- Not granted to anon or authenticated. Left to the service role only, so it
-- cannot be used from a browser to test whether an address has an account.

create or replace function public.user_id_for_email(p_email text)
returns uuid
language sql
security definer
stable
set search_path = public, auth
as $$
  select id from auth.users
  where lower(email) = lower(trim(p_email))
  order by created_at
  limit 1;
$$;

do $$
declare
  api_role text;
begin
  foreach api_role in array array['authenticated', 'anon'] loop
    if exists (select 1 from pg_roles where rolname = api_role) then
      execute format('revoke execute on function public.user_id_for_email(text) from %I', api_role);
    end if;
  end loop;
  revoke execute on function public.user_id_for_email(text) from public;
end
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- Applying a grant that arrived before the account
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Extends the signup trigger from migration 002. Somebody who paid last week
-- and signs up today should be a member the moment they land, without an
-- administrator noticing and doing it by hand.
--
-- The term arithmetic is duplicated from netlify/lib/membership.mjs, and that
-- is deliberate: this path runs inside the database with no application in the
-- loop. It is the simpler half — a profile created a moment ago holds no
-- existing expiry to extend from, so "later of now and what they have" reduces
-- to now.

create or replace function public.apply_pending_grants()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  g record;
  new_until timestamptz;
  new_tier text;
begin
  for g in
    select * from public.membership_grants
    where applied_at is null and lower(email) = lower(new.email)
    order by created_at
  loop
    select p.member_until, p.tier into new_until, new_tier
    from public.profiles p where p.user_id = new.id;

    if g.until is not null then
      new_until := g.until;
    else
      new_until := greatest(coalesce(new_until, now()), now())
        + make_interval(months => coalesce(g.months, 12));
    end if;

    -- Defence in depth rather than a live guard: handle_new_user created this
    -- profile as 'free' a moment ago, so on the signup path this can never
    -- see a tier worth protecting. The case that matters — someone whose
    -- standing does not expire paying anyway, and being converted to a
    -- membership that runs out next year — is handled by applyGrant in
    -- netlify/lib/membership.mjs, which runs against an existing profile.
    if new_tier is null or new_tier not in ('perpetual', 'admin') then
      new_tier := g.tier;
    end if;

    update public.profiles
      set tier = new_tier, member_until = new_until
      where user_id = new.id;

    update public.membership_grants
      set user_id = new.id, applied_at = now(), member_until = new_until
      where id = g.id;
  end loop;

  return new;
end;
$$;

-- After handle_new_user, which is what created the profile this updates.
drop trigger if exists on_auth_user_created_grants on auth.users;
create trigger on_auth_user_created_grants
after insert on auth.users
for each row execute function public.apply_pending_grants();
