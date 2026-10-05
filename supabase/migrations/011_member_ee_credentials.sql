-- Migration date: 2026-10-05
-- Member-supplied Earth Engine credentials (WANT-3).
--
-- One encrypted service-account key per member. The ciphertext is AES-256-GCM
-- under EE_CREDENTIAL_KEY (environment, not the database). RLS is enabled with
-- NO policies, so neither anon nor authenticated can read or write the table;
-- only the service role (the Netlify functions) can. The browser sees a summary
-- built by the function, never the key.
--
-- Idempotent.

create table if not exists public.member_ee_credentials (
  user_id uuid primary key references auth.users(id) on delete cascade,
  ciphertext text not null,
  client_email text not null,
  project_id text not null,
  validated_at timestamptz not null default now(),
  created_at timestamptz not null default now()
);

comment on table public.member_ee_credentials is
  'Encrypted member Earth Engine service-account keys. Service-role access only; no RLS policies by design.';

alter table public.member_ee_credentials enable row level security;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon') then
    revoke all on public.member_ee_credentials from anon;
  end if;
  if exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on public.member_ee_credentials from authenticated;
  end if;
end $$;
