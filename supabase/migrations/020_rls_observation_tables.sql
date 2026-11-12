-- Turn on row level security for tables in the exposed public schema.
--
-- Supabase's rls_disabled_in_public advisor flagged three tables. Each one had
-- full table grants for the anon and authenticated roles, so anyone holding the
-- public anon key could read, change or truncate them over the REST API.
--
--   observations, observation_enrichments
--     Nothing in the app or the Netlify functions reads them through the API:
--     they go through the service role (which bypasses RLS). Enabling RLS with
--     no policy and revoking the anon/authenticated grants closes them to the
--     API without changing anything the app does.
--
--   spatial_ref_sys (PostGIS)
--     Reference table of coordinate systems that ST_Transform reads, so it
--     needs a read policy for everyone. Its owner is supabase_admin, so the
--     ALTER and REVOKE are wrapped: on a database where we are not the owner
--     they are skipped with a notice rather than aborting the migration.
--
-- Idempotent. Hosted-only roles are checked before they are named.

do $$
begin
  if exists (select 1 from pg_tables where schemaname = 'public' and tablename = 'observations') then
    alter table public.observations enable row level security;
  end if;
  if exists (select 1 from pg_tables where schemaname = 'public' and tablename = 'observation_enrichments') then
    alter table public.observation_enrichments enable row level security;
  end if;
end $$;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon')
     and exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke all on public.observations, public.observation_enrichments from anon, authenticated;
  end if;
end $$;

do $$
begin
  if exists (select 1 from pg_tables where schemaname = 'public' and tablename = 'spatial_ref_sys') then
    alter table public.spatial_ref_sys enable row level security;
    if not exists (
      select 1 from pg_policies
      where schemaname = 'public' and tablename = 'spatial_ref_sys' and policyname = 'spatial_ref_sys_read'
    ) then
      create policy spatial_ref_sys_read on public.spatial_ref_sys
        for select to anon, authenticated, public using (true);
    end if;
  end if;
exception
  when insufficient_privilege then
    raise notice 'Skipped spatial_ref_sys RLS: this role does not own the table.';
end $$;

do $$
begin
  if exists (select 1 from pg_roles where rolname = 'anon')
     and exists (select 1 from pg_roles where rolname = 'authenticated') then
    revoke insert, update, delete, truncate on public.spatial_ref_sys from anon, authenticated;
  end if;
exception
  when insufficient_privilege then
    raise notice 'Skipped spatial_ref_sys revokes: this role does not own the table.';
end $$;
