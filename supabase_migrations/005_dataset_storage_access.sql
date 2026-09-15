-- Who can read the files behind a job result or a saved dataset.
--
-- Run after 004_membership_api.sql. Safe to run more than once.
--
--   psql "$DATABASE_URL" -f supabase_migrations/005_dataset_storage_access.sql
--
-- Until this ran there was no storage.objects policy at all, which left one
-- unhappy choice: a private bucket where a member cannot read their own result,
-- or a public one where anybody holding a path can read anybody's. Running a
-- job is most of what membership buys, and the result is the member's own work,
-- so neither was right.
--
-- The bucket must be PRIVATE. These policies decide who reads what inside it;
-- a public bucket serves every object to anyone with the URL and never consults
-- them at all.
--
-- ─────────────────────────────────────────────────────────────────────────────
-- The bucket name is written here as a literal
-- ─────────────────────────────────────────────────────────────────────────────
--
-- SUPABASE_DATASETS_BUCKET can rename it for the server, and SQL cannot read an
-- environment variable. So if the bucket is ever renamed, these policies have to
-- be edited too — otherwise they silently stop matching and every read is
-- refused, with nothing naming the cause. Leaving the variable at its default
-- is the recommended arrangement.

do $$
begin
  if to_regprocedure('public.is_admin()') is null then
    raise exception using
      message = 'Migration 002 has not been applied in full: is_admin() missing.',
      hint = 'Run 002, then 003, then 004, then this file.';
  end if;
end
$$;

-- storage.objects is owned by the storage extension, and on a plain Postgres
-- with no Supabase storage installed it does not exist. Guarded so this file
-- can be applied and tested anywhere.
do $$
begin
  if to_regclass('storage.objects') is null then
    raise notice 'No storage.objects table: skipping the bucket policies. %',
      'Install Supabase Storage and re-run this file before members save datasets.';
    return;
  end if;

  -- ── Reading ────────────────────────────────────────────────────────────────
  --
  -- A member reads what is under their own jobs/<uid>/ prefix, and nothing else.
  -- Admins read everything, because supporting somebody through a failed job
  -- means being able to look at what it produced.
  --
  -- Datasets shared with other members are deliberately NOT readable here. A
  -- shared dataset is read through the server, which resolves the row, applies
  -- the visibility rule in netlify/lib/dataset-access.mjs and streams the file.
  -- Making the browser path clever enough to handle sharing would mean encoding
  -- that rule a second time, in a place where it has to agree with the first.
  execute $p$
    drop policy if exists "read own job results" on storage.objects
  $p$;
  execute $p$
    create policy "read own job results" on storage.objects
      for select to authenticated
      using (
        bucket_id = 'datasets'
        and (
          public.is_admin()
          or (
            (storage.foldername(name))[1] = 'jobs'
            and (storage.foldername(name))[2] = auth.uid()::text
          )
        )
      )
  $p$;

  -- ── Writing ────────────────────────────────────────────────────────────────
  --
  -- There is no insert, update or delete policy, and that is the decision
  -- rather than an omission. Row-level security denies what it does not permit,
  -- so every write to this bucket goes through the service role: the worker
  -- writing a result it just produced.
  --
  -- A browser write policy would have to be scoped by path prefix, and a member
  -- who can write under their own prefix can overwrite the result of a job they
  -- already ran — which is the file a saved dataset points at, and which
  -- somebody may have shared. The pipeline's output should be the pipeline's.
end
$$;

-- ─────────────────────────────────────────────────────────────────────────────
-- Saved datasets are created server-side only
-- ─────────────────────────────────────────────────────────────────────────────
--
-- Also stated rather than omitted. 002 gives saved_datasets a select policy and
-- an admin-manages-everything policy, and nothing else, so a member cannot
-- insert or update a row from the browser.
--
-- That is deliberate and it is not merely tidiness. The row carries `path`,
-- which names a file in the bucket. A member who could insert their own row
-- could point it at jobs/<somebody else>/<their job>.geojson and then read it
-- through the server, which trusts the path on the row it resolved. The
-- authorization would be perfectly correct and would authorize the wrong file.
--
-- So datasets are created by netlify/functions/datasets.mjs, which sets `path`
-- from a job row it has already confirmed belongs to the caller. The member
-- never supplies a path at all.
--
-- The check below is a guard against that going wrong later: a member-owned row
-- may only point inside its owner's own prefix. Admin-owned and FRMS-published
-- datasets are exempt, since those legitimately point at curated files that
-- nobody's job produced.

create or replace function public.dataset_path_is_owned()
returns trigger
language plpgsql
as $$
begin
  if new.owner_id is not null
     and new.path is not null
     and new.path like 'jobs/%'
     and new.path not like 'jobs/' || new.owner_id::text || '/%' then
    raise exception
      'A dataset owned by % may not point at %', new.owner_id, new.path
      using hint = 'A member-owned dataset can only reference that member''s own job results.';
  end if;
  return new;
end;
$$;

drop trigger if exists saved_datasets_path_owned on public.saved_datasets;
create trigger saved_datasets_path_owned
before insert or update on public.saved_datasets
for each row execute function public.dataset_path_is_owned();

-- Datasets are listed by owner on the member's own screen, which 002 indexed,
-- and looked up by slug on every job that names one, which the unique
-- constraint already covers.
