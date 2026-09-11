-- Map layers an administrator registers, pointing at their own Earth Engine
-- assets.
--
-- Run after 002_membership_jobs_and_admin.sql. Safe to run more than once.
--
--   psql "$DATABASE_URL" -f supabase_migrations/003_custom_ee_layers.sql
--
-- The built-in fire layers are a fixed catalogue in code. This is the open
-- half: compute something in the Earth Engine Code Editor, Export.image
-- .toAsset() it into your project, and register the asset here. The app renders
-- it through the same path as the built-ins.
--
-- What is stored is an asset ID and how to paint it — never Earth Engine code.
-- An asset ID names something already computed under an account we control; a
-- script would be arbitrary compute on the society's billing.

create table if not exists public.ee_custom_layers (
  id uuid primary key default gen_random_uuid(),

  -- How the app addresses it. Namespaced as custom:<slug> in the API so it can
  -- never collide with a built-in layer key.
  slug text not null unique,
  name text not null,
  -- Which heading it appears under in the layer picker. Free text, so a society
  -- can group its own layers however it thinks about them.
  "group" text not null default 'Custom',

  asset_id text not null,
  asset_type text not null default 'image'
    check (asset_type in ('image', 'image_collection')),
  -- Selected before the collection is reduced. A palette on a multi-band image
  -- is refused by Earth Engine, so this matters whenever the asset has more
  -- than one band.
  band text,
  reducer text check (reducer in ('mosaic', 'mean', 'median', 'max', 'min', 'first')),
  date_from date,
  date_to date,

  vis_min double precision,
  vis_max double precision,
  palette text[] not null default '{}',
  -- Values at or below this are hidden. Without it an asset whose no-data is
  -- zero paints the whole world the bottom of the ramp.
  mask_below double precision,
  opacity double precision not null default 0.8,

  tier text not null default 'member' check (tier in ('free', 'member', 'admin')),
  attribution text,
  note text,

  created_by uuid references auth.users(id) on delete set null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists ee_custom_layers_group_idx on public.ee_custom_layers ("group");

alter table public.ee_custom_layers enable row level security;

-- Readable by anyone whose tier reaches the layer, so the picker can list it.
-- Listing is not rendering: the tile function checks the tier again before
-- spending any Earth Engine quota.
drop policy if exists "read layers by tier" on public.ee_custom_layers;
create policy "read layers by tier" on public.ee_custom_layers
  for select using (
    tier = 'free'
    or (tier = 'member' and public.is_member())
    or public.is_admin()
  );

-- Only an administrator writes one, and only through the API, which validates
-- the asset ID before it reaches Earth Engine.
drop policy if exists "admins manage layers" on public.ee_custom_layers;
create policy "admins manage layers" on public.ee_custom_layers
  for all using (public.is_admin()) with check (public.is_admin());

drop trigger if exists ee_custom_layers_set_updated_at on public.ee_custom_layers;
create trigger ee_custom_layers_set_updated_at
before update on public.ee_custom_layers
for each row execute function public.set_updated_at();
