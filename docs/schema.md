# Database Schema

Applied incrementally via `supabase/migrations/`. Run each file in order on a new or existing database.

> **Observation data lives in Supabase Storage** (GeoJSON files), not in Postgres. The old `observations` and `observation_enrichments` tables are dropped by migration 012.

---

## Tables

### `public.user_settings`

Per-user display preferences stored as a single JSON blob. Avoids migrations for new toggles/preferences.

| Column | Type | Notes |
|---|---|---|
| `user_id` | `uuid` | **Primary key** — FK → `auth.users(id)` (cascade delete) |
| `settings` | `jsonb` | Arbitrary preference blob (default `{}`) |
| `updated_at` | `timestamptz` | Auto-updated on every write via trigger |

**RLS:** Enabled. Policy `"own settings"` — users can only read/write their own row (`auth.uid() = user_id`).

**Triggers:**
- `user_settings_set_updated_at` — sets `updated_at = now()` before every update

---

### `public.datasets`

Named, reusable datasets promoted from job results or uploaded directly. Renamed from `saved_datasets` in migration 011.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | **Primary key** — auto-generated |
| `owner_id` | `uuid` | FK → `auth.users(id)` (cascade delete) |
| `title` | `text` | Human-readable name |
| `slug` | `text` | URL-safe identifier, unique |
| `visibility` | `text` | `'private'` or `'public'` |
| `path` | `text` | Supabase Storage path to the GeoJSON file |
| `created_at` | `timestamptz` | Row creation time |
| `updated_at` | `timestamptz` | Auto-updated on every write via trigger |

**Indexes:** `datasets_visibility_idx`, `datasets_owner_idx`

**RLS:** Enabled. See migration 002 for policies.

---

### `public.saved_charts`

User-authored saved chart configurations. Each row is one saved chart with an ordered position within the user's list.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | **Primary key** — auto-generated (`gen_random_uuid()`) |
| `user_id` | `uuid` | FK → `auth.users(id)` (cascade delete) |
| `config` | `jsonb` | Full chart builder config (schema-free, forward-compatible) |
| `title` | `text` | Display name (nullable) |
| `position` | `integer` | Ordering within the user's chart list (default `0`) |
| `created_at` | `timestamptz` | Row creation time (default `now()`) |
| `updated_at` | `timestamptz` | Auto-updated on every write via trigger |

**Indexes:**
- `saved_charts_user_idx` — on `(user_id, position)`

**RLS:** Enabled. Policy `"own charts"` — users can only read/write their own rows (`auth.uid() = user_id`).

**Triggers:**
- `saved_charts_set_updated_at` — sets `updated_at = now()` before every update

---

## Shared Functions

### `public.set_updated_at()`

Trigger function used by all four tables. Sets `NEW.updated_at = now()` before any `UPDATE`.

```sql
create or replace function public.set_updated_at()
returns trigger as $$
begin
  new.updated_at = now();
  return new;
end;
$$ language plpgsql;
```

---

## Upsert Pattern

```sql
insert into public.observations
  (inat_id, uuid, species, date, lat, lon, location,
   num_identification_agreements, quality_grade, raw_payload)
values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
on conflict (inat_id) do update set
  uuid                          = excluded.uuid,
  species                       = excluded.species,
  date                          = excluded.date,
  lat                           = excluded.lat,
  lon                           = excluded.lon,
  location                      = excluded.location,
  num_identification_agreements = excluded.num_identification_agreements,
  quality_grade                 = excluded.quality_grade,
  raw_payload                   = excluded.raw_payload,
  updated_at                    = now();
```

---

## Entity Relationships

```
auth.users
  └─ user_settings  (1:1, cascade delete)
  └─ saved_charts   (1:many, cascade delete)
  └─ datasets       (1:many, cascade delete via owner_id)

model_configs
  └─ datasets       (FK source_dataset_id, SET NULL on delete)
```
