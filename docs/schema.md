# Database Schema

Full canonical schema lives in `supabase_schema.sql`. Apply it to a fresh or existing database — the script is idempotent.

---

## Tables

### `public.observations`

Canonical observation table for iNaturalist sync. One row per iNaturalist record, keyed by `inat_id`.

| Column | Type | Notes |
|---|---|---|
| `inat_id` | `bigint` | **Primary key** — iNaturalist record ID |
| `uuid` | `text` | iNaturalist UUID |
| `species` | `text` | Taxon/species name |
| `date` | `date` | Observation date |
| `lat` | `double precision` | Latitude |
| `lon` | `double precision` | Longitude |
| `location` | `text` | Human-readable location string |
| `num_identification_agreements` | `integer` | Agreement count from iNat |
| `quality_grade` | `text` | e.g. `"research"`, `"needs_id"` |
| `raw_payload` | `jsonb` | Full raw API response |
| `created_at` | `timestamptz` | Row creation time (default `now()`) |
| `updated_at` | `timestamptz` | Auto-updated on every write via trigger |

**Indexes:**
- `observations_species_idx` — on `species`
- `observations_date_idx` — on `date`
- `observations_location_idx` — spatial GiST index on `ST_GeomFromText('POINT(' || lon || ' ' || lat || ')', 4326)`

**Triggers:**
- `observations_set_updated_at` — sets `updated_at = now()` before every update

---

### `public.observation_enrichments`

Optional enrichment table for data appended by the Python/GEE pipeline. One row per observation, 1:1 with `observations`.

| Column | Type | Notes |
|---|---|---|
| `inat_id` | `bigint` | **Primary key** — FK → `observations.inat_id` (cascade delete) |
| `elevation` | `double precision` | Elevation (m) |
| `tavg` | `double precision` | Average temperature |
| `tmin` | `double precision` | Minimum temperature |
| `tmax` | `double precision` | Maximum temperature |
| `soil_moisture` | `double precision` | Soil moisture index |
| `ndvi` | `double precision` | NDVI value |
| `precip_7d` | `double precision` | 7-day precipitation |
| `cluster` | `integer` | MaxEnt/clustering label |
| `created_at` | `timestamptz` | Row creation time (default `now()`) |
| `updated_at` | `timestamptz` | Auto-updated on every write via trigger |

**Indexes:**
- `observation_enrichments_cluster_idx` — on `cluster`

**Triggers:**
- `observation_enrichments_set_updated_at` — sets `updated_at = now()` before every update

---

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

observations
  └─ observation_enrichments  (1:1, cascade delete)
```
