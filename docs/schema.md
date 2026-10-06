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
| `accuracy_m` | `double precision` | Coordinate uncertainty radius (metres). `NULL` = unknown, not precise |
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

### Foray planner additions (migration `012_model_contributions_and_access.sql`, WANT-17)

- `model_results.contributions` (`jsonb`, `{predictor: %}`) and `model_results.predictor_ranges` (`jsonb`, `{predictor: {p25,p75,min,max}}`). Written by `ee-worker` via `netlify/lib/model-persist.mjs` when a `model_runs` row exists for the job. Contributions exist only for `autoOptimize` runs; `predictor_ranges` is not yet computed by the runner.
- `access_areas` — PAD-US units (`source`, `source_id`, `name`, `manager`, `designation`, `access_class` open/restricted/closed/unknown, `geom` MultiPolygon 4326, GiST). Unique `(source, source_id)`.
- `access_lines` — OSM roads/trails (`osm_id` unique, `kind` road/trail, `highway`, `name`, `geom` LineString 4326, GiST).
- RLS: select-all on both; writes via service role only.

### Access database (migration `013_access_database.sql`, WANT-17)

- `access_areas` adds: `manager_type` (blm/usfs/nps/fws/state_park/state/local/federal_other/private/tribal/other/unknown), `public_access` (open/restricted/closed/unknown, from PAD-US `Pub_Access`), `fee_status` (free/fee/unknown) + `fee_source` (`estimated`|`ridb`|null), `collecting` (allowed/likely_allowed/restricted/prohibited/unknown; `likely_allowed` is estimated only, for open BLM/USFS land not restricted by designation) + `collecting_source` (`estimated`|null), `region` (name of the load that wrote the row). **All `estimated` values are heuristics from manager type/designation, not regulations**; `ridb` fee values come from a matching Recreation.gov facility record. Collecting is never estimated as plain `allowed`. Migration 014 adds `likely_allowed` to the CHECK.
- `access_lines.region` added.
- `access_regions` — `name` unique, `bbox` jsonb `[w,s,e,n]`, `status` queued/loading/loaded/partial/failed, `area_count`, `line_count`, `ridb_matched`, `sources`, `error`, `requested_by`, `loaded_at`. RLS select-all; writes service role only.
- Functions: `access_upsert_areas/lines(rows jsonb, region)` (service role only; keep RIDB fee data over later estimates), `access_areas_in_bbox`, `access_lines_in_bbox` (simplified GeoJSON, used by `netlify/functions/access.mjs`).

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

### `public.member_ee_credentials`

Member-supplied Earth Engine service-account key (WANT-3). Migration `011_member_ee_credentials.sql`. RLS on with **no policies**; service role only. The browser only ever sees a summary from `ee-credentials`.

| Column | Type | Notes |
|---|---|---|
| `user_id` | `uuid` | **Primary key** — FK → `auth.users` (cascade delete) |
| `ciphertext` | `text` | AES-256-GCM `v1.iv.tag.data`, key `EE_CREDENTIAL_KEY` (env), owner id bound as AAD |
| `client_email` | `text` | Service-account email (shown to owner) |
| `project_id` | `text` | Cloud project the jobs run under |
| `validated_at` | `timestamptz` | Last successful EE round-trip |
| `created_at` | `timestamptz` | Default `now()` |

---

## Entity Relationships

```
auth.users
  └─ user_settings  (1:1, cascade delete)
  └─ saved_charts   (1:many, cascade delete)
  └─ member_ee_credentials  (1:1, cascade delete)

observations
  └─ observation_enrichments  (1:1, cascade delete)
```

### Member access-area sets (migration `015_access_area_sets.sql`, WANT-17)
- No club/org concept existed, so minimal `clubs` (`id`, `name`, `created_by`) and `club_members` (`club_id`, `user_id`, `role` owner/admin/member; PK `(club_id, user_id)`). RLS: members read; writes service role only (via `access-sets.mjs`).
- `access_area_sets` — `name`, `scope` user/club, `owner_id`, `club_id` (set iff scope=club), `created_at`. RLS: user scope owner-only; club scope readable by members, writable by club owner/admin or the set's creator while a member.
- `access_set_areas` — `set_id` (cascade), `name`, `geom` MultiPolygon 4326 + GiST, `fee_status` free/fee/unknown, `collecting` allowed/likely_allowed/restricted/prohibited/unknown (**owner-asserted**, so plain `allowed` is valid, unlike `access_areas`), `notes`, `created_by`. RLS follows the parent set; club areas are also editable by their `created_by` while a member.
- Functions (service role only): `access_set_area_insert`, `access_set_area_update_geom` (validate/ST_MakeValid/ST_Multi), `access_set_areas_geojson(p_set_id)` (all areas of one set as GeoJSON, incl. `created_by`), `access_set_areas_in_bbox(p_user, w, s, e, n, tol, limit)` (simplified GeoJSON of that user's visible sets, mirrors `access_areas_in_bbox`). Endpoints: `netlify/functions/access-sets.mjs` (CRUD contract in file header), `access.mjs?include_sets=1` merges them.
