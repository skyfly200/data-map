# Nexstrata

A map of environmental conditions around mushroom observations. iNaturalist
observations are enriched with remotely-sensed environmental layers and then
clustered by environmental similarity. The frontend is a Nuxt app.

**The name.** *strata*: layers. Where a mushroom grows is not one fact but a
stack of them: the weather of the week before, the canopy over it, the moisture
in the soil, the shape and aspect of the ground, how much sun and wind that
shape lets through. *nex*, from *nexus*, a binding together. An observation is
the one place all those layers meet. The mark shows it: a beam passing down
through every stratum onto a geotag on the ground.

The repository is still `data-map`; the app is Nexstrata.

## Enrichment Pipeline (JavaScript / Earth Engine)

Observation data is enriched entirely server-side in Node via
`netlify/lib/ee-pipeline.mjs` and `netlify/lib/ee-runner.mjs`. There is no
Python requirement to run the app or refresh data.

### How enrichment works

A member submits an enrichment job from the **Jobs** page (or one is queued
automatically after an import). The job queue (`netlify/lib/job-queue.mjs`)
picks it up, and `ee-runner.mjs` calls Earth Engine's `reduceRegions` for each
stage. Requests are batched by observation date — points sharing a date share
one composite and one round trip — so a 7-day weather history costs one EE call
per date, not one per observation.

| Column | Earth Engine dataset |
| --- | --- |
| `ndvi` | `COPERNICUS/S2_SR_HARMONIZED` |
| `soil_moisture` | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| `prcp_d0..d6` | `UCSB-CHG/CHIRPS/DAILY` |
| `tmax_d0..d6`, `tmin_d0..d6` | `ECMWF/ERA5_LAND/DAILY_AGGR` |
| `land_cover` | `ESA/WorldCover/v200` |
| `elevation`, `slope`, `aspect` | `USGS/SRTMGL1_003` |
| `solar_exposure`, `wind_exposure`, `water_retention` | derived terrain indices from SRTM + `MERIT/Hydro/v1_0_1` |

Enrichment results are written back to Supabase Storage and become immediately
available in the app without a redeploy.

### Data sources

- **iNaturalist** — `netlify/functions/fetch-species.mjs` (on-demand, auth-gated)
  and `netlify/functions/refresh-observations.mjs` (scheduled every 6 h). Best
  for recent sightings and smaller taxon pulls (< 5 000 records or < 60 days).
- **GBIF** — `components/GbifImporter.vue` handles CSV uploads today. A live
  GBIF API import flow is planned (`WANT-9` in the roadmap); prefer GBIF for
  large or historical pulls.

After any import, new features arrive with `enrichment_level: 'none'`. An
automatic enrichment trigger is planned (`WANT-10`); until then, submit an
enrichment job manually from the Jobs page.

### Credentials

Set these in your Netlify environment (or `.env` for local dev):

| Variable | Purpose |
| --- | --- |
| `EARTHENGINE_SERVICE_ACCOUNT_KEY` | JSON key for a Google Cloud service account with Earth Engine access |
| `EARTHENGINE_PROJECT` | Cloud project ID registered for Earth Engine (e.g. `my-project-451208`) |
| `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` | Supabase project — stores jobs, results, and datasets |

**Finding `EARTHENGINE_PROJECT`**: [console.cloud.google.com](https://console.cloud.google.com/)
lists every project with its ID column. Not registered?
[code.earthengine.google.com/register](https://code.earthengine.google.com/register)
attaches a Cloud project to Earth Engine (free for noncommercial use).

### Legacy Python pipeline

> **Obsolete.** The Python scripts (`iNat.py`, `enrich_with_rasters.py`,
> `ee_enrich.py`, `cluster.py`, `export_geojson.py`, `run_pipeline.py`) and the
> GitHub Action (`refresh-data.yml`) that ran them are no longer the primary
> data path. They are retained for reference until the JS pipeline covers all
> stages end-to-end in production (see `WANT-10` in the roadmap). Do not add new
> enrichment columns to the Python scripts; add them to `STAGES` in
> `netlify/lib/ee-pipeline.mjs`.

The old Kaggle notebook (`notebooks/kaggle_pipeline.ipynb`) and the raster
fallback path (`fetch.py`, `terrain_pipeline.py`) remain in the repository but
are not run in CI.

## Nuxt frontend & Netlify deploy

The frontend is a Nuxt 3 app that renders the observations on a Leaflet map,
colored by environmental cluster, with the enriched attributes in each popup.

**Filtering.** The **Data** tab is the control centre: pick species, and narrow
by **location** (country / state / county, parsed from each record's place
string, or a lat/lng centre + radius in km) and **time** (year, month, ISO
week, or a from/to date range). Filters live in shared state, so they apply
everywhere at once (map, table, charts, explore); a "Filters: N" chip in the
header links back to the Data tab from any view. When you fetch a *new* species
while location/time filters are set, the fetch is scoped to match (iNaturalist
radius + observed-date range) instead of pulling the whole history.

**Data flow:**

```
iNat API / GBIF import
       ↓
Netlify Functions (fetch-species, refresh-observations, gbif-fetch)
       ↓
Supabase Storage (observations.geojson, species/<slug>.geojson)
       ↓
ee-jobs → ee-worker → ee-pipeline.mjs (Earth Engine enrichment)
       ↓
Supabase Storage (enriched dataset written back)
       ↓
useObservations (progressive chunk loading) → map / table / charts
```

Run the site locally:

```bash
npm install
npm run dev        # http://localhost:3000
```

**Netlify:** `netlify.toml` pins the build (`npm run build`, publish `dist`,
Node 20); Nuxt's Nitro auto-selects the Netlify preset. Only the small GeoJSON
in `public/data/` is served: keep the raster folders (`soil/ ndvi/ dem/ …`)
out of the deploy (they are gitignored). Do **not** put Earth Engine /
OpenTopography / CDS credentials in Netlify; those belong only to the offline
pipeline.

### Keeping the data fresh (automated)

- **Scheduled Netlify Function** (`netlify/functions/refresh-observations.mjs`,
  every 6 h) fetches recent iNaturalist sightings, merges new ones onto the
  current dataset, and writes the result to **Supabase Storage** (or Netlify
  Blobs as a fallback). The serving function (`netlify/functions/observations.mjs`)
  returns that fresh copy. Configure the iNaturalist query with env vars
  (`INAT_TAXON`, `INAT_LAT`, `INAT_LNG`, `INAT_RADIUS`) in Netlify site settings.

  New observations arrive with `enrichment_level: 'none'`. Submit an enrichment
  job from the Jobs page to populate the EE columns. Automatic post-import
  enrichment is tracked in `WANT-10`.

> **Note:** The GitHub Action (`refresh-data.yml`) that ran the Python
> enrichment pipeline daily is no longer the primary refresh path. It is
> retained for reference but should not be relied on for new deployments. See
> the [Legacy Python pipeline](#legacy-python-pipeline) note above.

### Serving datasets from Supabase Storage (optional)

By default the datasets live in `public/data/` (committed) and the interim
refresh uses Netlify Blobs. You can instead store them in **Supabase Storage**,
which becomes the source of truth the frontend and functions read from. It's
entirely opt-in: with no `SUPABASE_*` env set, everything falls back to the
committed files / Blobs.

Setup:

1. Pick (or create) a Supabase project and a **public** Storage bucket, e.g.
   `datasets`.
2. Set env vars:
   - GitHub Actions secrets: `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
     (and optionally `SUPABASE_DATASETS_BUCKET`). The `refresh-data` workflow
     runs `node scripts/upload_datasets.mjs` after export to push
     `observations.geojson`, `species/*.geojson`, and a rewritten
     `datasets.json` (with Supabase public URLs) to the bucket.
   - Netlify env: `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` (so the
     functions read/write Storage instead of Blobs), and
     `NUXT_PUBLIC_DATASETS_MANIFEST_URL` = the bucket's public
     `datasets.json` URL (so the frontend loads datasets from Supabase).
3. The service-role key is **write-only server-side**, it lives in Actions /
   Netlify env, never in the browser bundle.

Data flow: import (iNat / GBIF) → EE enrichment job → `observations.geojson`
written to Supabase Storage → frontend + `observations` function read from
Supabase; the scheduled `refresh-observations` function merges new sightings
into the same bucket.

### Membership from an automation

`netlify/functions/membership.mjs` turns a payment into a member without anyone
signing in: the PayPal webhook on the FRMS site calls it with an email and a
term. It is authenticated with a shared key (`MEMBERSHIP_API_KEY`) rather than a
session, is idempotent when given the processor's transaction id, and records a
grant for an address that has no account yet so it applies itself at signup.

Full reference, including the PayPal wiring: [docs/membership-api.md](docs/membership-api.md).

Needs `supabase_migrations/004_membership_api.sql`.

### Sign-in to protect the live-fetch endpoints (Supabase Auth)

Browsing (map, table, charts, explore) is fully open. The endpoints that spend
something on the FRMS account's behalf are gated behind **Supabase Auth** so they
can't be hammered anonymously: `fetch-species` (Data tab) calls iNaturalist,
`ee-tiles` and `ee-jobs` call Earth Engine, and `admin-members` and `ee-worker`
require the admin tier.

- **Server side:** `netlify/lib/auth.mjs` validates the caller's Supabase
  access token (`Authorization: Bearer <jwt>`) via `auth.getUser`. Enforcement
  turns on automatically whenever Supabase is configured (`SUPABASE_URL` +
  `SUPABASE_ANON_KEY` in the Netlify env). Overrides: `AUTH_DISABLED=true`
  forces the endpoints open even when configured; `AUTH_REQUIRED=true` fails
  closed and refuses traffic until Supabase is configured.
- **Browser side:** set the public keys so the login UI works and fetches send
  the token:
  - Netlify / `.env`: `NUXT_PUBLIC_SUPABASE_URL`,
    `NUXT_PUBLIC_SUPABASE_ANON_KEY` (the anon key is public by design; the
    service-role key stays server-only).
- **Enable sign-in methods** in the Supabase dashboard (Authentication →
  Providers): Email (password + magic link) is on by default; enable **GitHub**
  and **Google** OAuth and add your site URL + `…/login` to the redirect
  allow-list. Turn on **Passkeys / WebAuthn** there too. The `/login` page
  surfaces all of these: "Sign in with a passkey" for returning users, and
  "Add a passkey to this account" once you're signed in (the client enables the
  experimental passkey API automatically).

When Supabase public keys are **not** set, the login UI shows a
“not configured” notice and fetches run unauthenticated (which the functions
allow only because the server is likewise unconfigured), so local dev works
with zero credentials.

### Job completion notifications (email + push)

A pipeline or model job runs on the server for minutes, so a member usually
leaves the page. When a job settles, the `ee-worker` function notifies its owner
over two independent channels, **both opt-out and on by default**, and each a
silent no-op until its env is set. Members turn either off (or turn push on for
a browser) from the **Jobs** page → *Notifications*.

First apply the migration, which adds the two preference columns to `profiles`
and a `push_subscriptions` table:

```
supabase_migrations/006_job_notifications.sql
```

**Email — [Resend](https://resend.com).** No new dependency; the worker calls
the HTTP API with `fetch`. Set in the Netlify env:

- `RESEND_API_KEY` — an API key from the Resend dashboard.
- `NOTIFY_FROM_EMAIL` — a **verified** sender on that account, e.g.
  `Nexstrata <notifications@your-domain.org>`. An unverified From is rejected,
  so without this, email stays off.
- `NOTIFY_APP_URL` *(optional)* — the base URL used for the “view it” link.
  Defaults to Netlify's own `URL` / `DEPLOY_PRIME_URL`.

**Push — Web Push (VAPID).** Uses the `web-push` package (added to
`dependencies`). Generate a key pair once:

```bash
npx web-push generate-vapid-keys
```

Then set in the Netlify env:

- `VAPID_PUBLIC_KEY` — the public half; also handed to the browser (via
  `/.netlify/functions/job-notifications`) so it can subscribe. Public by design.
- `VAPID_PRIVATE_KEY` — the private half; **server-only**, never in the bundle.
- `VAPID_SUBJECT` *(optional)* — a `mailto:` or `https:` contact for the push
  service, e.g. `mailto:admin@your-domain.org`.

Push also needs a service worker, which the app already ships (`public/sw.js`,
shared with offline support) — its `push` / `notificationclick` handlers show
the message and focus the Jobs page on click. Delivery requires HTTPS (Netlify
serves HTTPS; on `localhost` browsers allow it for testing). Subscriptions the
push service reports as gone (404/410) are pruned automatically.

With neither channel configured the queue behaves exactly as before: jobs run,
progress is written to the row, and the browser polls it — no notifications.

Look at the [Nuxt documentation](https://nuxt.com/docs/getting-started/introduction) to learn more.

## Setup

Make sure to install dependencies:

```bash
# npm
npm install

# pnpm
pnpm install

# yarn
yarn install

# bun
bun install
```

## Development Server

Start the development server on `http://localhost:3000`:

```bash
# npm
npm run dev

# pnpm
pnpm dev

# yarn
yarn dev

# bun
bun run dev
```

## Production

Build the application for production:

```bash
# npm
npm run build

# pnpm
pnpm build

# yarn
yarn build

# bun
bun run build
```

Locally preview production build:

```bash
# npm
npm run preview

# pnpm
pnpm preview

# yarn
yarn preview

# bun
bun run preview
```

Check out the [deployment documentation](https://nuxt.com/docs/getting-started/deployment) for more information.

## Citations

### Data Sources

* **Species Occurrences:** GBIF.org (18 September 2026) GBIF Occurrence Download https://doi.org/10.15468/dl.vea6qx
* **Terrain & Climate:** USGS 3DEP 10m DEM, NLCD 2021, ERA5-Land (via Google Earth Engine)
