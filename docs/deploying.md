# Deploying: what to set, and in what order

Everything here is optional in the sense that the app runs without any of it —
it serves the committed dataset and works offline. Each block below switches on
one more thing, and nothing depends on a block below it.

`.env.example` documents the same variables inline; this is the deployment view.

---

## 1. Nothing at all

The map, charts, analysis, phenology and offline saving all work from the
committed `public/data/observations.geojson`. No accounts, no keys.

---

## 2. Accounts, saved settings and saved charts

Set these and the sign-in UI appears, settings and charts sync across devices.

| Variable | Where | What it is |
|---|---|---|
| `NUXT_PUBLIC_SUPABASE_URL` | build + browser | `https://<ref>.supabase.co` |
| `NUXT_PUBLIC_SUPABASE_ANON_KEY` | build + browser | The anon key. Public by design. |
| `SUPABASE_URL` | functions | Same URL again, server side. |
| `SUPABASE_ANON_KEY` | functions | Same anon key, used to verify tokens. |

The `NUXT_PUBLIC_*` pair is read at **build** time and baked into the bundle, so
changing them needs a redeploy, not just a restart. The unprefixed pair is read
at request time by the Netlify functions.

They must point at the same project. Pointing them at different ones produces a
401 on every function call, which is the failure `FetchSpecies` has a whole
paragraph of error text about.

Then run, in order:

```
supabase_migrations/001_user_settings_and_charts.sql
```

---

## 3. Membership tiers, quotas and the admin console

| Variable | Where | What it is |
|---|---|---|
| `SUPABASE_SERVICE_ROLE_KEY` | functions | Bypasses row-level security. **Server only.** |

This one is the difference between a member being able to *read* their tier and
the server being able to *set* it. Tier and quota columns are deliberately not
writable by any member — the browser holds the anon key, so a self-writable tier
column would be a free membership button — which means the only thing that can
grant membership is a caller holding this key.

Never expose it to the browser, never put it in a `NUXT_PUBLIC_*` variable.

Then:

1. Run `supabase_migrations/002_membership_jobs_and_admin.sql`.
2. Dashboard → **Authentication → Hooks → Custom Access Token**, point it at
   `public.custom_access_token_hook`. Until this is enabled every account reads
   as `free`, including yours.
3. Make yourself an admin with the SQL at the bottom of that migration, then
   **sign out and back in** — the tier travels in the token, so it arrives with
   a new one.

---

## 4. Earth Engine: the pipeline and the computed map layers

| Variable | Where | What it is |
|---|---|---|
| `EARTHENGINE_SERVICE_ACCOUNT_KEY` | functions | The service account's JSON key, or base64 of it. |
| `EARTHENGINE_PROJECT` | functions | The Google Cloud project id registered with Earth Engine. |

Setup:

1. Create a service account in a Google Cloud project.
2. Register it for Earth Engine at <https://signup.earthengine.google.com>. As a
   501(c)(3), the noncommercial tier applies — worth confirming once with Google,
   since it is your licence.
3. Give it a JSON key.

Netlify environment variables mangle embedded newlines, and a service account
key is mostly newlines, so **base64 the whole file**:

```
base64 -w0 service-account.json
```

Both forms are accepted: the raw JSON if it survives, base64 otherwise.

Earth Engine bills the **project**, not the caller, so every member's job spends
from one pool. The per-member quotas on the `profiles` row are what divide it —
see `netlify/lib/quotas.mjs`.

### Optional

| Variable | Default | What it does |
|---|---|---|
| `EE_REQUEST_DEADLINE_MS` | `0` (off) | Per-request deadline. **Leave off.** A fixed deadline kills the slow-but-valid samplers (land cover, NDVI, soil moisture) and leaves those columns empty, which is worse than a hang. Retry and backoff is what recovers transient failures. |
| `SUPABASE_DATASETS_BUCKET` | `datasets` | Storage bucket for job results and saved datasets. |
| `AUTH_DISABLED` | unset | Forces the API open even with Supabase configured. For a private preview. |
| `AUTH_REQUIRED` | unset | Fails closed until Supabase is configured. |

---

## Which fire layers are public

Set per layer in `netlify/lib/ee-tile-layers.mjs`, not by an environment
variable, because it is a decision about the layer rather than the deployment:

- **Years since fire** and **Burn severity (US)** — `tier: 'free'`. Together
  they answer "which burn scar is worth walking next spring", which is the
  question the society exists to help with.
- **Burn scars this year**, **Active fires**, **dNBR** — members. dNBR in
  particular is computed from raw Sentinel-2 scenes as you look at it, so it
  does not share one cached render the way a published asset does.

Change a layer's `tier` to move it either way. Anything without a `tier` falls
back to `DEFAULT_TIER`, which is `member`.

---

## Checking it worked

- **Signed out**, the fire group shows five layers, two of them without a
  "members" badge. Those two draw.
- **Signed in as a member**, all five draw and `/jobs` accepts a job.
- **As an admin**, `/admin` lists accounts with their usage against their quota.

If a layer is switched on and nothing appears, look for the error card on the
map — an Earth Engine failure is reported there by name and with the reason. A
fire layer must never fail silently: blank ground reads as ground that never
burned.
