# Deploying: what to set, and in what order

Everything here is optional in the sense that the app runs without any of it —
it serves the committed dataset and works offline. Each block below switches on
one more thing, and nothing depends on a block below it.

`.env.example` documents the same variables inline; this is the deployment view.

---

## The migrations, in order

The blocks below introduce these one at a time, alongside the variables each
needs. This is the same list in one place, for checking what has already been
run. Each file is safe to run again, and running one twice is the normal way to
pick up changes to it.

| | | |
|---|---|---|
| 1 | `001_user_settings_and_charts.sql` | Saved settings and saved charts. Needs block 2 below. |
| 2 | `002_membership_jobs_and_admin.sql` | Profiles, tiers, the Earth Engine job queue, saved datasets, and the access token hook. Block 3. |
| — | **Dashboard → Authentication → Hooks** | Point the Custom Access Token hook at `public.custom_access_token_hook`. **Until this is on, every account reads as `free`, including yours** — the tier travels in the token and nothing is stamping it. |
| — | **Promote yourself** | The SQL at the bottom of 002. Then sign out and back in. |
| 3 | `003_custom_ee_layers.sql` | Earth Engine assets registered as map layers from the admin screen. Block 4. |
| 4 | `004_membership_api.sql` | The grants table and the signup trigger behind the membership API. See `membership-api.md`. |
| 5 | `005_dataset_storage_access.sql` | Who can read the files behind a job result or a saved dataset. Needs the storage bucket below. |

Run them in that order. 003 checks that 002 applied in full and says so by name
if it did not — if you see *"Migration 002 has not been applied in full"*,
re-run 002 and read its output rather than skipping ahead.

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
| `WORKER_POKE_SECRET` | functions | Optional. Any long random string. Lets the submit endpoint start the job worker the instant a job is queued, instead of waiting for the next scheduled run. Without it, jobs still start — on the worker's every-minute schedule. **Server only.** |

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

### The four tiers

| | |
|---|---|
| `free` | Signed in. Reads the shipped data. |
| `member` | Dues paid. May run pipeline jobs. Lapses on `member_until`. |
| `perpetual` | A member whose standing does not run out: honorary and life members, founders. Same powers as `member`; `member_until` is ignored. |
| `admin` | Manages other people's tiers, quotas and datasets. Also does not expire. |

`perpetual` and `admin` ignore the expiry date rather than requiring it to be
empty — the date may well be set, as a record of dues that were in fact paid,
and it simply stops governing access.

Admin is exempt for a structural reason rather than a generous one: this screen
is the only place a membership date can be corrected, and the token hook is the
only thing that mints the admin claim. An admin demoted by their own dues date
would lose the one place where that date could be fixed, theirs included, and
the recovery would be hand-written SQL.

If you are upgrading a database where 002 ran before these tiers existed,
**re-run 002 and 004**. Both are idempotent, and both re-apply their tier check
constraints on the way through — `create table if not exists` leaves an existing
constraint alone, so without the re-run `perpetual` is rejected as an invalid
tier.

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
| `SUPABASE_DATASETS_BUCKET` | `datasets` | Storage bucket for job results and saved datasets. Renaming it also means setting `NUXT_PUBLIC_DATASETS_BUCKET` and editing the policy in migration 005, which names it as a SQL literal. Leaving it alone is the recommended arrangement. |
| `AUTH_DISABLED` | unset | Forces the API open even with Supabase configured. For a private preview. |
| `AUTH_REQUIRED` | unset | Fails closed until Supabase is configured. |

### The storage bucket

Create a bucket called `datasets` in Dashboard → Storage, and leave it
**private**. Then run `005_dataset_storage_access.sql`, which decides who reads
what inside it.

A public bucket serves every object to anyone holding the URL and never consults
a policy at all, so a member's job results would be readable by anyone who could
guess a path. Private plus the policy is the arrangement: a member reads their
own `jobs/<uid>/` prefix, admins read everything, and nothing writes from a
browser.

Without the bucket, jobs run and fail at the point of writing their result.
Without the migration, the bucket refuses every read and a member cannot open
their own finished job.

### Saved datasets

A member can name a finished job, which turns it into a dataset another job can
be run over. Those are **private by default**, and a member may share one with
other FRMS members but not publish it to the open web — that stays an admin
action, in the admin screen.

---

## 5. Granting membership from a payment

| Variable | Where | What it is |
|---|---|---|
| `MEMBERSHIP_API_KEY` | functions | A shared key for the membership endpoint. Generate with `openssl rand -base64 32`. |

Needs `004_membership_api.sql`. With no key set the endpoint refuses everything
rather than falling open, so leaving it unset is a safe way to keep it off.

The caller is the FRMS website's PayPal webhook handler, which grants a year of
membership from the same function that sends the welcome email. The endpoint,
the idempotency rules and a tested function to call it with are in
[`membership-api.md`](membership-api.md).

The key grants membership to any address, so it belongs with the service role
key: server-side environment only, never in a browser, never committed.

---

## 6. Putting it on your own domain

Most of the app does not care what it is served from: in-app links are relative,
the service worker and the share/embed links build from `window.location.origin`,
and `manifest.webmanifest` uses relative paths. Earth Engine, the storage bucket
and every row-level security policy key on user ids and paths, not on an origin.

Four things do care.

**Supabase → Authentication → URL Configuration.** Set **Site URL** to the new
origin, with no trailing slash. That is what confirmation and magic-link emails
are built from, so getting it wrong sends every member a link to the old host.
Then add the sign-in landing page to **Redirect URLs**:

```
https://your-domain/login
```

`useAuth` sends `${window.location.origin}/login` as the redirect for magic
links, sign-up confirmations and OAuth, and Supabase refuses a redirect that is
not on that list. The symptom is a bounce to an error page rather than a broken
session, so it is visible immediately if you test one sign-in.

Nothing else in Supabase changes. The GitHub and Google OAuth apps point at
Supabase's own `https://<ref>.supabase.co/auth/v1/callback`, which does not move
with your app.

**Netlify.** Add the domain and make it the **primary** one, so the
`*.netlify.app` address redirects instead of serving the same app from a second
origin. Two live origins means two separate sessions, two separate offline
caches and two sets of browser preferences, with no sign that they are the same
app.

**`NUXT_PUBLIC_SITE_URL`.** Only used to make `og:image` absolute for link
previews, which do not resolve relative URLs. Unset, the app works and the logo
is missing from previews in Slack, Discord and iMessage. Set it in the Netlify
environment; it reaches the browser through the rendered payload, so it takes a
deploy rather than a restart.

**The FRMS website's PayPal function.** `NEXSTRATA_URL` is the origin it posts
memberships to. See [`membership-api.md`](membership-api.md).

### What members lose in the move

Browser storage is per-origin, and there is no way to carry it across. Worth
saying out loud before the switch rather than fielding it afterwards.

- **Passkeys stop working.** A WebAuthn credential is bound to the domain that
  created it. Anyone who signed in with a passkey has to use email or OAuth once
  on the new domain and register a new one.
- **Saved offline areas have to be downloaded again.** The tile cache is keyed
  by origin and lives in the browser, so the new domain starts empty.
- **Theme and the last map view reset.** Local-only, and small.

Preferences that sync do come back on the first sign-in: appearance, chart
layout and order, units, the map overlay settings, saved filters and saved
charts. Everything on the account — membership, jobs, saved datasets — is
untouched, because none of it is addressed by origin.

---

## Which fire layers are public

Set per layer in `netlify/lib/ee-tile-layers.mjs`, not by an environment
variable, because it is a decision about the layer rather than the deployment:

- **Years since fire** and **Burn severity (US)** — `tier: 'free'`. Together
  they answer "which burn scar is worth walking next spring", which is the
  question FRMS exists to help with.
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
