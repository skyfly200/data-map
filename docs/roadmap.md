# Roadmap

What is known to be missing, and why. Ordered by what blocks something else
first, then by what is most often asked for.

Each entry says what is true today, not just what should be. An item here is a
commitment to an open question; when it is closed the entry moves to the bottom
section with the commit that closed it.

---

## Blocking

Nothing open.

---

## Wanted

### Member-defined enrichment stages

`STAGES` is six hardcoded entries and `normaliseSpec` refuses any `kind` but
`enrich`, so adding a seventh source is a code change and a deploy. That
allowlist is deliberate — Earth Engine bills the project rather than the caller,
so a member sending EE expressions would be spending FRMS's money on code
nobody read — and any answer here has to keep that property.

It is closer than it looks. `runDated` is already a generic sampler taking
`{scale, reducer, bands, imageFor}`, and soil moisture, rainfall and temperature
are thin wrappers over it. `ee_custom_layers` already validates and stores
exactly the spec such a stage needs: asset id, asset type, band, reducer, date
window. A member-defined stage is that spec pointed at points instead of tiles,
with no EE code execution and the same allowlist posture.

The bespoke runners stay bespoke and should: terrain derives TPI at three radii,
solar and wind exposure and upstream area; NDVI and land cover do cloud and
water masking. Those are not "sample a band".

Worth doing after there is evidence somebody has hit the wall of six stages.
Composing jobs came first because it needed no new Earth Engine surface at all.

### Coverage page, reframed

`/coverage` inventories the **local raster cache** — 25.9 GB of CHIRPS, ERA5 and
NDVI files on disk. As enrichment moves to Earth Engine that cache stops
existing, and the page describes an artifact of the architecture being replaced.

The question that survives the move is field coverage: what fraction of records
actually carry each enrichment column. A mean over the 23% of rows that have
soil moisture is a different claim from a mean over all of them, and nothing in
the app currently says which you are looking at.

Keep the URL, replace the contents, and let the raster inventory go when the
cache does.

---

## Data quality

### Taxonomy resolution has stalled

Genus, family and order are populated for under 4% of the store, so every view
that groups above species is working from a small and probably unrepresentative
slice. The ranks are offered in the UI as if they were populated.

Either the resolution pass needs to run to completion, or the views that group
by an unpopulated rank need to say what fraction they are drawn from.

---

## Verification debt

### No Earth Engine layer has rendered against the real API

The catalogue's asset IDs, band names and `system:index` values were written
from documentation and from working Code Editor scripts, never executed against
Earth Engine from this codebase. The soil and forest layers in particular carry
band names (`r_cm_p`, `r_0_cm_p`) taken from a script rather than verified here.

This is why `ee-tiles` reports failures loudly and by name: a layer that comes
back blank looks exactly like ground with nothing on it. It is not a substitute
for rendering each one once and looking at it.

---

## Closed

Entries move here with the commit that closed them, so the reason an item
existed survives the fix.

### Saved filters did not sync

Recorded as `localStorage`-only, so not following a member between devices.
That was already untrue when it was written: `useSavedFilters` writes to the
`saved-filters` key and `SETTINGS_KEYS` in `useCloudSync` lists that key, so it
is snapshotted to Supabase with the other preferences. Removed rather than
fixed, since there was nothing to fix.

Noticed while working out what survives a move to a new domain — browser
storage is per-origin, so the question of which preferences live only in the
browser is the same question.

### Data export

The app computed a filtered, enriched set of records and then would not hand it
over: the only downloads that existed were chart SVG/PNG and map PNG.

Closed with `composables/dataExport.js` and an `ExportMenu` on the data table,
each finished job and each saved dataset. GeoJSON and CSV, with column
selection defaulting to what the table is showing — the enriched row runs to
about fifty columns, so which ones is a real question.

It is open rather than a membership benefit, and that was the decision the entry
asked for. The reference dataset is already a public file the app fetches by URL,
so asking somebody to sign in to download what they could already fetch directly
would be theatre; a job result is the member's own work and they had to be a
member to produce it.

Two things the encoders are careful about, both silent when wrong. A CSV field
containing a comma, a quote or a newline shifts every column after it unless it
is quoted and its quotes doubled — one species note is enough. And a cell
beginning `=`, `+`, `-` or `@` is a formula in every spreadsheet, while the text
in these fields comes from iNaturalist, which is to say from the public; those
are prefixed with an apostrophe rather than stripped, so the value survives.

### Storage access rules for job results

`ee-worker` wrote every finished job to `jobs/<user_id>/<job_id>.geojson` in the
Supabase `datasets` bucket, and `useEeJobs.fetchResult` downloaded it from the
browser with the member's own session — with no `storage.objects` policy in any
migration. A private bucket meant a member could not read their own result; a
public one meant anyone with a path could read anyone's.

Closed by `005_dataset_storage_access.sql`: the bucket is private, a member
reads their own `jobs/<uid>/` prefix, admins read everything, and there is no
write policy at all so results stay the pipeline's to write. A dataset shared
with other members is read through the server instead, because sharing lives in
the row and a path-prefix policy cannot see it.

The same work found something worse next door. `loadSource` built a storage
path straight from the slug in a job spec and read it, with no check that the
caller was allowed to — harmless while only admins could create datasets, and a
way to read anyone's private work the moment members could. The rule now lives
in `netlify/lib/dataset-access.mjs` and is applied at submission and again in
the worker.

### Bucket name is declared twice

`useEeJobs` hardcoded `'datasets'` while the server read
`SUPABASE_DATASETS_BUCKET`, so renaming the bucket sent the client looking
somewhere the server was not writing, with nothing naming the cause.

Closed with the storage policies, which made it three declarations rather than
two — a policy has to name the bucket as a SQL literal. The client now reads
`runtimeConfig.public.datasetsBucket`, and both the config and the migration say
that renaming it means changing all three together.
