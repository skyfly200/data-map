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

### Species distribution modeling (the point of the enrichment)

The home page now frames Nexstrata as an ecosystem-modeling platform with
enrichment as one stage of four — observe, enrich, model, predict. The first
two ship; the model and predict stages do not yet, and the page marks that
section "Roadmap" rather than claiming otherwise.

The intended method is a presence-only species distribution model, MaxEnt the
obvious first: the enrichment already produces exactly the feature matrix such a
model reads — every environmental layer sampled at each presence point, at its
own date. The pieces were background/pseudo-absence sampling, the fit itself, and
projecting the fitted surface back across a region as a habitat suitability
raster the map can draw beside the layers it was built from.

The modelling core now ships as `netlify/lib/maxent.mjs`: a predictor registry
(terrain, NDVI and standing soil moisture, all free and static), request
validation (`normaliseModelSpec`), the effort-aware background plan
(`backgroundPlan`, which never draws fewer background points than presences and
weights toward them by default), and `buildSuitabilityImage`, which samples the
predictors at the presences and at random background, trains
`ee.Classifier.amnhMaxent`, and classifies the stack into a 0–1 suitability
image. It is split so only that last function touches `ee`, and it is tested end
to end against a stub.

The `model` job kind now runs end to end on the server. `normaliseSpec` routes a
`kind: 'model'` spec to `normaliseModelSpec`, sharing the same source checker so a
model's presences come from the same dataset or bounding box an enrichment job's
do; `job-queue` prices it with `estimateModelUnits` and draws its bar with
`modelPlan`; and the worker branches on the kind, carrying the presences it
already loads (`loadSource`) into `runModel`, which builds the presence
FeatureCollection, fits the model, mints a tile template from the fitted surface,
and stores it in the job's `result_meta` — a raster has no GeoJSON file, so the
result is a tile URL rather than a stored feature collection.

The sampling-bias problem is not a footnote here: presence-only modeling
inherits the observer-effort bias the caveats already name, so background
sampling has to be weighted by effort and every surface labelled with the
confounds behind it, the same way the density heatmaps already are. The
background plan holds the first half; the surface labelling is still owed.

The UI now exists. The jobs page carries an Enrich / Model toggle: model mode
offers the predictors instead of the enrichment stages, and a finished model
shows a "view suitability on map" action that draws the surface as an overlay
with its legend beside the layers it was built from.

The trained model and its surface are stored durably rather than only for as long
as one map id lasts. The model's definition — predictors, region, source and the
cross-validation score — persists in the job row, and the surface is re-servable
from it on demand: `model-tiles` re-mints the template from the stored model
(sharing the result cache, so within its TTL it is a blob read rather than any
Earth Engine work), the jobs page fetches a fresh one before drawing, and when a
drawn surface's tiles do expire the map's legend offers to refresh it in place
from the saved model rather than sending the viewer back to re-run the job. What
is not yet stored is the raster itself as a file — an Earth Engine export to
GeoTIFF for download (V12-MOD-4) is the heavier, asynchronous follow-up.

The per-date layers the first cut left out now have their honest place. A
suitability surface is a claim about a place, not a day, so a per-record daily
value has nothing to project onto a pixel; the weather layers therefore return as
climate normals — `precip_normal` and `temp_normal`, the multi-year means, which
ARE a property of the place — offered as predictors alongside the terrain and
vegetation ones. Phenology stays out on purpose: "when in the year" is not a
property of a pixel, so it belongs to a per-date question the static surface does
not ask.

A surface now comes with a number for how much to trust it. `runModel` runs a
spatially blocked cross-validation — presences and background assigned to folds
by the 0.25° square they sit in, not at random, so a test point is not judged
beside a training neighbour it is correlated with — trains a fold out at a time,
and reports the held-out AUC as its mean and spread across folds. The AUC itself
is computed in JavaScript (`rocAuc`, `crossValidationSummary`) from the held-out
predictions Earth Engine returns, so the arithmetic that judges the model is
tested without one. The score rides in `result_meta.cv` and shows on the job and
in the map legend, with a plain grade beside it and an honest "not scored" when
there are too few presences (`MIN_CV_PRESENCES`) for it to mean anything.

None of it has run against the real Earth Engine API yet — `buildSuitabilityImage`,
`crossValidate` and `runModel` are tested against a stub, the same verification
debt the map layers carry below.

Worth finishing once the enrichment output is being loaded as datasets often
enough that "and then what" is a real question rather than a hypothetical.

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

### Member-supplied Earth Engine credentials

Every job runs under one service account against one Cloud project, so all
Earth Engine spend bills FRMS. That single pool is the whole reason `quotas.mjs`
exists: the monthly units, the per-day and concurrency caps, and the admin floor
all divide one budget fairly rather than protect against any one member. A member
who needs more than their share, or who wants to run heavier processing than the
shared budget should carry, has no answer today but "ask an admin to raise the
number", which moves the cost onto FRMS rather than onto them.

The way out is to let a member run their own jobs under their own Earth Engine
project. `initEarthEngine` already takes a service-account key from the
environment; the change is to take the job owner's stored credential instead when
they have one, falling back to the shared account when they do not. A job on a
member's own project spends the member's own Earth Engine budget, so `checkQuota`
skips the shared monthly cap for it — they are metered by Google, not by the
pool — while the point and concurrency caps stay, since those protect the worker
and the throttle, not the budget.

The hard part is not the wiring, it is holding the credential. A Google
service-account key is a long-lived secret that can read and spend against the
member's whole project, so storing one means a service-role-only column
encrypted at rest with a key that is not in the same table, a validation
round-trip that mints one throwaway tile before the key is trusted, and a way to
rotate or revoke it that a member can reach without an admin. The safer shape is
OAuth — the member authorises Nexstrata against their EE project and the app
holds a refresh token rather than a raw key — but that is a consent flow and a
token store rather than a form field, and it is more to build. Either way the
credential never reaches the browser and never appears in a job spec, the same
posture the custom-layer and dataset-access work already took.

There is a licensing edge to name in the UI, not just the code: Earth Engine
distinguishes commercial from noncommercial use per Cloud project, so a member
pointing the app at their own project is asserting their project's terms cover
what they are about to run. The guide already walks through making a Cloud
project and registering it for Earth Engine (`guide/layers`), so the member-facing
half is mostly written.

Worth doing when a member's needs exceed what the shared budget should fund —
the first real "I need my own quota" is the signal, the same way composing jobs
waited for the first real "and then what". Until then the admin floor and a
raised per-member quota cover it.

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

### Navigation and the pages behind it

The nav grew around the tools that existed when each was added, and it shows.
`/jobs` sits in it as a top-level destination, but a job is something you start
and then wait on, not a place you go — it belongs behind the thing that produces
it (the map, a dataset) or under an account menu, not beside Map and Charts. And
the pages that have since become the ones worth landing on are not all in the nav
at all.

Two moves, one question. Remove `/jobs` from the nav (the page stays, reached
from where a job is launched), and add the pages that earn a top-level slot to
it. The open question is which those are, and in what order — the nav is the
app's table of contents, so what is in it is a claim about what the app is for.

### A dashboard worth landing on

The default view is thin: it opens on not much, and the interesting state — how
many observations, how fresh, what has been enriched, what is worth looking at
today — is scattered across pages a member has to go find. A landing dashboard
should carry that at a glance.

More widgets, and better ones: totals and recency, a small map or heat preview,
the enrichment coverage the reframed Coverage page will compute, recent jobs and
datasets, maybe a "finds like today's conditions" prompt once the model can
answer it. The work is partly which widgets (each has to answer a real question,
not decorate), partly the layout that makes them read as one view rather than a
pile, and partly the shared state so a widget reflects the same filters the rest
of the app is under.

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

There is now a way to run that pass rather than only describe it:
`scripts/verify_ee_layers.mjs` resolves every layer at its defaults, runs its
prepare and count steps, mints a tile template, and prints a pass/fail line per
layer, exiting non-zero if any failed. What is left is not code — it is running
it once on a deployment that has Earth Engine credentials and eyeballing the
layers that pass, because a mint proves the asset and bands but not that the
pixels are right. Until then the band names remain unverified; the harness just
makes verifying them a command rather than a project.

---

## Closed

Entries move here with the commit that closed them, so the reason an item
existed survives the fix.

### Taxonomy resolution has stalled

Genus, family and order were populated for under 4% of the store, so every view
that grouped above species was working from a small and probably unrepresentative
slice while the UI offered the ranks as if they were populated. The entry asked
for one of two fixes: run the resolution pass to completion, or have the views say
what fraction they are drawn from.

Closed with the second. `composables/fieldCoverage.ts` computes the fraction of
records carrying a field, and `coverageNote` turns a thin rank into the sentence
a view shows; the map's "colour by" and the chart builder now carry it when the
points or bars are grouped by a rank most records lack, naming the count and the
percent so a key of five families no longer reads as the whole dataset. Genus and
species are exempt — genus is split from the binomial when missing, so both are
complete — and only the resolved-from-ancestry ranks (family and above) trip the
note. The deeper fix, running the resolution pass to completion, is still worth
doing; the app no longer lies about the ranks while it waits.

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
