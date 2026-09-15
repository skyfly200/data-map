# Roadmap

What is known to be missing, and why. Ordered by what blocks something else
first, then by what is most often asked for.

Each entry says what is true today, not just what should be. An item here is a
commitment to an open question; when it is closed the entry moves to the bottom
section with the commit that closed it.

---

## Blocking

### Storage access rules for job results

`ee-worker` writes every finished job to `jobs/<user_id>/<job_id>.geojson` in the
Supabase `datasets` bucket, and `useEeJobs.fetchResult` downloads it **from the
browser, with the member's own session**. No `storage.objects` policy exists in
any migration.

So today: a private bucket means a member cannot read their own result; a public
bucket means anyone with a path can read anyone's. Neither is right, and a member
running a job is the main thing membership buys.

The decision is whether the bucket is private with a policy scoping
`jobs/<uid>/` to its owner plus admins — recommended — or public with results
treated as shareable. Either way it is one migration.

### Bucket name is declared twice

`composables/useEeJobs.js` hardcodes `'datasets'`; the server reads
`SUPABASE_DATASETS_BUCKET`. Setting that variable to anything else sends the
client looking in a bucket the server is not writing to, with no error that
names the cause. One source of truth before anything else is built on it.

---

## Wanted

### Data export

There is no way to get data out of the app. The only downloads that exist are
chart SVG/PNG and map PNG — no GeoJSON, no CSV, for either the reference dataset
or a job result.

This is the most-cited gap and the least excusable one: the app computes a
filtered, enriched set of records and then will not hand it over. Wanted:

- the current filtered selection, as GeoJSON and CSV
- a job result, from the jobs page, without going through the storage bucket
- column selection, since the full enriched row is wide

Worth deciding at the same time whether export is open or a membership benefit.

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

### Saved filters do not sync

Settings and saved charts sync to Supabase per account. Saved filters are
`localStorage` only, so they do not follow a member between devices. One table,
matching the shape of `saved_charts`.

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

Nothing yet. Entries move here with the commit that closed them, so the reason
an item existed survives the fix.
