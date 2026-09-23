# WANT-9 GBIF API direct loading

Today GBIF data enters only through a manual CSV export. The GBIF Occurrence API (`https://api.gbif.org/v1/occurrence/search`) is public and requires no authentication for reads.

A new Netlify function wrapping it could accept the same filter parameters the app already uses — taxon, bounding box, date range, coordinate uncertainty threshold — and return a GeoJSON FeatureCollection in the same shape `useObservations` expects. Member flow: search by species name, see record count, load directly into dataset.

Important constraints:
- GBIF search returns up to 100,000 records per request (paginated at 300/page). Enforce a per-request cap and surface record counts before fetching.
- Coordinate uncertainty filtering should default on to match the quality bar the CSV importer already applies.
- `ISSUE-2` (large imports timing out) should be fixed first — same timeout risk applies here. Streaming the response into Supabase Storage incrementally rather than building full GeoJSON in memory is the right fix for both.

Worth doing alongside any work to improve the import experience, and before `WANT-2` since GBIF API records are a natural source for member-defined stages.
