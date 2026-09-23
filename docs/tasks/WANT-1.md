# WANT-1 Species distribution modeling

MaxEnt ships end to end: `netlify/lib/maxent.mjs` holds the predictor registry, background plan, cross-validation, and `buildSuitabilityImage`; the worker runs `runModel` for a `kind: 'model'` job; `model-tiles` re-mints an expired surface from the stored spec; the jobs page has an Enrich / Model toggle with predictor selection; the map legend shows the AUC score, the effort-bias confound note, and a refresh button when tiles expire.

Open items:
- None of it has run against the real Earth Engine API — tested against a stub only (see `VDEBT-1`).
- GeoTIFF export (`V12-MOD-4`) is a follow-up tracked in `refactor.md`.

Worth exercising once enrichment output is loaded as datasets often enough that "and then what" is a real question.
