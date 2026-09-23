# Context

Domain glossary for the data-map PWA. Implementation details belong in code and docs/; only canonical terms live here.

---

## Terms

**Observation** — A single occurrence record sourced from iNaturalist or GBIF. Carries a location (lat/lon), taxon, date, and raw metadata. The atomic unit the entire pipeline operates on.

**Dataset** — A named, persisted collection of Observations. May be raw (import only) or enriched (has EE columns attached). Survives navigation and is the subject of modelling runs.

**EnrichmentJob** — A request to the GEE pipeline to append environmental variables to a Dataset's Observations. Runs under the FRMS shared EE service account (or a member's own project at higher tiers). Produces a set of EE columns on each matched Observation row.

**EE Column** — An environmental variable sampled from Earth Engine and attached to an Observation (e.g. elevation, landcover class, daily rainfall). The enrichment pipeline's output unit.

**Stage** — A named group of EE Columns that share the same Earth Engine asset and sampling logic. Defined in the server-side allowlist (`STAGES` in `netlify/lib/ee-pipeline.mjs`); members select stages, never write EE expressions.

**MaxEntModel** — A species distribution model trained from a Dataset's enriched Observations using the MaxEnt algorithm. Produces a suitability surface (raster) over a study area.

**Layer** — A visual overlay on the interactive map. Types: marker cluster (raw Observations), heatmap (density surface), EE raster tile (any Earth Engine image), suitability surface (MaxEntModel output).

**Member** — An authenticated user with active FRMS membership. Members may submit EnrichmentJobs and train MaxEntModels against the shared EE quota. Tiers: `free` (view only), `member`, `perpetual`, `admin`.

**FRMS** — The Fungal Recording and Mapping Society. The organisation this app is built for. Members share a pooled EE billing project; quota is metered per member.

**Quota** — The monthly Earth Engine unit budget assigned to a Member. Enforced server-side before any job is admitted. Default: 500 units/month for new members; uncapped floor for admins.

**Foray** — An FRMS field event where members record observations in the field, typically on mobile. A release milestone.
