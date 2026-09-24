# Refactor

Technical debt and enhancement tasks. An item here is open work; when closed, delete the entry and its task file.

---

## Technical Debt

- [ ] `DEBT-7` **Netlify Backend TypeScript Migration** — migrate `netlify/lib/*.mjs` and `netlify/functions/*.mjs` to `.ts` with esbuild/tsup. See `tasks/DEBT-7.md`.
- [ ] `DEBT-8` **MushroomMap Decomposition** — split ~2,800-line component into focused composables and extract toolbar. See `tasks/DEBT-8.md`.

---

## UI & UX

- [ ] `V12-UI-1` **Pipeline UX overhaul** (was: Intuitive Navigation): guided experience from area/source pick → enriched dataset on map. Wayfinder map at [#140](https://github.com/skyfly200/data-map/issues/140) holds all open decisions; see also `WANT-15` and `WANT-16` in the roadmap.
- [ ] `V12-UI-2` **Area picker improvement**: replace raw lat/lng number inputs with a map-draw or geocoder interaction. Decision ticket: [#144](https://github.com/skyfly200/data-map/issues/144).
- [ ] `V12-UI-3` **Visual Hierarchy**: Improve contrast and layout of side panels
- [ ] `V12-UI-4` **Interactive Data Previews**: Instant visual summaries on dataset selection
- [ ] `V12-UI-5` **User observation uploads**: CSV/GeoJSON upload to Supabase Storage as a new dataset source. Decision ticket: [#143](https://github.com/skyfly200/data-map/issues/143).
- [ ] `V12-UI-6` **Post-job CTA flow**: replace the flat "open on map / open on charts / save" menu with a guided next-step prompt. Depends on WANT-11 (composable state persistence). Decision ticket: [#145](https://github.com/skyfly200/data-map/issues/145).
- [ ] `V12-UI-7` **Mobile Map Controls Discovery**: Replace compact-mode hidden controls with accessible bottom-sheet or collapsible toolbar
- [ ] `V12-UI-8` **Fluid Breakpoints**: Supplement binary compact/not-compact with CSS `@media` rules at tablet widths

## Advanced Modeling

- [ ] `V12-MOD-1` **Ensemble Modeling**: Average predictions from multiple model runs
- [ ] `V12-MOD-2` **Projection Tools**: Project models to future climate scenarios (CMIP6)
- [ ] `V12-MOD-3` **Batch Processing**: Train models for multiple species simultaneously
- [ ] `V12-MOD-4` **Model Export**: Download suitability rasters as GeoTIFF
- [ ] `V12-MOD-5` **Threshold Optimization**: Automatic threshold selection (MaxSSS, 10th percentile)

## Data Quality & Ethics

- [ ] `V12-ETH-1` **Sampling Bias Correction**: Automated thinning and bias file generation (see `WANT-7`)
- [ ] `V12-ETH-2` **Spatial Autocorrelation Checks**: Warn about clustered occurrence records
- [ ] `V12-ETH-3` **Extrapolation Risk Maps**: Highlight areas outside training environmental space (MOP/MEX)
- [ ] `V12-ETH-4` **Sensitive Species Protection**: Automatic coordinate obscuring for threatened species

## Collaboration & Sharing

- [ ] `V12-COLL-1` **Public Model Gallery**: Browse and reuse models from other users
- [ ] `V12-COLL-2` **Team Workspaces**: Shared projects for research groups
- [ ] `V12-COLL-3` **Model Citation Generator**: Auto-generate citations for published models
- [ ] `V12-COLL-4` **Export to R/Python**: Generate reproducible scripts for external analysis

## Performance

_All PERF-1–5 items shipped: LazyVisible for gallery charts (PERF-1), single-pass bucketing for tempHighLowDist/elevationData (PERF-2), single-pass coverageData counter (PERF-3), 40 ms coloring debounce (PERF-4), debounced heatmap rebuild (PERF-5)._

## Documentation

- [ ] `V12-DOC-2` **Video Guides**: Short screencasts for key workflows
- [ ] `V12-DOC-4` **Example Datasets**: Pre-loaded sample data for practice runs
