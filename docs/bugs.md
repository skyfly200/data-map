# Bugs & Verification Debt

Open issues and unverified assumptions. When closed, delete the entry and its task file.

---

## Known Issues

- [ ] `ISSUE-2` Large GBIF exports (>10k records) may timeout during import. See `tasks/ISSUE-2.md`.
- [ ] `ISSUE-4` Earth Engine asset validation doesn't check geometry types comprehensively. See `tasks/ISSUE-4.md`.
- [ ] `ISSUE-6` **Trained models never get a `model_results` row.** Only the register-asset path inserts one (`modeling-maxent.mjs`), and `model_runs.status` is never moved off `pending` by the worker. So the contributions/ranges write-back (`model-persist.mjs`) updates zero rows for trained models, and `model-tiles?model=<config>` only resolves registered ones. The foray layers work around it by reading `ee_jobs.result_meta` (`foray-layers.mjs`). Fix: have the worker mark the run succeeded and upsert a `model_results` row (needs `suitability_asset_path` relaxed to nullable, since trained surfaces are refit, not stored as assets).

_Fixed this session:_
- `ISSUE-5` **EE asset path validation error message** was misleading ("Expected format: users/username/project/dataset" even though 2-segment paths are valid). Fixed in `netlify/lib/ee-assets.mjs` — error message now shows both legacy and modern path formats and the UI placeholder matches. Also fixed `datasets.mjs` to return 503 (not 400) for server-side EE configuration errors.


---

## Verification Debt

- [ ] `VDEBT-2` **WANT-17 access and foray, nothing live-tested.** Migrations 011-015 unapplied. No Colorado load run. RIDB, Overpass and PAD-US URL/field names (`Mang_Name`, `Des_Tp`, `Pub_Access`) unverified. `/foray`, `/areas` and the map Access layer not checked in a browser against real data. Predictor ranges, the ensemble/habitat layers (`foray-layers`), migration 016 and the collecting-rule `Loc_Mang` match have never run against real Earth Engine or PAD-US.
- [ ] `VDEBT-3` **Member EE credentials (WANT-3)**: validation, encrypted storage and the `own_project` worker path never run against real Earth Engine; needs migration 011 and `EE_CREDENTIAL_KEY`; no settings UI.
- [ ] `VDEBT-4` **Vercel adapter (WANT-8)** and the Supabase storage backend never deployed; GeoTIFF export (`model-tiles?download=1`) never run against real EE; observation uploads (WANT-16) and the 404 page not browser-checked.

_None other open. VDEBT-1 EE layer rendering was diagnosed and all confirmed breaks were fixed; TREEMAP private-path access requires live verification against a real EE deployment._
