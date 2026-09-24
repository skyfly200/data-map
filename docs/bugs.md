# Bugs & Verification Debt

Open issues and unverified assumptions. When closed, delete the entry and its task file.

---

## Known Issues

- [ ] `ISSUE-2` Large GBIF exports (>10k records) may timeout during import. See `tasks/ISSUE-2.md`.
- [ ] `ISSUE-4` Earth Engine asset validation doesn't check geometry types comprehensively. See `tasks/ISSUE-4.md`.

_Fixed this session:_
- `ISSUE-5` **EE asset path validation error message** was misleading ("Expected format: users/username/project/dataset" even though 2-segment paths are valid). Fixed in `netlify/lib/ee-assets.mjs` — error message now shows both legacy and modern path formats and the UI placeholder matches. Also fixed `datasets.mjs` to return 503 (not 400) for server-side EE configuration errors.


---

## Verification Debt

_None open. VDEBT-1 EE layer rendering was diagnosed and all confirmed breaks were fixed; TREEMAP private-path access requires live verification against a real EE deployment._
