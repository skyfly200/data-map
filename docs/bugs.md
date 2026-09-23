# Bugs & Verification Debt

Open issues and unverified assumptions. When closed, delete the entry and its task file.

---

## Known Issues

- [ ] `ISSUE-1` LayerManager state persistence occasionally fails on mobile Safari. See `tasks/ISSUE-1.md`.
- [ ] `ISSUE-2` Large GBIF exports (>10k records) may timeout during import. See `tasks/ISSUE-2.md`.
- [ ] `ISSUE-3` Chart rendering slows with >50 data points in Saved Charts widget. See `tasks/ISSUE-3.md`.
- [ ] `ISSUE-4` Earth Engine asset validation doesn't check geometry types comprehensively. See `tasks/ISSUE-4.md`.

---

## Verification Debt

- [ ] `VDEBT-1` No Earth Engine layer has rendered against the real API — band names and asset IDs are untested outside a stub. Run `scripts/verify_ee_layers.mjs` on a deployment with EE credentials. See `tasks/VDEBT-1.md`.
