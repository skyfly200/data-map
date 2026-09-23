# WANT-2 Member-defined enrichment stages

`STAGES` is six hardcoded entries and `normaliseSpec` refuses any `kind` but `enrich`, so adding a seventh source is a code change and a deploy. That allowlist is deliberate — Earth Engine bills the project rather than the caller, so a member sending EE expressions would be spending FRMS's money on code nobody read — and any answer here has to keep that property.

It is closer than it looks. `runDated` is already a generic sampler taking `{scale, reducer, bands, imageFor}`, and soil moisture, rainfall and temperature are thin wrappers over it. `ee_custom_layers` already validates and stores exactly the spec such a stage needs: asset id, asset type, band, reducer, date window. A member-defined stage is that spec pointed at points instead of tiles, with no EE code execution and the same allowlist posture.

The bespoke runners stay bespoke and should: terrain derives TPI at three radii, solar and wind exposure and upstream area; NDVI and land cover do cloud and water masking. Those are not "sample a band".

Worth doing after there is evidence somebody has hit the wall of six stages. Composing jobs came first because it needed no new Earth Engine surface at all.
