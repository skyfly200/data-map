# VDEBT-1 No Earth Engine layer has rendered against the real API

The catalogue's asset IDs, band names and `system:index` values were written from documentation and from working Code Editor scripts, never executed against Earth Engine from this codebase. The soil and forest layers in particular carry band names (`r_cm_p`, `r_0_cm_p`) taken from a script rather than verified here.

This is why `ee-tiles` reports failures loudly and by name: a layer that comes back blank looks exactly like ground with nothing on it. It is not a substitute for rendering each one once and looking at it.

There is a way to run that pass: `scripts/verify_ee_layers.mjs` resolves every layer at its defaults, runs its prepare and count steps, mints a tile template, and prints a pass/fail line per layer, exiting non-zero if any failed. What is left is not code — it is running it once on a deployment that has Earth Engine credentials and eyeballing the layers that pass, because a mint proves the asset and bands but not that the pixels are right.
