# WANT-7 MaxEnt observation bias correction

Citizen science platforms produce severe sampling bias — observations cluster near human infrastructure. Uncorrected, MaxEnt learns human travel patterns instead of ecological niches.

**Critical rule:** Do not feed human footprint, population density, or trail distance as standard environmental covariates. These layers belong only in background sample weighting.

**Strategy A — Bias Grid (recommended):** Build a sampling effort surface combining trail proximity rasters, human population density (WorldPop or LandScan), and general observation density into a single continuous raster. Pass via `biasfile=sampling_effort.tif`. MaxEnt draws more pseudo-absence background points near high-traffic areas, canceling out observer travel bias.

**Strategy B — Target Group Background (TGB):** Restrict pseudo-absence background to ecologically and morphologically comparable taxa (e.g. large, charismatic macrofungi) that attract the same observers. Drop records from casual or one-time users; prioritise Research Grade observations.

`V12-ETH-1` in `refactor.md` covers the automated thinning and bias file generation that Strategy A requires.
