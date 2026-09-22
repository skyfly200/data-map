# Roadmap

## Wants / Future Features

### MaxEnt Observation Bias Correction

Citizen science platforms (iNaturalist, GBIF) produce severe sampling bias because observations cluster near human infrastructure (roads, trails). Uncorrected, MaxEnt learns human travel patterns instead of true ecological niches, producing suitability maps that mirror trail networks rather than habitat.

**Critical rules:**
- **Do NOT** feed human footprint, population density, or trail distance layers as standard MaxEnt environmental covariates — the model will interpret them as positive habitat preferences.
- **Do** use human traffic and infrastructure data exclusively for background sample selection to cancel reporting artifacts.

**Strategy A — Bias Grid (Recommended):**
1. Build a sampling effort surface by combining trail proximity rasters, human population density (WorldPop/LandScan), and general observation density into a single continuous raster.
2. Pass the raster to MaxEnt via the `biasfile=sampling_effort.tif` argument.
3. This instructs MaxEnt to draw more pseudo-absence background points near high-traffic areas (where observers actually look) and fewer in inaccessible terrain, canceling human travel bias.

**Strategy B — Target Group Background (TGB) with Conspicuousness Filtering:**
1. Restrict background sampling to ecologically and morphologically comparable taxa (e.g., large, charismatic macrofungi like visible boletes/amanitas) rather than all species.
2. Programmatically drop records from casual/one-time users; strictly prioritize Research Grade observations to minimize misidentification noise.
