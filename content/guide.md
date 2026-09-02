# Guide

What each part of this app does, and — just as importantly — what each number
does and does not mean.

One caveat runs through everything here. iNaturalist records are **opportunistic
observations, not surveys**. Somewhere with many records may have many mushrooms,
or may simply be near a trailhead. Where a view corrects for that, it says so;
where it cannot, it says that too.

This page has two halves. What follows is a tour of each part of the app. After
it, the [Option reference](#reference) documents every individual control — what
it does, and where it can mislead you. The small **?** beside a control in the
app links straight to its entry there, so you never have to come here and hunt.

---

## Map

Every observation as a point, with a grid of summaries underneath.

### Colouring and sizing

**Color by** takes any category (species, genus, cluster, land cover, year,
month, enrichment level) or any numeric field (elevation, slope, NDVI, soil
moisture, temperature, exposure). Categories get stable colours — a species is
the same colour here, in every chart, and in the legend. Numeric fields get a
light-to-dark gradient.

**Size by** scales each point by a numeric field, so two dimensions can be read
at once.

### Overlays

Grid summaries drawn under the points. 48,000 overlapping dots show where the
data is dense; a grid can show what is actually in an area.

| Overlay | What each cell shows | Watch out for |
| --- | --- | --- |
| Observation density | How many records fall in the cell | Effort. Cities and trailheads glow. |
| Species richness | How many distinct species | Also effort-sensitive — more visits find more species |
| Seasonal activity | Share of *this cell's own* finds inside the date window | Effort-neutral; needs 3+ records in the cell |
| In-season hotspots | That share, weighted by how well-sampled the cell is | A record of past finds, not a forecast |
| Most common species | The most-recorded species in the cell | Ties are broken arbitrarily |
| Wind / aspect vectors | Arrows for wind, or for the way slopes face | Says which source it used |

**Seasonal activity** and **In-season hotspots** take a date and a window
(±3 to ±60 days). They are the two worth trusting most, because dividing by each
cell's own total cancels most of the effort bias: a cell's seasonal *shape* does
not depend on how many people visited, only on when they found things.

Cell size runs from ~500 m to ~28 km. Smaller cells are more precise and noisier —
and below about a kilometre the grid stops summarising and starts drawing roughly
one cell per observation.

### Wind vectors

The observations carry terrain **aspect** — the compass direction a slope faces —
and a wind-exposure index, but no measured wind. So the overlay draws mean
terrain aspect, with arrow length showing how *consistently* a cell faces one way
and colour showing wind exposure. A short arrow means mixed terrain, not calm
air. Once the pipeline samples ERA5 wind into `wind_u` / `wind_v`, the overlay
switches to real wind by itself and relabels the legend.

### Other map controls

- **My location** — puts a dot at your position with its accuracy circle
- **Basemaps** — a light grey canvas by default, so the observations are the only
  saturated thing on screen; street, terrain and satellite are one click away.
  On top: USGS topo, USGS imagery, hillshade, relief, **hiking trails** and
  **land ownership**. Grey base plus hillshade gives relief without colour.
- **Live clustering** — k-means in the browser, by features or geography
- **Save image** — flattens the whole map, tiles and all, into a PNG

---

## Charts

### Gallery

Nineteen preset charts. Each card can be reordered, hidden, opened full screen,
or saved as PNG (shift-click for SVG, which stays sharp at any size).

Cards are a fixed height so the grid stays regular; a chart taller than its card
is clipped with a fade, and full screen shows it uncut.

### Build

Compose your own: scatter, bar, line, area, box plot, histogram, heatmap, radar
or donut. Pick fields for each axis, colour, shape, size and series, then **Save
to Charts** to keep it.

**Sort** orders grouped charts by value or by label. Largest-first answers "which
is biggest"; A–Z answers "where is X". The size cap applies before the sort, so
choosing A–Z never pushes the biggest categories off the chart.

### Style

Five palettes — including a colour-blind-safe one — three shape sets, point size,
opacity and outline, the colour ramp the grid overlays use, and per-value
overrides. Pin a species to a colour and it holds across the map and every chart.

**Shuffle colours** deals the same palette out differently, for when two species
land on shades you cannot separate. It is deterministic, so a shuffled view still
looks the same to whoever opens your shared link. Settings persist per viewer.

---

## Analysis

Statistics over whatever the filters currently select.

### What relates to what

A **Spearman** correlation matrix across every populated field. Rank-based, so a
relationship counts even when it bends — and these bend.

Each cell uses only the rows where *both* fields are present. That matters here:
soil moisture is on about 23% of rows, so dropping rows missing any field would
compute the whole matrix on an unrepresentative remainder.

Two confounds run through every pair:

- **Season.** High-elevation finds happen in summer, low ones in spring and
  autumn — which is why elevation and temperature appear to rise together
  (ρ +0.43). Hold the month still and that flattens to about zero.
- **Effort.** People record where people go.

### Species

**Fingerprints** show how far each species sits from the dataset average on every
field, in standard deviations. Positive means found higher, warmer, wetter or
later than average. Z-scores rather than raw means, because metres, degrees and
millimetres cannot be compared side by side.

**Found together** ranks species pairs by **lift** — how much more often they
co-occur than their individual frequencies predict — within the same ~5 km cell
and the same month. Lift, not a raw count, because a count would just rank the
two commonest species first whether or not they have anything to do with each
other.

### Year over year

Season timing (median day of year) and median elevation, per year, alongside the
recording effort that drives both. Median, not mean: a few winter records would
drag a mean badly.

Read the effort chart first. Recording has grown steeply, so a shift in either
trend may be a shift in who is looking.

### Data quality

Field coverage overall and by year. This is the one that tells you how much
weight the rest can carry — a chart drawn from a 23%-covered column looks exactly
as confident as one drawn from a full column. Thin fields are thin because
enrichment has not reached those rows; re-running the pipeline fills them.

---

## Data

### Species

Every species with its record count, grouped at genus, species or subspecies
level. Selecting narrows every view at once.

### Table

All observations with their enriched columns, sortable and searchable. Only the
visible rows are rendered, so the full set stays responsive.

### Fetch new

Pull fresh observations for a species straight from iNaturalist.

### Filters

Filters apply everywhere — map, charts, table, analysis.

| Filter | Notes |
| --- | --- |
| Country / state / county | Parsed from the iNaturalist place string |
| Radius | Distance from a chosen point |
| Year / month / week | ISO week, matching the week-of-year chart |
| Date range | From and to |
| Minimum records | Drops taxa below a threshold, by species or genus |

**Minimum records** is counted *after* the other filters, so it means "enough
records in what you are looking at" — narrowing to one county and asking for 25
gives well-sampled species in that county. A species seen twice cannot tell you
where it fruits.

**Saved subsets** store the whole filter state under a name and restore it in one
click. The chip describes itself from whatever is actually set.

---

## Sharing and saving

**Share** builds a link that reproduces the current view — map position, filters,
colouring, overlay and date window all travel with it. From there: QR code
(generated locally, so the link never reaches a third party), X, Bluesky,
Facebook, Reddit, email, SMS, and an iframe embed that drops the site header.

**Save image** exports charts as PNG or SVG, and the map as a PNG with its
basemap composited in.

**Accounts** are optional. Signed out, everything persists in the browser. Signed
in, settings and saved charts follow you across devices; the first sign-in on a
device merges rather than overwrites, so work done before making an account is
not lost.

---

## Coverage

Which environmental raster layers are cached, for what dates, and over what area —
the pipeline's side of the same question the Analysis page's data-quality tab
asks from the observations' side.

---

## Where the data comes from

Observations come from iNaturalist. Every environmental column is sampled from
Google Earth Engine at the observation point.

| Column | Source |
| --- | --- |
| `ndvi` | Sentinel-2 (`COPERNICUS/S2_SR_HARMONIZED`) |
| `soil_moisture` | ERA5-Land daily |
| `prcp_d0..d6` | CHIRPS daily rainfall |
| `tmax_d0..d6`, `tmin_d0..d6` | ERA5-Land daily |
| `wind_u`, `wind_v` | ERA5-Land 10 m wind |
| `land_cover` | ESA WorldCover |
| `elevation`, `slope`, `aspect` | SRTM |
| Solar / wind exposure, wetness index | Derived from terrain plus MERIT Hydro |

Wind is stored as vector components rather than a bearing because directions are
circular: the average of 350° and 10° is 180°, pointing exactly backwards.
