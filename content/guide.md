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

**Color by** takes any category — every taxonomic rank from kingdom down to
species, plus common name, the rank a record was identified to, cluster, land
cover, year, month, enrichment level — or any numeric field (elevation, slope,
NDVI, soil moisture, temperature, exposure). Categories get stable colours — a
species is the same colour here, in every chart, and in the legend. Numeric
fields get a light-to-dark gradient.

**Size by** scales each point by a numeric field, so two dimensions can be read
at once.

### Heatmaps

Grid summaries computed from the observations and drawn under the points. 48,000
overlapping dots show where the data is dense; a grid can show what is actually
in an area.

These are distinct from the **layers** in the map's layers control. A layer is
imagery from somebody else's server — hillshade, rainfall, land ownership. A
heatmap is our own numbers, binned.

| Heatmap | What each cell shows | Watch out for |
| --- | --- | --- |
| Observation density | How many records fall in the cell | Effort. Cities and trailheads glow. |
| Species richness | How many distinct species | Also effort-sensitive — more visits find more species |
| Seasonal activity | Share of *this cell's own* finds inside the date window | Effort-neutral; needs 3+ records in the cell |
| In-season hotspots | That share, weighted by how well-sampled the cell is | A record of past finds, not a forecast |
| Most common species | The most-recorded species in the cell | Ties are broken arbitrarily |
| Land cover | The most common land-cover class across the cell's finds | Only the winner shows; a 50/50 cell looks pure |
| Wind / aspect vectors | Arrows for wind, or for the way slopes face | Says which source it used |
| *Environmental fields* | The cell mean of one enriched value | Sampled at the finds, so a blank cell means nobody looked |

**Seasonal activity** and **In-season hotspots** take a date and a window
(±3 to ±60 days). They are the two worth trusting most, because dividing by each
cell's own total cancels most of the effort bias: a cell's seasonal *shape* does
not depend on how many people visited, only on when they found things.

The **environmental field** heatmaps average one enriched value across each cell:
7-day rainfall, temperature, soil moisture, wetness index (TWI), slope, aspect,
sun and wind exposure, NDVI, NDMI, elevation. They are the honest version of an
environmental raster — every observation already carries these values, and the
grid turns them into a surface. What they are *not* is coverage: a cell has a
value only where somebody recorded a find, so an empty cell means nobody went
there, not that the ground is dry or flat. Aspect is circular and gets a compass
key rather than a low-to-high bar, because 359° and 1° are neighbours.

Cell size runs from ~500 m to ~28 km. Smaller cells are more precise and noisier —
and below about a kilometre the grid stops summarising and starts drawing roughly
one cell per observation.

Cells are **hexagons** by default. Every neighbour of a hex is the same distance
away and shares a full edge, where a square's diagonal neighbours are 1.41×
further and meet at a point — so a hex grid reads as a surface rather than as a
grid, and a cluster of finds is not split differently depending on how it lands
against the axes. Squares are still available under Style, sized to the same area
so switching does not change the resolution.

**Heatmap opacity** and **Map layer opacity** are separate sliders under Style,
because they are separate stacks: a faint hillshade under a solid heatmap is the
normal case, not an odd one.

### Wind vectors

The observations carry terrain **aspect** — the compass direction a slope faces —
and a wind-exposure index, but no measured wind. So the heatmap draws mean
terrain aspect, with arrow length showing how *consistently* a cell faces one way
and colour showing wind exposure. A short arrow means mixed terrain, not calm
air. Once the pipeline samples ERA5 wind into `wind_u` / `wind_v`, the heatmap
switches to real wind by itself and relabels the legend.

### Other map controls

- **My location** — puts a dot at your position with its accuracy circle
- **Basemaps and layers** — a light grey canvas by default, so the observations
  are the only saturated thing on screen; street, terrain and satellite are one
  click away. On top, grouped by subject:
  - *Terrain* — hillshade, USGS topo, USGS imagery, OpenTopoMap relief. Grey base
    plus hillshade gives relief without colour.
  - *Weather* — live US radar, US rainfall over the past 24 hours, global
    satellite rainfall, land surface temperature.
  - *Ground* — ESA WorldCover land cover at 10 m, SMAP soil moisture.
  - *Vegetation* — MODIS NDVI greenness.
  - *Context* — place labels, **hiking trails**, **land ownership**.

  Every measured layer carries a key, collected into one **Map layers** panel that
  appears as you switch layers on. The ones that vary by day carry a date picker
  there too; it opens a few days back, because every satellite product has
  latency and asking for today returns blank tiles that would read as "no rain"
  rather than "not processed yet".

  These are **not** the same as the heatmaps. A layer is somebody else's raster,
  drawn everywhere because they measured it everywhere, and it shows current or
  recent conditions over historical finds — so it cannot explain the finds. The
  rainfall, temperature, soil-moisture and NDVI **heatmaps** use what the pipeline
  sampled at each observation's own date, which can.

  A layer that cannot be reached says so, rather than drawing nothing: an empty
  ownership layer would otherwise read as "no public land here".
- **Live clustering** — k-means in the browser, by features or geography
- **Save image** — flattens the whole map, tiles and all, into a PNG
- **Offline** — under Settings, and on the Options page

### Offline

A map of where things grow is most often read standing in the place it
describes, which is where there is least likely to be a connection. Three things
can be saved into the browser, separately, because they cost very different
amounts:

- **The app** — every page and the code behind it, so Charts and Analysis open
  offline too and not just the page you happened to be on.
- **Observations** — the dataset the map, table and charts all read from. This
  is the big one.
- **Map tiles for this view** — basemap imagery for the area on screen, plus
  however many zoom levels closer you ask for. Pan to where you are going first;
  the tile count and a rough size are shown before anything downloads, because
  the count roughly quadruples per extra zoom level.

Nothing is saved on its own beyond the app shell — pulling a dataset and a few
hundred tiles onto someone's mobile data unasked is not a feature. Saved data
lives in this browser on this device, is not uploaded, does not follow your
account, and goes when you clear the browser's site data. The app is also
installable to a home screen.

Tile services set their own terms on bulk downloading. Save the area you are
going to, not a region.

### Taxonomy

Every observation carries its full ancestry — **kingdom, phylum, class, order,
family, genus, species** — resolved against iNaturalist when the record is
fetched. All seven are dimensions you can colour by, group by, filter to, and
run the analysis over.

This replaces splitting the species name on its spaces, which gave a genus only
when a record happened to be identified to species and could not reach family or
above at all. A record identified only to *Amanitaceae* used to have a "species"
of "Amanitaceae" and a genus to match; it now has a family, and an empty genus
and species. That is the honest answer — and it means filtering at species level
quietly drops every record nobody pinned down that far, so filter at the coarsest
rank that answers your question.

On **Data → Species**, the **Rank** selector decides both what the list shows and
what the filter applies to, so the same picker narrows a view to one kingdom or
to one species. Only ranks the loaded dataset populates are offered — a dataset
exported before this existed carries species and genus and nothing else.
Switching rank clears the selection, because the names do not carry across.

**Fetch new** takes a name at any rank. "Amanita muscaria" imports one species,
"Amanitaceae" the family, "Fungi" the kingdom — and a dataset mixing kingdoms
stays comparable, because every record knows its own place in the tree.

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

Five palettes — including a colour-blind-safe one — point size,
opacity and outline, the colour ramp and opacity the grid heatmaps use, the
cell shape they bin into, the opacity of the reference layers, and per-value
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
colouring, heatmap and date window all travel with it. From there: QR code
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
