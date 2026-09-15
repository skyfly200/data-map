# Guide

Nexstrata is a workbench for environmental data, built on Google Earth Engine.
It samples raster layers at point coordinates, renders computed layers as map
tiles, and feeds the results into the map, the charts and the statistics.

Its founding dataset is mushroom observations from iNaturalist, enriched with
the ground each find sat on, and that is still what the map opens on. But the
observations are an example of the shape rather than the limit of it: anything
with a coordinate and a date can go through the same path.

This guide covers what each part of the app does and what each number means.

> **Note** iNaturalist records are opportunistic observations, not surveys. An
> area with many records may have many mushrooms, or may simply be near a
> trailhead. Where a view corrects for this, it says so. Where it cannot, it says
> that too.

## Start here

1. Open the **Map**. Every observation is a point; the colors are environmental
   clusters by default.
2. Use **Points** to color or size by something else, and **Heatmap** to draw a
   grid summary underneath.
3. Open **Layers** for the overlays — fire history, forest type, soil — and
   **Basemap** for the map underneath them.
4. Narrow what you are looking at on **Data → Filters**. Filters apply to every
   view at once.
5. Click any point to open its record in the side panel.

Beyond reading what is already here, the two things the platform is for:

- **[Run an enrichment job](/jobs)** — sample every environmental layer at your
  own points, over your own area and dates. A membership benefit.
- **[Publish your own layer](#your-own-earth-engine-layers)** — compute
  something in Earth Engine, export it as an asset, and register it so it draws
  on the map like any built-in.

Every control has a small **?** beside it that links to its entry in the
[Option reference](#reference) at the bottom of this page.

---

## Map

Every observation as a point, with an optional grid of summaries underneath.

### Color and size

Open the **Points** control.

- **Color by** accepts any category — every taxonomic rank from kingdom to
  species, plus common name, identification rank, cluster, land cover, year,
  month and enrichment level — or any numeric field. Categories get stable
  colors: a species is the same color on the map, in every chart and in the
  legend. Numeric fields get a light-to-dark gradient.
- **Size by** scales each point by a numeric field, so you can read two
  dimensions at once.

### Heatmaps

Open the **Heatmap** control. A heatmap bins the observations into a grid and
shades each cell by a summary statistic. Where 48,000 overlapping dots show only
where the data is dense, a grid shows what is actually in an area.

| Heatmap | What each cell shows | Watch out for |
| --- | --- | --- |
| Observation density | How many records fall in the cell | Effort. Cities and trailheads glow. |
| Species richness | How many distinct species | Also effort-sensitive: more visits find more species |
| Seasonal activity | Share of *this cell's own* finds inside the date window | Effort-neutral; needs 3+ records in the cell |
| In-season hotspots | That share, weighted by how well-sampled the cell is | A record of past finds, not a forecast |
| Most common species | The most-recorded species in the cell | Ties are broken arbitrarily |
| Land cover | The most common land-cover class across the cell's finds | Only the winner shows; a 50/50 cell looks pure |
| Wind / aspect vectors | Arrows for wind, or for the way slopes face | Says which source it used |
| *Environmental fields* | The cell mean of one enriched value | Sampled at the finds, so a blank cell means nobody looked |

**Seasonal activity** and **In-season hotspots** take a date and a window of ±3
to ±60 days, which appear in the same panel when you choose either. These two are
the most trustworthy, because dividing by each cell's own total cancels most of
the effort bias — a cell's seasonal *shape* does not depend on how many people
visited, only on when they found things.

The **environmental field** heatmaps average one enriched value across each
cell: 7-day rainfall, temperature, soil moisture, wetness index, slope, aspect,
sun and wind exposure, NDVI, NDMI and elevation.

> **Caution** A field heatmap is not a coverage map. A cell has a value only
> where somebody recorded a find, so an empty cell means nobody went there — not
> that the ground is dry or flat.

Aspect is circular, so it gets a compass key rather than a low-to-high bar: 359°
and 1° are neighbours.

#### Cell size and shape

Cell size runs from about 100 m to about 28 km. Smaller cells are more precise
and noisier; below roughly a kilometre the grid stops summarising and starts
drawing about one cell per observation.

Cells are **hexagons** by default. Every neighbour of a hexagon is the same
distance away and shares a full edge, where a square's diagonal neighbours are
1.41× further and meet at a point. A hex grid therefore reads as a surface rather
than a grid, and a cluster of finds is not split differently depending on how it
lands against the axes. Squares are available under **Style**, sized to the same
area so switching does not change the resolution.

**Heatmap opacity** and **Map layer opacity** are separate sliders under
**Style**, because they control separate stacks — a faint hillshade under a solid
heatmap is the normal case.

### Wind and aspect vectors

The observations carry terrain **aspect** (the compass direction a slope faces)
and a wind-exposure index, but no measured wind. The vector heatmap therefore
draws mean terrain aspect: arrow length shows how *consistently* a cell faces one
way, and color shows wind exposure.

> **Note** A short arrow means mixed terrain, not calm air. Once the pipeline
> samples ERA5 wind into `wind_u` and `wind_v`, this heatmap switches to real
> wind automatically and relabels its legend.

### Basemaps and layers

There are two controls, because these are two different questions. **Basemap**
is a single choice from five, made once and rarely revisited. **Layers** opens a
window for the overlays, which is where the list has grown: the built-in
catalogue, the Earth Engine layers, and whatever assets your society has
registered of its own.

The layer window stays open while you work the map, rather than closing on the
first click. Whether a layer was worth switching on is something you judge by
looking at the map, so the map stays visible and usable underneath it.

Inside it, everything currently drawn is listed first, topmost first:

- **Order** — arrows move a layer up or down the stack. Overlays hide each
  other, and land ownership under a hillshade is a different map from the same
  two the other way up. A layer you switch on goes to the top, which is where
  someone who just asked for it expects to find it.
- **Opacity** — each drawn layer has its own. This multiplies into the global
  tile opacity in Appearance rather than replacing it, so dimming the pile to
  read through it no longer dims the one layer you were trying to read.
- **Search** — by layer name anywhere, or by the start of a group name.

The basemap is a light gray canvas by default, so the observations are the only
saturated thing on screen. Street, terrain and satellite are one click away.
Reference layers stack on top, grouped by subject:

- **Terrain** — hillshade, USGS topo, USGS imagery, OpenTopoMap relief. Gray base
  plus hillshade gives relief without color.
- **Weather** — live US radar, US rainfall over the past 24 hours, global
  satellite rainfall, land surface temperature.
- **Ground** — ESA WorldCover land cover at 10 m, SMAP soil moisture.
- **Vegetation** — MODIS NDVI greenness, and **forest and land cover type**:
  USGS GAP ecological systems at 30 m, grouped into the types a forager
  separates. WorldCover says *tree cover*; this says *which trees*, which for
  most host-specific species is the difference between two different lists of
  what you might find.
- **Soil** — texture class, depth to bedrock, sand content as a drainage proxy,
  and a sand/silt/clay composite. Texture decides how long the ground stays wet
  after rain, which is the half of fruiting weather the rain layers cannot tell
  you. The composite has no scale to read a value off; its point is the
  boundaries, which often run with the ground rather than with anything visible
  on the surface.
- **Fire** — years since fire, burn severity, this year's burn scars, active
  fires, and a computed dNBR severity. Years since fire and burn severity are
  open to everyone; the rest are a membership benefit.
- **Context** — place labels, hiking trails, land ownership.

Every measured layer carries a key, collected into one **Map layers** panel that
appears as you switch layers on. Layers that vary by day carry a date picker
there too. It opens a few days back, because every satellite product has latency
and asking for today returns blank tiles that would read as "no rain" rather than
"not processed yet".

Zoomed in past a layer's native resolution, its tiles are stretched rather than
resolved finer, and the key says so.

> **Caution** Layers are not heatmaps. A layer is somebody else's raster, drawn
> everywhere because they measured it everywhere, showing current or recent
> conditions over historical finds — so it cannot explain those finds. The
> rainfall, temperature, soil-moisture and NDVI **heatmaps** use what the
> pipeline sampled at each observation's own date, which can.

A layer that cannot be reached says so rather than drawing nothing. An empty
ownership layer would otherwise read as "no public land here".

### Other map controls

- **Drop a point** places a marker anywhere. The panel gives its coordinates, the
  heatmap value in the cell beneath it, and the nearest loaded find. Drag to
  move it; press `Esc` to clear it.
- **My location** puts a dot at your position with its accuracy circle.
- **Live clustering** runs k-means in the browser, by features or geography.
- **Save image** flattens the whole map, tiles and all, into a PNG.
- **Map key** collapses when it is in the way, and steps aside when the record
  panel opens.

### The record panel

Click a point to open it. Each row's label explains what the value means when you
hover it, or tap it on a touch screen. Sections fold; the choice is remembered.

Several values are **modelled from terrain rather than measured** — the wetness
index, and the solar and wind exposure indices. The tips say which.

> **Caution** Check the **Accuracy** and **Precision** rows before trusting the
> terrain below them. iNaturalist blurs the location of sensitive taxa and of
> records whose owner asked it to, and everything below was sampled at the
> published point.

### Offline use

A map of where things grow is most often read standing in the place it
describes, which is where there is least likely to be a connection. Three things
can be saved into the browser separately, because they cost very different
amounts:

- **The app** — every page and the code behind it, so Charts and Analysis open
  offline too, not just the page you happened to be on.
- **Observations** — the dataset the map, table and charts all read. This is the
  large one.
- **An area** — imagery for the place on screen, plus however many zoom levels
  closer you ask for, across every layer currently drawn.

Pan to where you are going before saving an area. The count and a rough size are
shown before anything downloads, because the count roughly quadruples per extra
zoom level. A save past twenty thousand tiles is refused outright rather than
trimmed to fit: an area quietly missing its edges is worse than one that was
never saved, because you find out where there is no signal to fix it.

#### Managing saved areas

An area is named when you save it, and the collection lives on the **Offline**
page, reachable from the account menu. Each one can be renamed, re-saved to pick
up anything that failed, deleted, or opened — which puts the map back over it.

Name them. A list of saved places is read weeks later, and by then a set of
coordinates is a puzzle where "north ridge" is an answer.

Deleting an area frees only the tiles no other saved area still needs. Two areas
over the same valley share tiles, and removing one must not punch a hole in the
other.

The **Offline** page also shows what the browser says this site is occupying in
total. That figure is the accurate one. The per-area sizes are estimates, and
are labelled as such: tiles are fetched from hosts that do not all report a
length, and once stored they cannot be measured at all.

Nothing beyond the app shell is saved on its own: pulling a dataset and a few
hundred tiles onto someone's mobile data unasked is not a feature. Saved data
lives in this browser on this device, is not uploaded, does not follow your
account, and goes when you clear the browser's site data. The app is installable
to a home screen.

> **Note** Earth Engine layers can be saved into an area like any other. Their
> tile URLs carry a token that expires within hours, so they are filed under the
> layer rather than under the URL — otherwise every tile you saved would be
> unreachable by the time you were standing in the woods reading it.

> **Caution** Tile services set their own terms on bulk downloading. Save the
> area you are going to, not a region.

### Taxonomy

Every observation carries its full ancestry — **kingdom, phylum, class, order,
family, genus, species** — resolved against iNaturalist when the record is
fetched. All seven are dimensions you can color by, group by, filter to, and
run the analysis over.

This replaces splitting the species name on its spaces, which produced a genus
only when a record happened to be identified to species, and could not reach
family or above at all. A record identified only to *Amanitaceae* used to have a
"species" of "Amanitaceae" and a genus to match. It now has a family, and an
empty genus and species.

> **Note** That is the honest answer, and it means filtering at species level
> quietly drops every record nobody pinned down that far. Filter at the coarsest
> rank that answers your question.

On **Data → Species**, the **Rank** selector decides both what the list shows and
what the filter applies to, so the same picker narrows a view to one kingdom or
to one species. Only ranks the loaded dataset populates are offered: a dataset
exported before this existed carries species and genus and nothing else.
Switching rank clears the selection, because the names do not carry across.

---

## Charts

### Gallery

Nineteen preset charts. Each card can be reordered, hidden, opened full screen,
or saved as a PNG. Shift-click the save button for SVG, which stays sharp at any
size.

Cards are a fixed height so the grid stays regular. A chart taller than its card
is clipped with a fade; full screen shows it uncut.

### Build

Compose your own chart: scatter, bar, line, area, box plot, histogram, heatmap,
radar or donut. Pick fields for each axis, color, shape, size and series, then
choose **Save to Charts** to keep it.

**Sort** orders grouped charts by value or by label. Largest-first answers "which
is biggest"; A–Z answers "where is X". The size cap applies before the sort, so
choosing A–Z never pushes the biggest categories off the chart.

### Style

Five palettes including a color-blind-safe one, plus point size, opacity and
outline, the color ramp and opacity the grid heatmaps use, the cell shape they
bin into, the opacity of reference layers, and per-value overrides. Pin a species
to a color and it holds across the map and every chart.

**Shuffle colors** deals the same palette out differently, for when two species
land on shades you cannot separate. It is deterministic, so a shuffled view looks
the same to whoever opens your shared link. Settings persist per viewer.

---

## Analysis

Statistics over whatever the filters currently select.

### What relates to what

A **Spearman** correlation matrix across every populated field. Rank-based, so a
relationship counts even when it bends — and these bend.

Each cell uses only the rows where *both* fields are present. That matters here:
soil moisture is on about 23% of rows, so dropping rows missing any field would
compute the whole matrix on an unrepresentative remainder.

> **Caution** Two confounds run through every pair. **Season**: high-elevation
> finds happen in summer and low ones in spring and autumn, which is why
> elevation and temperature appear to rise together (ρ +0.43) — hold the month
> still and that flattens to about zero. **Effort**: people record where people
> go.

### Species

**Fingerprints** show how far each species sits from the dataset average on every
field, in standard deviations. Positive means found higher, warmer, wetter or
later than average. Z-scores are used rather than raw means because metres,
degrees and millimetres cannot be compared side by side.

**Found together** ranks species pairs by **lift**: how much more often they
co-occur than their individual frequencies predict, within the same ~5 km cell
and the same month. Lift rather than a raw count, because a count would rank the
two commonest species first whether or not they have anything to do with each
other.

### Fruiting timing

What moves a species' fruiting earlier or later from year to year, and whether
that tracks rainfall, temperature, or both. Timing is the median day of year;
drivers are ranked by partial correlation, so a driver that only looks
significant because it tracks another one is demoted.

The rainfall chart shows one year against the average and against previous years,
over a 30-day trailing window.

> **Note** Only species with at least 20 observations across four or more
> complete years are offered. Below that the year-to-year signal is noise.

### Year over year

Season timing (median day of year) and median elevation per year, alongside the
recording effort that drives both. Median rather than mean: a few winter records
would drag a mean badly.

> **Caution** Read the effort chart first. Recording has grown steeply, so a
> shift in either trend may be a shift in who is looking.

### Data quality

Field coverage overall and by year. This is the tab that tells you how much
weight the rest can carry: a chart drawn from a 23%-covered column looks exactly
as confident as one drawn from a full column. Thin fields are thin because
enrichment has not reached those rows; re-running the pipeline fills them.

---

## Data

### Species

Every species with its record count, grouped at genus, species or subspecies
level. Selecting one narrows every view at once.

### Table

All observations with their enriched columns, sortable and searchable. Only the
visible rows are rendered, so the full set stays responsive.

### Filters

Filters apply everywhere: map, charts, table and analysis.

| Filter | Notes |
| --- | --- |
| Country / state / county | Parsed from the iNaturalist place string |
| Radius | Distance from a chosen point |
| Year / month / week | ISO week, matching the week-of-year chart |
| Date range | From and to |
| Minimum records | Drops taxa below a threshold, by species or genus |

**Minimum records** is counted *after* the other filters, so it means "enough
records in what you are looking at". Narrowing to one county and asking for 25
gives well-sampled species in that county. A species seen twice cannot tell you
where it fruits.

**Saved subsets** store the whole filter state under a name and restore it in one
click. The chip describes itself from whatever is actually set.

---

## Pipeline jobs

Society members can run the enrichment pipeline from the app, without a Python
environment or Earth Engine credentials of their own.

1. Go to **Pipeline jobs** in the account menu.
2. Choose an area — or use the current map view — with an optional date range and
   taxon.
3. Pick which layers to sample.
4. Queue the job.

Progress appears against the job as it runs; you can leave the page. When it
finishes, **Open on map** loads the result as a dataset like any other, so the
map, charts and analysis all read it.

> **Note** Earth Engine bills the project rather than the caller, so every job
> spends from one shared pool. Each member has a monthly unit budget, a daily job
> count, a per-job point ceiling and a concurrency limit, all set by an
> administrator.

---

## Your own Earth Engine layers

An administrator can add map layers computed in Earth Engine, alongside the
built-in ones.

1. Build the layer in the Earth Engine Code Editor.
2. Export it with `Export.image.toAsset()` into your own Earth Engine project.
3. Grant the app's service account read access to the asset.
4. Register it on **Administration → Map layers**: the asset ID, which band to
   draw, a palette, and who can see it.

It then appears in the map's layer window under whatever group you name, with
its own key, and renders through the same path as every built-in layer.

### How the asset and the token fit together

The asset is permanent; the tile URL is not, and the app is built around that
difference.

`Export.image.toAsset()` writes your processed grid into a managed record inside
your own Earth Engine project. It stays there indefinitely and never expires.
But Earth Engine does not serve raw tiles publicly: calling `getMapId()` on the
asset mints a URL carrying a token that expires within hours.

So a function on the server is the bridge. The browser asks it for a tile URL;
it authenticates with the service account, loads the already-computed asset —
skipping every `remap()` and `visualize()` the export already did — calls
`getMapId()`, and hands the fresh URL back for Leaflet to draw. Nothing is
pre-sliced into thousands of stored `z/x/y.png` files: one asset in Earth
Engine, and a small function minting tokens on demand.

The app re-mints roughly hourly, because an expired template does not error — it
serves blank tiles, and blank ground reads as ground with nothing on it rather
than as a layer that failed.

> **Note** This is also why saving an Earth Engine layer for offline use files
> its tiles under the layer rather than under the URL. Keyed by URL, everything
> saved would be unreachable the moment the token rotated.

### A tree cover layer, end to end

Classify or threshold in the Code Editor, export once, register the asset. The
export is what makes it cheap to draw: the work happens once, not per tile.

```js
// Ten cover classes over an area of interest, exported once as an asset.
var aoi = ee.Geometry.Rectangle([-105.4, 39.5, -104.6, 40.2]);

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)
  .filterDate('2025-06-01', '2025-09-15')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('ndvi');

// Ten classes, 1..10. Zero is left free to mean nodata, which is what the
// layer's "Hide values below" setting masks out.
var classes = ndvi.multiply(10).ceil().clamp(1, 10).toByte().rename('classification');

Export.image.toAsset({
  image: classes.clip(aoi),
  description: 'cover_classes_2025',
  assetId: 'projects/your-project/assets/cover-classes-2025',
  region: aoi,
  scale: 20,
  maxPixels: 1e10
});
```

When the export finishes, grant the app's service account read access to the
asset, then register it on **Administration → Map layers**. Choose the
**Classified cover** preset: it fills the band, the 1–10 range, ten distinct
colours and the zero mask, which are the settings that are easy to get wrong and
produce a layer that renders badly rather than one that errors.

The presets cover the shapes these rasters usually take — percent cover, a
classified grid, canopy height in metres, a signed index, a probability. None of
them carries an asset ID: they describe how to paint a raster, and which asset
you have is yours.

Each layer carries its own access level — **everyone**, **members** or
**administrators** — so a finished layer can be public while a draft stays
internal. The picker lists a layer the viewer cannot render, marked, rather than
hiding it; switching it on explains why.

> **Note** What is stored is an asset ID, never a script. An asset ID names
> something already computed under your own project, where a script would be
> arbitrary compute on the society's Earth Engine budget. Exporting first also
> makes the layer cheap to draw, because the work happened once.

> **Caution** If the asset has more than one band, name the band. Earth Engine
> refuses a palette on a multi-band image, and the layer will not render. Set
> **Hide values below** as well when the asset's no-data is zero, or the whole
> world paints as the bottom of the ramp.

---

## Sharing and saving

**Share** builds a link that reproduces the current view. Map position, filters,
coloring, heatmap and date window all travel with it. From there you can produce
a QR code (generated locally, so the link never reaches a third party), or share
to X, Bluesky, Facebook, Reddit, email, SMS, or an iframe embed that drops the
site header.

**Save image** exports charts as PNG or SVG, and the map as a PNG with its
basemap composited in.

**Accounts** are optional. Signed out, everything persists in this browser.
Signed in, settings and saved charts follow you across devices, and the first
sign-in on a device merges rather than overwrites, so work done before making an
account is not lost.

---

## Coverage

Which environmental raster layers are cached, for what dates, and over what area.
This is the pipeline's side of the same question the Analysis page's data-quality
tab asks from the observations' side.

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
