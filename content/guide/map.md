# The map

The map draws each observation as one point. It can draw a grid of summaries
below the points, and reference layers below the grid.

![The map draws four things, in this order.](figure:map-stack)

## Points

Open the **Points** control.

### Color and size

- **Color by** sets what the color of a point means. You can select a category
  or a measurement. The categories include each taxonomic rank from kingdom to
  species, and also the common name, the identification rank, the cluster, the
  land cover, the year, the month and the enrichment level.
- **Size by** scales each point by a measurement. This lets you read two
  values at the same time.

A category gets a stable color. One species keeps the same color on the map, in
each chart and in the key. A measurement gets a color gradient instead. A
gradient reads as *more*; a set of hues does not.

### The point gradient

A measurement needs a scale of colors. Set that scale in **Style → Point
gradient**.

![A gradient is a list of color stops. The app spaces the stops evenly.](figure:ramp-stops)

- **Match the layer** is the default. Read the next section.
- The preset gradients are Blue, Green, Warm, Purple, Viridis-ish and Greyscale.
- **Custom** opens the gradient editor.

The editor holds between 2 and 8 stops. Use the **+** button between two
swatches to add a stop there. The new stop takes the color that the gradient
already had at that point, so the gradient does not change until you move it.
Use the **×** button on a swatch to remove that stop. Use **Reverse** to turn
the gradient end to end.

Two stops give you *low to high*. More stops give you a middle that you can
read. A diverging scale needs a neutral middle, so it needs three stops.

> **Note** More than eight stops stops reading as an order. It starts to read
> as a set of categories, and the reader must then decode it.

### Point colors that match the layer

Eight fields have a map layer that draws the same thing. Elevation, slope,
aspect, NDVI, NDMI, soil moisture, temperature and rainfall are all on the map
as layers and also on each record as a value.

When you color the points by one of these fields, the points take the colors of
that layer. A dot then reads against the ground below it. A green dot on the
NDVI layer sits on green ground when the two agree, and stands out when they do
not.

Some fields share the units of their layer. Elevation in metres is elevation in
metres. For these fields the points also borrow the layer's value range, so the
same color means the same number in both. The other fields are normalised, so
the points borrow only the colors.

To turn this off, select any other gradient in **Style → Point gradient**.

> **Caution** A layer shows current or recent conditions. A point shows the
> conditions on the date of the find. The colors match, but the dates do not.
> Read the section on [layers and heatmaps](#layers-are-not-heatmaps).

### The two keys

The map shows two keys, and they answer two different questions.

![The Observations key explains the dots. The Map layers key explains the tiles.](figure:two-keys)

- **Observations** explains the points. Below the title it names the field you
  colored by.
- **Map layers** explains the tiles below the points. It has one section for
  each layer that is switched on.

Click **Key** to fold both keys away. The app remembers your choice. The keys
also move aside when the record panel opens.

## Heatmaps

Open the **Heatmap** control. A heatmap puts the observations into a grid. It
then shades each cell by one summary statistic.

48,000 points on top of each other show you only where the data is dense. A grid
shows you what is in an area.

| Heatmap | What each cell shows | What to watch for |
| --- | --- | --- |
| Observation density | How many records are in the cell | Effort. Cities and trailheads glow. |
| Species richness | How many different species are in the cell | Effort again. More visits find more species. |
| Seasonal activity | The share of *this cell's own* finds inside the date window | Effort-neutral. Needs 3 or more records in the cell. |
| In-season hotspots | The same share, weighted by how well the cell is sampled | A record of past finds. Not a forecast. |
| Most common species | The species with the most records in the cell | The app breaks a tie arbitrarily. |
| Land cover | The most common land-cover class across the cell's finds | Only the winner shows. A 50/50 cell looks pure. |
| Wind and aspect vectors | Arrows for the wind, or for the direction slopes face | The key says which source it used. |
| Environmental fields | The cell mean of one enriched value | Sampled at the finds. An empty cell means that nobody looked. |

### The seasonal heatmaps

**Seasonal activity** and **In-season hotspots** need a date and a window. The
window is between ±3 and ±60 days. Both controls appear when you select either
heatmap.

These two are the most reliable. Each cell is divided by its own total, and that
cancels most of the effort bias. The *shape* of a cell's season does not depend
on how many people visited it. It depends only on when they found things.

### The field heatmaps

A field heatmap gives the cell mean of one enriched value. The fields are 7-day
rainfall, temperature, soil moisture, the wetness index, slope, aspect, sun
exposure, wind exposure, NDVI, NDMI and elevation.

> **Caution** A field heatmap is not a coverage map. A cell has a value only
> where a person recorded a find. An empty cell means that nobody went there. It
> does not mean that the ground is dry or flat.

Aspect is circular, so it gets a compass key. It does not get a low-to-high bar,
because 359° and 1° are neighbours.

### Cell size and shape

Cell size runs from about 100 m to about 28 km. A smaller cell is more precise
and more noisy. Below approximately one kilometre the grid stops to summarise.
It then draws about one cell for each observation.

![A hexagon has six neighbours at one distance. A square has four close neighbours and four corner neighbours that are 1.41 times further.](figure:hex-vs-square)

Cells are **hexagons** by default. Each neighbour of a hexagon is the same
distance away, and shares a full edge. A hex grid therefore reads as a surface.
A group of finds is also not divided differently when it moves against the axes.

Squares are available in **Style**. A square cell has the same area as the
hexagon it replaces, so a change of shape does not change the resolution.

**Heatmap opacity** and **Map layer opacity** are two sliders in **Style**. They
control two different stacks. A faint hillshade below a solid heatmap is the
usual case.

The heatmap gradient is set in **Style → Heatmap**. It uses the same editor as
the point gradient, and holds the same 2 to 8 stops.

## Wind and aspect vectors

Each observation carries a terrain **aspect** and a wind-exposure index. Aspect
is the compass direction that a slope faces. No observation carries a measured
wind.

The vector heatmap therefore draws the mean terrain aspect. The length of an
arrow shows how *consistently* the cell faces one way. The color shows the wind
exposure.

> **Note** A short arrow means mixed terrain. It does not mean calm air. The
> heatmap changes to real wind, and relabels its key, after the pipeline samples
> ERA5 wind into `wind_u` and `wind_v`.

## Basemaps and layers

There are two controls, because these are two different questions.

- **Basemap** is one choice from five. You make it once and rarely change it.
- **Layers** opens a window for the overlays.

The layer window stays open while you work on the map. It does not close on the
first click. You judge a layer by what the map looks like, so the map stays
visible below the window.

### The layer window

The window lists each layer that is drawn, topmost first. Each one has:

- **Order** — the arrows move a layer up or down the stack. **⤒** and **⤓** send
  it straight to the top or the bottom. Overlays hide each other. Land ownership
  below a hillshade is a different map from the same two in the opposite order.
  A layer that you switch on goes to the top.
- **Solo** (**S**) — draws that layer on its own. The other layers stay switched
  on, and the list shows them dimmed. Read [Solo](#solo-one-layer).
- **Opacity** — each drawn layer has its own. This multiplies into the global
  tile opacity in **Style**. It does not replace it. You can therefore dim the
  whole stack without dimming the one layer you want to read.
- **Blend** — how the layer combines with the layers below it. Read [Blend
  modes](#blend-modes).

The window also has a **Search** box. Search by a layer name anywhere, or by the
start of a group name.

### Solo one layer

Click **S** beside a drawn layer. The map then draws only that layer.

The other layers stay switched on. The list shows them dimmed, and they come
back exactly as they were. This lets you answer "what is this one layer adding"
without a loss of the stack that you built.

Click **S** again, or click **Un-solo** at the top of the window, to draw
everything again. When you switch any layer on or off, the solo also ends.

### Blend modes

Opacity is not the correct control for two rasters. Two layers at 50% is two
layers that are half washed out. The thing that you wanted — the shape of the
hillshade over the color of the land cover — is what 50% of each destroys.

A blend mode keeps both layers at full strength. It combines them by value.

| Mode | What it does | When to use it |
| --- | --- | --- |
| Normal | Draws over. Nothing is combined. | The default. |
| Multiply | Keeps what is dark in both layers. | Relief below color. |
| Screen | Keeps what is light in both layers. | Burn scars or cloud over terrain. |
| Overlay | Multiplies the dark parts, screens the light parts. | More contrast. |
| Darken | The darker of the two, for each color channel. | A hard version of multiply. |
| Lighten | The lighter of the two, for each color channel. | A hard version of screen. |
| Difference | What the two layers disagree about. | The same product on two dates. |
| Luminosity | The brightness of this layer, the color of the one below. | Shape from one layer, class from another. |

The browser does this work for each frame. No tile is fetched again, and
nothing is computed again. A blend mode therefore costs nothing to try.

Each layer starts at **Default**. Default follows the **Blend when stacked**
setting in **Style**, and applies only when two or more layers are drawn. When
you select a mode for one layer, that layer keeps it.

> **Caution** A blend mode combines a layer with everything below it, and that
> includes the basemap. The same two layers over satellite imagery and over the
> gray canvas do not look the same. Neither result is the layer on its own. Use
> **Solo** for that.

### What the layer groups contain

The basemap is a light gray canvas by default, so the observations are the only
saturated thing on the screen. Street, terrain and satellite are one click away.

- **Terrain** — hillshade, USGS topo, USGS imagery, OpenTopoMap relief. The gray
  base plus the hillshade gives you relief without color.
- **Weather** — live US radar, US rainfall over the past 24 hours, global
  satellite rainfall, land surface temperature.
- **Ground** — ESA WorldCover land cover at 10 m, SMAP soil moisture.
- **Vegetation** — MODIS NDVI greenness, forest and land cover type, and
  Sentinel-2 greenness and canopy moisture (**NDVI** and **NDMI**). The forest
  type layer is USGS GAP ecological systems at 30 m, grouped into the types a
  forager separates. WorldCover says *tree cover*; this says *which trees*. The
  Sentinel-2 layers come in two forms. The free form is a rolling composite of
  the last few weeks. The member form is a median over one season and one year,
  for a comparison of the same phenological moment across years.
- **Forest structure** — USFS TreeMap canopy density, stand height and stand-size
  class at 30 m. Together they are a proxy for the maturity of a stand. Many
  ectomycorrhizal fungi need an old host. US forests only, modelled from 2016.
- **Terrain analysis** — slope, aspect, topographic wetness (TWI), wind exposure
  and solar exposure. All five are computed from the SRTM elevation model.
  Wetness finds the ground that collects water. Exposure and solar find the
  ground that dries quickly. Aspect and slope are below both. Open to everyone.
- **Soil** — texture class, depth to bedrock, sand content as a proxy for
  drainage, a sand/silt/clay composite, and the two soil taxonomy layers below.
  Texture decides how long the ground stays wet after rain. That is the half of
  fruiting weather that the rain layers cannot tell you. The composite has no
  scale to read a value from. Its purpose is the boundaries, which often follow
  the ground and not the surface.
- **Fire and disturbance** — years since fire, burn severity, this year's burn
  scars, active fires, a computed dNBR severity, forest loss and cutting (Hansen
  Global Forest Change), and a Sentinel-2 canopy-moisture crash. The last one
  flags beetle-kill and drought die-off. Years since fire, burn severity and
  forest loss are open to everyone. The computed layers are a membership benefit.
- **Context** — place labels, hiking trails, land ownership.

### Soil taxonomy

Two layers draw the USDA soil classification, at 250 m, worldwide.

**Soil taxonomy (USDA orders)** paints the soil order at each pixel. There are
twelve orders, and they are the top of the hierarchy. Orders separate soils by
how they formed, so the boundaries often follow the geology and the climate, and
not anything that you can see on the surface.

**Soil taxonomy: matsutake ground** paints only the eighteen great groups that
FRMS members have flagged as matsutake ground on the Front Range. These are the
cool, acid, sandy and volcanic soils, across five orders. All other ground stays
blank.

> **Caution** The matsutake layer is a soil filter and not a prediction. It says
> that the ground is the correct kind. It does not say that anything fruits
> there, and it knows nothing about the host trees that decide whether anything
> can.

#### The class browser

The source data is at the **great group** level, which is the third of the six
levels of the hierarchy. It has approximately four hundred classes. Four hundred
swatches is not a key. It is a lookup table that you must read line by line.

So the key for these layers is a browser. Switch either layer on, and the **Map
layers** panel gives you:

- **The twelve orders**, as the key. Click one to show only its own great
  groups. This answers "what are all the podzols here", which a search box
  cannot: no podzol is named "podzol".
- **A search box**. It matches the name of a great group and also the name of
  its order.
- **The list**. Each row shows the color of its order, its name, and a 🍄 if it
  is one of the eighteen flagged great groups.

Hover on a row to read a one-line description. Click it to open the whole
description, and a link to the Wikipedia article for that order.

#### How to read a great group name

You do not need the list to read a soil name. The names are built from a fixed
vocabulary, and they are built to be read:

| Part | Where | What it says |
| --- | --- | --- |
| `Dystro` | front | Low base saturation: acid, and short of calcium |
| `cry` | middle | Cold: a mean soil temperature below about 8 °C |
| `epts` | end | Inceptisols, the order |

**Dystrocryepts** is therefore a cold, acid soil with weak horizons. The app
decodes the name for you in the class browser, part by part.

The ending is always the order. Read [Soils](/guide/learning#soils) for the
twelve orders and where to read more about them.

### Layer keys and dates

Each measured layer carries a key. All the keys collect into one **Map layers**
panel, which appears when you switch a layer on.

A layer that changes by day carries a date picker in that panel. The picker
opens a few days in the past. Each satellite product has a latency, and a
request for today returns blank tiles. Blank tiles would read as "no rain" and
not as "not processed yet".

When you zoom in past the native resolution of a layer, the app stretches its
tiles. It does not resolve them more finely. The key says so.

A layer that the app cannot reach says so. It does not draw nothing. An empty
ownership layer would otherwise read as "no public land here".

### Layers are not heatmaps

> **Caution** A layer is a raster from somebody else. It is drawn everywhere,
> because they measured it everywhere. It shows current or recent conditions
> over historical finds, so it cannot explain those finds. The rainfall,
> temperature, soil-moisture and NDVI **heatmaps** use what the pipeline sampled
> at the date of each observation. Those can.

## Other map controls

- **Drop a point** puts a marker anywhere. The panel gives its coordinates, the
  heatmap value in the cell below it, and the nearest loaded find. Drag the
  marker to move it. Press `Esc` to remove it.
- **My location** puts a dot at your position, with its accuracy circle.
- **Live clustering** runs k-means in the browser, by features or by geography.
- **Save image** flattens the whole map, with its tiles, into a PNG.
- **Map key** folds away when it is in your way.

## The record panel

Click a point to open its record. Hover on the label of a row to read what the
value means. On a touch screen, tap the label.

Sections fold. The app remembers which ones you folded.

Three values are **modelled from terrain and not measured**: the wetness index,
the solar exposure index and the wind exposure index. The tips say which.

> **Caution** Read the **Accuracy** and **Precision** rows before you trust the
> terrain below them. iNaturalist obscures the location of sensitive taxa, and of
> records whose owner asked for this. Everything below was sampled at the
> published point.

## Offline use

You most often read a map of where things grow in the place that it describes.
That is where a connection is least likely.

Three things save into the browser separately, because they cost very different
amounts:

- **The app** — each page and the code behind it. Charts and Analysis then open
  offline too, and not only the page that you were on.
- **Observations** — the dataset that the map, the table and the charts read.
  This is the large one.
- **An area** — the imagery for the place on the screen, for each layer that is
  drawn. You choose how many zoom levels closer to save.

Move the map to your destination before you save an area. The app shows the tile
count and an approximate size before it downloads anything. The count increases
approximately four times for each additional zoom level.

The app refuses a save of more than twenty thousand tiles. It does not trim the
save to fit. An area that quietly has no edges is worse than an area that was
never saved, because you find out where there is no signal to correct it.

### How to manage saved areas

Give an area a name when you save it. The collection is on the **Offline** page,
which you open from the account menu. You can rename an area, save it again to
collect anything that failed, delete it, or open it. **Open** moves the map back
over it.

Name your areas. You read a list of saved places weeks later. By then a set of
coordinates is a puzzle, and "north ridge" is an answer.

When you delete an area, the app frees only the tiles that no other saved area
needs. Two areas over the same valley share tiles. The app must not make a hole
in the second area when you remove the first.

The **Offline** page also shows the total size that the browser reports for this
site. That figure is the accurate one. The size of each area is an estimate, and
the page says so. Tiles come from hosts that do not all report a length, and the
app cannot measure a tile after it is stored.

### What the app does not save

The app saves nothing more than its own shell without a request. A dataset and
several hundred tiles on somebody's mobile data is not a feature.

Saved data stays in this browser on this device. The app does not upload it. It
does not follow your account. It goes when you clear the site data of the
browser. You can install the app to a home screen.

> **Note** You can save an Earth Engine layer into an area like any other layer.
> The tile URL of an Earth Engine layer carries a token that expires in hours.
> The app therefore files these tiles under the layer, and not under the URL.
> Under the URL, each tile that you saved would be unreachable when you read it
> in the woods.

> **Caution** Each tile service sets its own terms for bulk downloads. Save the
> area that you go to. Do not save a region.

## Taxonomy

Each observation carries its full ancestry: **kingdom, phylum, class, order,
family, genus, species**. The app resolves the ancestry against iNaturalist when
it fetches the record.

All seven ranks are dimensions. You can color by them, group by them, filter to
them, and run the analysis over them.

This replaces a split of the species name on its spaces. That method produced a
genus only when a record was identified to species. It could not reach family or
above at all. A record identified only to *Amanitaceae* had a "species" of
"Amanitaceae" and a genus to match. It now has a family, an empty genus and an
empty species.

> **Note** That is the honest answer. It also means that a filter at species
> level quietly removes each record that nobody identified that far. Filter at
> the coarsest rank that answers your question.

On **Data → Species**, the **Rank** selector sets what the list shows and what
the filter applies to. The same control therefore narrows a view to one kingdom
or to one species.

The app offers only the ranks that the loaded dataset fills. A dataset exported
before this existed carries species and genus and nothing else. A change of rank
clears the selection, because the names do not carry across.
