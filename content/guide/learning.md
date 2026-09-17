# Learning

This page is for the step after you read the map. It covers the three subjects
that this app sits on: GIS, Earth Engine and machine learning.

Each section gives you the smallest set of ideas that makes the next document
readable. Each one then links to the documents themselves.

> **Note** These links go to other organisations. They are not part of
> Nexstrata, and FRMS does not control them.

## GIS basics

GIS is the handling of data that has a position on the Earth.

### Vector and raster

There are two kinds of spatial data, and almost everything is one of them.

- **Vector** data is points, lines and polygons. Each shape carries attributes.
  An observation is a point. A county boundary is a polygon. The observations in
  this app are vector data.
- **Raster** data is a grid of cells. Each cell holds one number for each band.
  Elevation, rainfall and satellite imagery are rasters. Each map layer in this
  app is a raster.

The pipeline does one job: it reads the raster cell below each vector point.

### Coordinates

A coordinate needs a reference system before it means a place. This app uses
**WGS 84** (EPSG:4326), which is latitude and longitude in degrees. GPS units
report the same system, so the numbers agree.

The map draws in **Web Mercator** (EPSG:3857), like almost every web map. Web
Mercator makes areas near the poles look much larger than they are. Greenland is
the usual example.

### Resolution and scale

The **resolution** of a raster is the ground size of one cell. Sentinel-2 is
10 m. SRTM is 30 m. CHIRPS rainfall is approximately 5 km.

A value that you read at a point describes the whole cell. At 5 km, "the
rainfall at this log" is the rainfall across an area larger than most towns.

Zoom does not add resolution. When you zoom in past the native resolution of a
layer, the app stretches the tiles. Read [Layer keys and
dates](/guide/map#layer-keys-and-dates).

### GIS: where to read more

- [A Gentle Introduction to
  GIS](https://docs.qgis.org/latest/en/docs/gentle_gis_introduction/) — free,
  short, and does not assume software.
- [QGIS](https://qgis.org/) — free desktop GIS. Open an export from this app in
  it. Read [Export](/guide/data#export).
- [The QGIS training
  manual](https://docs.qgis.org/latest/en/docs/training_manual/) — a longer,
  practical course.
- [EPSG.io](https://epsg.io/) — look up a coordinate reference system by code.

## Earth Engine

Google Earth Engine is a catalogue of public raster data, with a compute engine
next to it. You do not download the rasters. You send the calculation to them.

### The four objects

Almost every script uses four types:

- **`ee.Image`** — one raster, with one or more bands.
- **`ee.ImageCollection`** — many images, usually a time series. Sentinel-2 is
  an image collection.
- **`ee.Feature`** and **`ee.FeatureCollection`** — vector data. Your points are
  a feature collection.
- **`ee.Reducer`** — how to combine many values into fewer. A mean, a median, a
  maximum, a histogram.

A typical script filters a collection by date and by area, reduces it to one
image, and then samples that image at a set of points. That is what the
enrichment pipeline of this app does.

### The one idea that confuses everyone

Your script runs in your browser. The calculation runs on Google's servers.

An `ee.Number` is not a JavaScript number. It is a description of a calculation
that has not happened yet. You cannot print it with `console.log` and see a
value. You must call `getInfo()`, or `print()` in the Code Editor, and wait.

This is also why a loop over an `ee.List` in JavaScript is slow or impossible.
Use `map()` on the Earth Engine object instead.

### Cost

Earth Engine charges for compute. A commercial project pays. A noncommercial
project has a quota.

Two habits keep the cost low:

- Filter first. Filter by date and by area before you do anything else.
- Export once. When a result does not change, write it to an asset and read the
  asset. Read [Your own layers](/guide/layers).

### Earth Engine: where to read more

- [The Earth Engine
  guides](https://developers.google.com/earth-engine/guides) — the official
  documentation. Start with "Get Started".
- [The data
  catalogue](https://developers.google.com/earth-engine/datasets) — each dataset
  with its bands, its resolution and its terms.
- [The Code Editor](https://code.earthengine.google.com/) — write and run
  scripts in the browser.
- [Google Earth Engine 101: An Introduction for Complete
  Beginners](https://www.youtube.com/watch?v=oAElakLgCdA) — video. Start here if
  it is easier to watch somebody work than to read a page of documentation.
- [*Cloud-Based Remote Sensing with Google Earth
  Engine*](https://www.eefabook.org/) — a free book of tutorials, from first
  script to applied work.
- [geemap](https://geemap.org/) — Earth Engine from Python, with maps in a
  notebook.
- [The Awesome GEE Community
  Catalog](https://gee-community-catalog.org/) — datasets that the official
  catalogue does not hold.

## Machine learning basics

Machine learning fits a function to examples. You give it inputs and answers. It
produces a rule that maps one to the other.

### The words

- A **feature** is one input column. Elevation, slope and NDVI are features.
- A **label** is the answer that you want. "This species fruits here" is a
  label.
- **Supervised** learning has labels. **Unsupervised** learning does not. The
  live clustering on the map is unsupervised: it groups the points without being
  told what the groups are.
- A **model** is the fitted rule.

### The three mistakes

**Overfitting.** A model can memorise the examples instead of learning the
pattern. It then performs well on the data that it saw and badly on new data.

Always hold data back. Fit the model on one part. Test it on the other part. A
score on the data that the model was fitted to is not a result.

**Leakage.** Nearby points are not independent. Two finds 50 m apart share their
elevation, their slope and their weather. A random split therefore puts almost
the same record in both the training set and the test set, and the score comes
out too high.

Split by block or by region instead. This is spatial cross-validation.

**Bias in the examples.** A model learns the data that you gave it, including
its collection bias. A model fitted on iNaturalist records learns where people
walk as well as where mushrooms fruit.

### Machine learning: where to read more

- [scikit-learn: an introduction to machine
  learning](https://scikit-learn.org/stable/tutorial/basic/tutorial.html) — the
  standard Python library, with a short tutorial.
- [Google's Machine Learning Crash
  Course](https://developers.google.com/machine-learning/crash-course) — free,
  and it covers the three mistakes above properly.
- [*An Introduction to Statistical
  Learning*](https://www.statlearning.com/) — a free textbook. It is the one to
  read when the crash course leaves you with questions.

## Species distribution models

A species distribution model (SDM) fits the environment where a species was
found, and then maps where similar environments are.

This is the natural next step for the data in this app. The observations carry
the coordinates. The enrichment carries the environment at those coordinates.

### Presence-only data

An SDM usually has a hard problem: the data records where a species **was**
found and never where it **was not**.

An absence of records means one of two things. The species is not there, or
nobody looked. An iNaturalist dataset has a great deal of the second.

A presence-only method therefore compares the presences against **background**
points. Background points describe the range of environments that are available,
and not the places where the species is absent.

### MaxEnt

MaxEnt is the most used presence-only method. It finds the distribution that
matches the environment of the presence points, and is otherwise as close to the
background as possible. "Maximum entropy" is that last clause: assume nothing
that the data does not say.

In practice MaxEnt is a form of penalised regression. The penalty, called
regularisation, is what stops it from fitting noise.

### How to begin with the data in this app

1. **Choose one species.** Start with a species that has several hundred
   records. Fewer than approximately 30 presences is not enough for a model that
   you can test.
2. **Run an enrichment job** over those records, with the terrain, soil and
   vegetation layers switched on. Read [Pipeline jobs](/guide/jobs).
3. **Export the result** as CSV or GeoJSON. Read [Export](/guide/data#export).
4. **Remove the obscured records.** iNaturalist obscures the location of
   sensitive taxa. The environment at an obscured point describes a place where
   the observation was not. The export carries the accuracy and the precision
   columns, so you can filter on them.
5. **Thin the records.** Keep one record for each cell of approximately 1 km.
   This reduces the effect of one well-visited trailhead.
6. **Choose the background.** Use the other records in the dataset as the
   background, and not a random sample of the map. Those records are the places
   where people looked and found something else. This is the target-group
   background method, and it is the most effective correction for the bias in
   opportunistic data.
7. **Fit and test the model.** Use spatial blocks for the test split, and not a
   random split.
8. **Read the response curves before the map.** A response curve shows what the
   model learned about one variable. A curve that rises with the distance from a
   road tells you that the model learned the effort and not the species.

### Which software

| Tool | Where it runs | Notes |
| --- | --- | --- |
| [`ee.Classifier.amnhMaxent`](https://developers.google.com/earth-engine/apidocs/ee-classifier-amnhmaxent) | Earth Engine | MaxEnt inside Earth Engine. The predictors are already there. |
| [elapid](https://earth-chris.github.io/elapid/) | Python | MaxEnt and related models, with a scikit-learn interface. |
| [SDMtune](https://consbiol-unibern.github.io/SDMtune/) | R | Fits, tunes and tests. Good documentation on the tests. |
| [Wallace](https://wallaceecomod.github.io/) | R, in a browser | A guided interface. It writes the code for what you did. |

### MaxEnt: where to read more

- Elith et al., [A statistical explanation of MaxEnt for
  ecologists](https://doi.org/10.1111/j.1472-4642.2010.00725.x) — read this
  first. It is the clearest description of what MaxEnt actually computes.
- Merow et al., [A practical guide to MaxEnt
  ](https://doi.org/10.1111/j.1600-0587.2013.07872.x) — the settings, and what
  each one does to the result.
- Phillips et al., [Sample selection bias and presence-only distribution
  models](https://doi.org/10.1890/07-2153.1) — the target-group background
  method in step 6 above.
- Valavi et al., [blockCV](https://doi.org/10.1111/2041-210X.13107) — spatial
  cross-validation, and why a random split gives you a score that is too high.

## Soils

Soil decides how long the ground stays wet after rain. That is the half of
fruiting weather that a rainfall layer cannot tell you.

The soil layers in this app draw texture class, depth to bedrock and sand
content. Read [What the layer groups
contain](/guide/map#what-the-layer-groups-contain).

The formal classification of soils in the United States is **USDA soil
taxonomy**. It is a hierarchy of six levels: order, suborder, great group,
subgroup, family and series. The twelve orders at the top are the level that is
useful on a map.

- [USDA soil taxonomy on
  Wikipedia](https://en.wikipedia.org/wiki/USDA_soil_taxonomy) — the twelve
  orders, with what defines each one.
- [Keys to Soil
  Taxonomy](https://www.nrcs.usda.gov/resources/guides-and-instructions/keys-to-soil-taxonomy)
  — the official reference from the USDA NRCS.
- [Web Soil Survey](https://websoilsurvey.nrcs.usda.gov/) — look up the mapped
  soil at one place in the United States.
- [Soil Orders and the Hierarchy of Soil
  Taxonomy](https://www.youtube.com/watch?v=h0NmxOERbhs) — video. Six levels and
  twelve orders is a lot of vocabulary to meet on a page.
- [Soil Classification](https://youtu.be/Kk7yarqfDoU) — video. How soils are
  grouped and why the class tells you how the ground behaves.

## Mycology

- [The Front Range Mycological Society](https://frontrangemycosociety.org/) —
  forays, identification help and talks.
- [iNaturalist](https://www.inaturalist.org/) — where the observations in this
  app come from. Add your own.
- [MycoPortal](https://www.mycoportal.org/) — herbarium records, which are
  vouchered where an iNaturalist record usually is not.
- [Index Fungorum](https://www.indexfungorum.org/) — the current accepted name
  for a fungus.
