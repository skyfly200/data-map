# Guide

Nexstrata is a workbench for environmental data. It runs on Google Earth Engine.
It does three things:

- It samples raster layers at point coordinates.
- It draws computed layers as map tiles.
- It sends the result to the map, the charts and the statistics.

The first dataset is mushroom observations from iNaturalist. Each observation
carries a description of the ground below it. The map opens on this dataset. But
the dataset is an example of the shape, not a limit. Any record with a coordinate
and a date can go through the same path.

![Source data goes through a job into a dataset. Every view reads the dataset.](figure:data-flow)

> **Note** iNaturalist records are opportunistic observations. They are not
> surveys. An area with many records can contain many mushrooms. It can also be
> near a trailhead. Each view tells you if it corrects for this.

## Start here

1. Open the **Map**. Each observation is one point. The colors show
   environmental clusters until you change them.
2. Open **Points** to set what the color and the size of each point mean.
3. Open **Heatmap** to draw a grid of summaries below the points.
4. Open **Layers** to add overlays. Open **Basemap** to change the map below
   them.
5. Go to **Data → Filters** to narrow the data. A filter applies to all views at
   the same time.
6. Click a point to open its record in the side panel.

## The two things this platform is for

You can read the data that is already here. You can also add your own:

- **[Run an enrichment job](/jobs)** samples all the environmental layers at
  your own points, in your own area, for your own dates. This is a membership
  benefit. See [Pipeline jobs](/guide/jobs).
- **[Publish your own layer](/guide/layers)** computes something in Earth
  Engine and draws it on the map with the built-in layers.

## The pages of this guide

| Page | What it covers |
| --- | --- |
| [The map](/guide/map) | Points, heatmaps, layers, the record panel and offline use |
| [Data and export](/guide/data) | The species list, the table, the filters, and how to export |
| [Charts](/guide/charts) | The chart gallery, the chart builder and the style controls |
| [Analysis](/guide/analysis) | Correlations, species statistics and fruiting timing |
| [Pipeline jobs](/guide/jobs) | How to run a job, save the result and chain jobs |
| [Your own layers](/guide/layers) | How to compute a layer in Earth Engine and register it |
| [Learning](/guide/learning) | Where to start with GIS, Earth Engine and machine learning |
| [Data sources](/guide/sources) | Which product each column comes from |
| [Option reference](/guide/reference) | Every control, in one list |

## How to get help on one control

Each control has a small **?** next to it. The **?** links to the entry for that
control in the [Option reference](/guide/reference). Hover on the **?** to read
one sentence. Click it to read the full entry.
