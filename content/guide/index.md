# Guide

Nexstrata maps where species occur and models where they might. It uses Google Earth Engine to:

- **Sample** environmental data at each observation's location and date
- **Map** computed layers as interactive tiles
- **Analyze** patterns through charts and statistics

It works with any presence-point data: iNaturalist, GBIF, or your own uploads.

![Source data goes through a job into a dataset. Every view reads the dataset.](figure:data-flow)

> **Note** Observation records are opportunistic, not systematic surveys. Record density reflects observer effort as much as actual occurrence. Each view states whether it corrects for this bias.

## Quick start

1. Open the **Map**. Points show observations, colored by environmental similarity.
2. Use **Points** to change what colors and sizes mean.
3. Try **Heatmap** to see density patterns in a grid.
4. Add **Layers** like terrain or weather. Change the **Basemap** underneath.
5. Filter data in **Data → Filters**—applies everywhere at once.
6. Click any point to see its full details.

## What you can do

Explore existing data or add your own:

- **[Run an enrichment job](/jobs)** adds environmental data to your own points (membership feature). See [Pipeline jobs](/guide/jobs).
- **[Publish your own layer](/guide/layers)** computes custom data in Earth Engine and displays it on the map.

## Guide sections

| Page | What it covers |
| --- | --- |
| [The map](/guide/map) | Points, heatmaps, layers, and offline use |
| [Data and export](/guide/data) | Species list, table view, filters, and exporting |
| [Charts](/guide/charts) | Chart gallery, builder, and styling |
| [Analysis](/guide/analysis) | Correlations, species stats, and timing |
| [Pipeline jobs](/guide/jobs) | Running jobs, saving results, and chaining |
| [Your own layers](/guide/layers) | Computing layers in Earth Engine |
| [Learning](/guide/learning) | GIS, Earth Engine, and machine learning basics |
| [Data sources](/guide/sources) | Where each data column comes from |
| [Option reference](/guide/reference) | All controls explained |

## Help on any control

Every control has a small **?** button. Hover for a quick tip, click for the full explanation in the [Option reference](/guide/reference).
