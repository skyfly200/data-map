// What every option in the app does, in one place.
//
// A tooltip has room for a sentence; some of these controls need a paragraph
// and a warning. So each option is written once here, and read twice: the
// control shows `summary` on hover next to a ? that links to /guide#<id>, and
// the guide renders the whole entry under that anchor. A control cannot promise
// something the guide contradicts, because there is only one text.
//
// `summary`  one sentence, shown on hover. No markup.
// `detail`   paragraphs for the guide.
// `caveat`   what the number does NOT mean — the part that stops a reader
//            over-trusting it. Rendered as a callout, and worth writing
//            wherever a value is easy to misread.
// `also`     ids of related options, cross-linked in the guide.

export interface OptionDoc {
  id: string
  group: string
  title: string
  summary: string
  detail: string[]
  caveat?: string
  also?: string[]
}

export const OPTION_DOCS: OptionDoc[] = [
  // ── Map: encoding ─────────────────────────────────────────────────────────
  {
    id: 'map-color-by',
    group: 'Map',
    title: 'Color by',
    summary: 'What the color of each dot means, pick any category or measurement.',
    detail: [
      'Categories (species, genus, cluster, land cover, year, month, enrichment level) get distinct colors from the active palette. Numeric fields (elevation, slope, NDVI, soil moisture, temperature, exposure) get a light-to-dark gradient instead, because a ramp reads as "more" and a set of hues does not.',
      'Category colors are stable and shared: a species is the same color here, in every chart, and in the legend. That is what makes it possible to look at the map and a box plot side by side and match them up. Change one under Style and it changes everywhere.',
    ],
    also: ['map-size-by', 'appearance-palette', 'appearance-overrides'],
  },
  {
    id: 'map-size-by',
    group: 'Map',
    title: 'Size by',
    summary: 'Scale each dot by a numeric field, so two dimensions read at once.',
    detail: [
      'Dot area follows the chosen measurement, so you can color by species and size by elevation and see both together.',
      'Area, not radius, carries the value, sizing by radius makes a value twice as large look four times as big.',
    ],
    caveat: 'With 48,000 overlapping points, size is easy to misjudge in dense areas. Use it to spot gradients across a region, not to compare two individual dots.',
    also: ['map-color-by', 'appearance-point-size'],
  },
  {
    id: 'map-show-points',
    group: 'Map',
    title: 'Show observations',
    summary: 'Hide the individual dots to read a heatmap on its own.',
    detail: [
      'The points and the grid heatmap compete for the same space. Turning the points off leaves the heatmap legible, which is the difference between seeing a seasonal pattern and seeing a wall of dots.',
      'The choice is remembered between visits.',
    ],
    also: ['map-heatmap', 'appearance-point-opacity'],
  },

  // ── Map: overlays ─────────────────────────────────────────────────────────
  {
    id: 'map-heatmap',
    group: 'Map',
    title: 'Heatmap',
    summary: 'A grid of per-cell summaries, computed from the observations and drawn under the points.',
    detail: [
      '48,000 overlapping dots show you where the data is dense. A grid shows what is actually in an area. Each cell aggregates the observations inside it and is shaded by the result.',
      'The heatmaps differ mainly in how much they are distorted by recording effort, how many people went looking there. That difference matters more than anything else on this page, so each mode below says where it stands.',
    ],
    also: ['map-heatmap-density', 'map-heatmap-richness', 'map-heatmap-season',
      'map-heatmap-hotspots', 'map-heatmap-common', 'map-heatmap-land-cover',
      'map-heatmap-wind', 'map-heatmap-field', 'map-cell-size'],
  },
  {
    id: 'map-heatmap-density',
    group: 'Map',
    title: 'Heatmap: Observation density',
    summary: 'How many records fall in each cell.',
    detail: [
      'A straight count. Useful for seeing the shape of the dataset, where the records are and where there are none.',
    ],
    caveat: 'This is a map of where people went, not where mushrooms are. Cities, trailheads and popular parks glow because they are visited, not because they are productive. Do not read it as habitat.',
    also: ['map-heatmap-season'],
  },
  {
    id: 'map-heatmap-richness',
    group: 'Map',
    title: 'Heatmap: Species richness',
    summary: 'How many distinct species were recorded in each cell.',
    detail: [
      'Counts unique species rather than records. A cell with 200 observations of one species scores 1.',
    ],
    caveat: 'Still effort-sensitive, and in a specific direction: more visits find more species, so richness climbs with sampling long before it reflects real diversity. Compare cells with similar record counts, or use Seasonal activity instead.',
    also: ['map-heatmap-density', 'map-heatmap-season'],
  },
  {
    id: 'map-heatmap-season',
    group: 'Map',
    title: 'Heatmap: Seasonal activity',
    summary: "The share of a cell's own finds that fall inside your date window.",
    detail: [
      'For each cell: of all the observations ever recorded there, what fraction happened within ±window of your chosen day of the year, across all years.',
      'Dividing by the cell\'s own total is what makes this worth trusting. A cell visited 500 times and a cell visited 10 times are on the same scale, because effort cancels out of a ratio, the seasonal *shape* of a place does not depend on how many people went, only on when they found things.',
      'Cells with fewer than 3 records are left blank rather than shown at 0% or 100% on a single observation.',
    ],
    caveat: 'It describes when finds happened historically. It is not a forecast, and a warm or dry year will not match it.',
    also: ['map-heatmap-hotspots', 'map-season-day', 'map-season-window'],
  },
  {
    id: 'map-heatmap-hotspots',
    group: 'Map',
    title: 'Heatmap: In-season hotspots',
    summary: 'Seasonal activity, weighted by how well-sampled the cell is.',
    detail: [
      'Takes the Seasonal activity share and weights it by how much evidence sits behind it, so a cell at 80% from 4 records does not outrank a cell at 60% from 200.',
      'This is the closest thing here to "where would I go this week", which is why it is weighted, a confident 60% is a better bet than a noisy 80%.',
    ],
    caveat: 'It is a record of past finds, not a prediction. Weighting by sample size also reintroduces some effort bias, since well-sampled cells are well-visited ones.',
    also: ['map-heatmap-season', 'map-season-day', 'map-season-window'],
  },
  {
    id: 'map-heatmap-common',
    group: 'Map',
    title: 'Heatmap: Most common species',
    summary: 'The most-recorded species in each cell, by color.',
    detail: [
      'Shows regional character, where one species takes over. Colors match the rest of the app, so a cell can be read against the legend and the charts.',
    ],
    caveat: 'Only the winner is shown, so a cell split 51/49 between two species looks identical to one that is 100% a single species. Ties are broken arbitrarily.',
    also: ['map-color-by'],
  },
  {
    id: 'map-heatmap-land-cover',
    group: 'Map',
    title: 'Heatmap: Land cover',
    summary: 'The most common land-cover class recorded across each cell.',
    detail: [
      'Reads the land-cover class the pipeline sampled at each observation and shows whichever one appears most often in the cell: forest, shrubland, grassland, and so on.',
      'This is land cover *at the finds*, not a land-cover map. It says what kind of ground the records in a cell sit on, which is the question worth asking of a foraging map; a real land-cover raster is available separately as a reference layer in the layers control.',
    ],
    caveat: 'Only the winner is shown, so a cell that is half forest and half meadow looks the same as one that is all forest. Cells with no finds are blank, not unclassified ground.',
    also: ['map-heatmap-common', 'map-basemaps'],
  },
  {
    id: 'map-heatmap-wind',
    group: 'Map',
    title: 'Heatmap: Wind / aspect vectors',
    summary: 'Arrows for wind where it exists, and for the way slopes face where it does not.',
    detail: [
      'The observations carry terrain **aspect**: the compass direction a slope faces, and a wind-exposure index, but no measured wind. Until the pipeline samples ERA5 wind into `wind_u` / `wind_v`, this heatmap draws mean terrain aspect: arrow direction is the way the ground faces, arrow length is how *consistently* the cell faces one way, and color is wind exposure.',
      'When real wind data is present the heatmap switches to it automatically and relabels its legend, so the arrows never silently change meaning.',
    ],
    caveat: 'A short arrow means mixed terrain, not calm air. Aspect is averaged as a vector, because averaging compass degrees numerically puts the mean of 350° and 10° at 180°, exactly backwards.',
    also: ['map-heatmap-field', 'map-cell-size'],
  },
  {
    id: 'map-heatmap-field',
    group: 'Map',
    title: 'Heatmap: environmental fields',
    summary: 'The cell mean of one enriched value, rainfall, soil moisture, NDVI, slope, aspect, TWI, sun or wind exposure.',
    detail: [
      'Every observation carries the terrain and weather the pipeline sampled at its location. Averaging that across a grid cell turns 48,000 scattered readings into a surface you can read: where the ground is steep, where it holds water, where the canopy is wet, how much rain fell in the week before finds were made.',
      'Aspect is the exception and is treated as circular: it is averaged as a vector, so 350° and 10° average to 0° rather than 180°, and its key is a compass wheel rather than a low-to-high bar.',
      'The tooltip on a point reports the cell mean along with how many observations went into it, because a mean of two is a different claim from a mean of two hundred.',
    ],
    caveat: 'These are not rasters. A cell has a value only where somebody recorded a find, so a blank cell means nobody looked there, not that the ground is dry, flat or bare. And because the sample points are wherever people walked, a cell mean describes the places finds were made in that cell, not the cell as a whole.',
    also: ['map-heatmap', 'map-cell-size'],
  },
  {
    id: 'map-cell-size',
    group: 'Map',
    title: 'Cell size',
    summary: 'How coarse the heatmap grid is, from about 500 m to about 28 km.',
    detail: [
      'Smaller cells resolve more detail and hold fewer records each, so they are noisier; larger cells are steadier but blur real boundaries.',
      'If a heatmap looks like static, the cells are probably too small for the number of records in view.',
      'Below about a kilometre the grid stops summarising and starts drawing roughly one cell per observation, at which point it is the point layer with square markers. The legend reports the cell count, so compare it against how many records are in view to see whether that has happened.',
    ],
    also: ['map-heatmap'],
  },
  {
    id: 'map-season-day',
    group: 'Map',
    title: 'Date',
    summary: 'The day of the year the seasonal heatmaps are centred on.',
    detail: [
      'Year is ignored, every year\'s records are pooled by day of the year, so this asks "what happens around this time of year", not "what happened on this date".',
    ],
    also: ['map-season-window', 'map-heatmap-season'],
  },
  {
    id: 'map-season-window',
    group: 'Map',
    title: 'Window',
    summary: 'How many days either side of the date to count, from ±3 to ±60.',
    detail: [
      'A narrow window is specific but thin on data; a wide one is steadier but smears the season. ±14 days is a reasonable starting point for most species.',
      'The date span the current setting covers is spelled out under the slider.',
    ],
    also: ['map-season-day', 'map-heatmap-season'],
  },

  // ── Map: other controls ───────────────────────────────────────────────────
  {
    id: 'map-offline',
    group: 'Map',
    title: 'Offline',
    summary: 'Keep the app, the observations and an area of map tiles in this browser, for use with no signal.',
    detail: [
      'A map of where things grow is most often read standing in the place it describes, which is where there is least likely to be a connection. Three things can be saved, separately, because they cost very different amounts:',
      '**The app**: every page and the code behind it, so Charts and Analysis open offline too and not just the page you happened to be on.',
      '**Observations**, the dataset the map, table and charts all read from. This is the big one, tens of megabytes.',
      '**An area**: the imagery for the place on screen, plus however many zoom levels closer you ask for, across every layer currently drawn. Pan and zoom to where you are going first: this saves what is in front of you, not the world. The tile count and a rough size are shown before anything is downloaded, because the count roughly quadruples per extra zoom level.',
      'An area is named when you save it, and the collection is managed on the **Offline** page: rename, re-save, delete, or jump the map back to one. Naming matters more than it sounds — a list of saved places is read weeks later, by which point a set of coordinates is a puzzle where "north ridge" is an answer.',
      'Deleting an area frees only the tiles no other saved area still needs. Two areas over the same valley share tiles, and removing one must not punch a hole in the other.',
      'Nothing is saved on its own beyond the app shell. Downloading a dataset and a few hundred tiles onto someone\'s mobile data without being asked is not a feature.',
    ],
    caveat: 'Saved data lives in this browser on this device. It is not uploaded, does not follow your account, and clearing the browser\'s site data removes it. Sizes shown per area are estimates: tiles are fetched from hosts that do not all report a length, and once stored they cannot be measured — the total on the Offline page is the browser\'s own figure and is the accurate one. Tile services also set their own terms on bulk downloading; save the area you are going to, not a region.',
    also: ['map-basemaps', 'map-layer-order'],
  },
  {
    id: 'map-layer-order',
    group: 'Map',
    title: 'Layer order and opacity',
    summary: 'Which overlay draws over which, and how far through each one you can see.',
    detail: [
      'Overlays stack, and they hide each other. Land ownership under a hillshade is a different map from the same two the other way up, and until there was an order there was no way to say which you meant — a layer switched on landed wherever the catalogue happened to put it.',
      'In the layer manager, everything currently drawn is listed at the top, topmost first, with arrows to move it. A layer you switch on goes to the top, which is where someone who just asked for it expects to find it.',
      'Each drawn layer also has its own opacity. This multiplies into the global **Tile opacity** in Appearance rather than replacing it, so a hillshade meant to sit lightly stays proportionally lighter, and dimming the whole pile to read through it no longer dims the one layer you were trying to read.',
      '**⤒** and **⤓** send a layer straight to the top or the bottom. With eight layers on, reaching the top by pressing "up" eight times is a counting exercise rather than an ordering control.',
    ],
    also: ['map-layer-solo', 'map-layer-blend', 'map-basemaps', 'appearance-tile-opacity'],
  },
  {
    id: 'map-soil-taxonomy',
    group: 'Map',
    title: 'Soil taxonomy layers',
    summary: 'The USDA soil classification, with a browser for its four hundred classes.',
    detail: [
      '**Soil taxonomy (USDA orders)** paints the soil order at each pixel, at 250 m worldwide. Twelve orders, which is the top of the hierarchy: they separate soils by how they formed, so the boundaries follow geology and climate rather than anything visible on the surface.',
      '**Soil taxonomy: chosen classes** paints only the great groups you tick, and leaves everything else blank. Each one keeps its order’s colour, so a selection spanning several orders can still be told apart.',
      'The source is the great-group level, which has around four hundred classes. Four hundred swatches is not a key, so these layers get a browser instead: the twelve orders as clickable chips, a search box that matches a great group or its order, and a row per class that opens into what the name means and a link to the article for its order.',
      'On the chosen-classes layer each row also gains a checkbox, and **Pick these** takes everything the search and the order chip leave — filter to Spodosols, press it, and you have selected the podzols. **FRMS set** fills in eighteen great groups members flagged as matsutake ground on the Front Range, as a starting point to add to or cut down.',
      'Soil names are compositional and the browser decodes them. *Dystrocryepts* is dystro- (acid, low in bases) plus cry- (cold) plus -epts (Inceptisols), so it is a cold acid soil with weak horizons. The ending is always the order.',
    ],
    caveat: 'The chosen-classes layer is a soil filter, not a prediction. It says the ground is the kind you asked for — not that anything grows there, and nothing at all about the trees that decide whether anything can. Both layers are a model prediction rather than a soil survey: right about a hillside, unreliable about a square metre.',
    also: ['map-basemaps', 'map-layer-order'],
  },
  {
    id: 'map-layer-solo',
    group: 'Map',
    title: 'Solo a layer',
    summary: 'Draw one layer on its own, without switching the rest off.',
    detail: [
      'The **S** beside a drawn layer hides every other overlay while leaving them switched on. They stay in the list, dimmed, and come back exactly as they were.',
      'The question this answers is "what is this one actually contributing", which comes up constantly in a stack of five and which you otherwise pay for by dismantling the stack and rebuilding it.',
      'Switching any layer on or off ends the solo, because asking for a second layer is asking to see two.',
    ],
    also: ['map-layer-order', 'map-layer-blend'],
  },
  {
    id: 'map-layer-blend',
    group: 'Map',
    title: 'Blend mode',
    summary: 'How a layer combines with the layers below it, rather than simply covering them.',
    detail: [
      'Opacity answers "how much of this do I want" and answers it badly for two rasters: two layers at 50% is each of them half washed out, and the thing you wanted — the shape of the hillshade over the colour of the land cover — is exactly what that destroys.',
      'A blend mode keeps both at full strength and combines them by value instead. **Multiply** keeps what is dark in both, which is the one to reach for when relief should read through colour. **Screen** keeps what is light in both, which suits burn scars or cloud over terrain. **Difference** shows what two layers disagree about, which is how you compare the same product on two dates.',
      'The browser composites this per frame. No tile is re-fetched and nothing is recomputed, so it costs nothing to try one and change your mind.',
      'The default for each layer is **Default**, which follows the **Blend when stacked** setting in Appearance. Choosing a mode here pins that layer, and it then keeps what you set whatever the default does later.',
    ],
    caveat: 'A blend mode combines a layer with whatever is below it, which includes the basemap. The same pair of layers over satellite imagery and over the gray canvas will not look the same, and neither result is the layer on its own — use Solo for that.',
    also: ['appearance-stack-blend', 'map-layer-solo', 'map-layer-order'],
  },
  {
    id: 'map-basemaps',
    group: 'Map',
    title: 'Basemaps and layers',
    summary: 'A muted basemap, plus reference layers on top: weather, land cover, greenness, relief, trails and ownership.',
    detail: [
      'One basemap at a time, any number of the reference layers on top of it. Those are somebody else\'s imagery, distinct from the **Heatmap** picker in the control bar, which draws a grid computed from the observations themselves.',
      'They are two controls because they are two questions. The **Basemap** button is a single choice from five, made once and rarely revisited. The **Layers** button opens a window for the overlays, which is where the list has grown — the built-in catalogue, the Earth Engine layers, and and whatever assets FRMS has registered of its own. That window stays open while you work the map, because whether a layer was worth switching on is something you judge by looking at the map, not at the list.',
      'The map opens on a light grey canvas on purpose. A street or topo map is drawn to be read on its own, and once 48,000 colored dots sit on it, its color competes with the data for the same hues. A grey base leaves the dots as the only saturated thing on screen. For terrain without that cost, keep the grey base and switch on **Hillshade**, relief in grey, color still reserved for the observations.',
      'These are third-party tile services. If one cannot be reached the map says so rather than drawing nothing, an empty ownership layer would otherwise read as "no public land here".',
    ],
    also: ['map-layer-weather', 'map-layer-ground', 'map-trails', 'map-land-ownership',
      'appearance-tile-opacity', 'map-heatmap'],
  },
  {
    id: 'map-layer-weather',
    group: 'Map',
    title: 'Weather layers',
    summary: 'Live US radar, US rainfall over the past 24 hours, global satellite rainfall, and land surface temperature.',
    detail: [
      '**Radar (US, now)** is live NEXRAD base reflectivity. It is what the radar is seeing this minute, not an accumulation, and reflectivity is not rainfall: hail, bright banding at the melting layer, and ground clutter near the antenna all return signal.',
      '**Rain past 24h (US)** is the River Forecast Centers\' quantitative precipitation estimate: radar corrected against gauges. It is the closest thing here to "how much fell", and it is still an estimate.',
      '**Rainfall (global)** is NASA\'s GPM IMERG precipitation rate at about 10 km. That cell is larger than most of the places on this map, so it tells you the weather over an area, not whether it rained on a particular hillside.',
      '**Land surface temp** is the temperature of the ground itself, not the air, bare rock in full sun reads far hotter than the air above it, and a forest canopy reads cooler.',
      'The global layers vary by day, and the key carries a date picker. Each product has its own latency, so the date opens a few days back: asking for today generally returns blank tiles, which would read as "no rain" rather than "not processed yet".',
    ],
    caveat: 'None of these is measured at the observations. They are current or recent conditions drawn over historical finds, so they cannot explain the finds, for that, the rainfall and temperature **heatmaps** use the values the pipeline sampled at each observation\'s own date.',
    also: ['map-heatmap-field', 'appearance-tile-opacity'],
  },
  {
    id: 'map-layer-ground',
    group: 'Map',
    title: 'Ground and vegetation layers',
    summary: 'ESA land cover at 10 m, modelled soil moisture, and MODIS greenness.',
    detail: [
      '**Land cover (ESA)** is WorldCover 2021 at 10 m, the highest-resolution global land-cover map there is, with its own official class colors so it matches every other rendering of the product. It is from 2021: a burn, a clear-cut or a new development since then is not in it.',
      '**Soil moisture** is SMAP L4, a model that assimilates satellite retrievals, giving water in the top 5 cm at about 9 km. A model output over a wide cell is not a reading of your patch.',
      '**NDVI (greenness)** is an 8-day MODIS composite at 250 m. Dense conifer and dense broadleaf both saturate near the top of the scale, so it separates bare ground from vegetation far better than it separates one forest from another.',
    ],
    caveat: 'Resolution and currency pull in opposite directions here. Land cover is sharp and three years stale; soil moisture is current and 9 km coarse. Read each for what it is good at rather than treating them as one picture of the ground.',
    also: ['map-heatmap-field', 'map-basemaps'],
  },
  {
    id: 'map-trails',
    group: 'Map',
    title: 'Hiking trails layer',
    summary: 'Waymarked hiking routes from OpenStreetMap, drawn over the basemap.',
    detail: [
      'Renders OpenStreetMap\'s hiking *route relations*, named, waymarked routes, rather than every footpath in the database. It answers "how would I get near this" for a cluster of finds that looks promising.',
    ],
    caveat: 'An unmapped path is missing from this layer, not abs...',
  },
  {
    id: 'map-land-ownership',
    group: 'Map',
    title: 'Land ownership (US)',
    summary: 'BLM\'s Surface Management Agency layer: which federal agency, state, or private party manages each parcel.',
    detail: [
      '**BLM\'s Surface Management Agency layer** paints the soil order at each pixel, at 250 m worldwide. Twelve orders, which is the top of the hierarchy: they separate soils by how they formed, so the boundaries follow geology and climate rather than anything visible on the surface.',
      '**Chosen classes** paints only the great groups you tick, and leaves everything else blank. Each one keeps its order’s colour, so a selection spanning several orders can still be told apart.',
      'The source is the great-group level, which has around four hundred classes. Four hundred swatches is not a key, so these layers get a browser instead: the twelve orders as clickable chips, a search box that matches a great group or its order, and a row per class that opens into what the name means and a link to the article for its order.',
      'On the chosen-classes layer each row also gains a checkbox, and **Pick these** takes everything the search and the order chip leave — filter to Spodosols, press it, and you have selected the podzols. **FRMS set** fills in eighteen great groups members flagged as matsutake ground on the Front Range, as a starting point to add to or cut down.',
      'Soil names are compositional and the browser decodes them. *Dystrocryepts* is dystro- (acid, low in bases) plus cry- (cold) plus -epts (Inceptisols), so it is a cold acid soil with weak horizons. The ending is always the order.',
    ],
    caveat: 'The chosen-classes layer is a soil filter, not a prediction. It says the ground is the kind you asked for — not that anything grows there, and nothing at all about the trees that decide whether anything can. Both layers are a model prediction rather than a soil survey: right about a hillside, unreliable about a square metre.',
    also: ['map-basemaps', 'map-layer-order'],
  },
  {
    id: 'appearance-palette',
    group: 'Appearance',
    title: 'Palette',
    summary: 'The set of colors used to draw observations by species or cluster.',
    detail: [
      'A palette is a list of distinct hues. When the data grows, we add new colors to the list. If you have a favorite set, you can override it in the settings.',
    ],
    also: ['appearance-overrides', 'map-color-by'],
  },
  {
    id: 'appearance-overrides',
    group: 'Appearance',
    title: 'Palette Overrides',
    summary: 'Manually assign a color to a specific species or cluster.',
    detail: [
      'If a species is assigned a generic blue by the palette, you can override it with a specific forest-green. This override persists across sessions.',
    ],
    also: ['appearance-palette', 'map-color-by'],
  },
  {
    id: 'appearance-point-size',
    group: 'Appearance',
    title: 'Point size',
    summary: 'The base diameter of an observation dot.',
    detail: [
      'A larger base size makes dots easier to see but increases overlap in dense clusters. The choice is remembered between visits.',
    ],
    also: ['map-size-by'],
  },
  {
    id: 'appearance-point-opacity',
    group: 'Appearance',
    title: 'Point opacity',
    summary: 'How transparent the dots are, so you can see the heatmap beneath.',
    detail: [
      'Lower opacity makes the a "cloud" of points, where the density is revealed by the saturation of the color. Full opacity creates a wall of dots.',
    ],
    also: ['map-show-points'],
  },
  {
    id: 'appearance-tile-opacity',
    group: 'Appearance',
    title: 'Tile opacity',
    summary: 'The global transparency of all reference layers combined.',
    detail: [
      'This acts as a master fader for all overlays. Dimming the whole pile makes the observations the most saturated thing on screen, while increasing it emphasizes the environmental context.',
    ],
    also: ['map-basemaps', 'map-layer-order'],
  },
  {
    id: 'appearance-stack-blend',
    group: 'Appearance',
    title: 'Blend when stacked',
    summary: 'The default blend mode for overlays, unless a layer has its own pinned mode.',
    detail: [
      'If set to Multiply, all layers combine by multiplying their values. This is usually the best for relief over colour.',
    ],
    also: ['map-layer-blend'],
  },
]

// ─── Lookups ─────────────────────────────────────────────────────────────────
// The reference page, the tooltips and the guide all read the same table through
// these, so an option is described once and linked to the same place everywhere.

/** The entry for an option id, or null when there is none. */
export function docFor(id: string): OptionDoc | null {
  return OPTION_DOCS.find((d) => d.id === id) || null
}

/** An option's one-line summary, or '' when the id is unknown. */
export function docSummary(id: string): string {
  return docFor(id)?.summary || ''
}

/**
 * The anchor for an option on the reference page.
 *
 * Namespaced with `opt-` because the reference and the guide's prose slug their
 * headings the same way, and several options share a name with a section
 * ("Filters", "Heatmap", "Share"); without the prefix the two claim one id and a
 * tooltip scrolls to the wrong thing.
 */
export function docAnchor(id: string): string {
  return `opt-${id}`
}

/** A link straight to an option's entry on the reference page. */
export function docHref(id: string): string {
  return `/guide/reference#${docAnchor(id)}`
}

/** The options grouped for display, each group named once, in declaration order. */
export function docGroups(): { name: string; items: OptionDoc[] }[] {
  const groups: { name: string; items: OptionDoc[] }[] = []
  for (const doc of OPTION_DOCS) {
    let group = groups.find((g) => g.name === doc.group)
    if (!group) { group = { name: doc.group, items: [] }; groups.push(group) }
    group.items.push(doc)
  }
  return groups
}
