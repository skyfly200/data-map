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

export const OPTION_DOCS = [
  // ── Map: encoding ─────────────────────────────────────────────────────────
  {
    id: 'map-color-by',
    group: 'Map',
    title: 'Color by',
    summary: 'What the colour of each dot means — pick any category or measurement.',
    detail: [
      'Categories (species, genus, cluster, land cover, year, month, enrichment level) get distinct colours from the active palette. Numeric fields (elevation, slope, NDVI, soil moisture, temperature, exposure) get a light-to-dark gradient instead, because a ramp reads as "more" and a set of hues does not.',
      'Category colours are stable and shared: a species is the same colour here, in every chart, and in the legend. That is what makes it possible to look at the map and a box plot side by side and match them up. Change one in Appearance and it changes everywhere.',
    ],
    also: ['map-size-by', 'appearance-palette', 'appearance-overrides'],
  },
  {
    id: 'map-size-by',
    group: 'Map',
    title: 'Size by',
    summary: 'Scale each dot by a numeric field, so two dimensions read at once.',
    detail: [
      'Dot area follows the chosen measurement, so you can colour by species and size by elevation and see both together.',
      'Area, not radius, carries the value — sizing by radius makes a value twice as large look four times as big.',
    ],
    caveat: 'With 48,000 overlapping points, size is easy to misjudge in dense areas. Use it to spot gradients across a region, not to compare two individual dots.',
    also: ['map-color-by', 'appearance-point-size'],
  },
  {
    id: 'map-show-points',
    group: 'Map',
    title: 'Show observations',
    summary: 'Hide the individual dots to read an overlay on its own.',
    detail: [
      'The points and the grid overlay compete for the same space. Turning the points off leaves the overlay legible, which is the difference between seeing a seasonal pattern and seeing a wall of dots.',
      'The choice is remembered between visits.',
    ],
    also: ['map-overlay', 'appearance-point-opacity'],
  },

  // ── Map: overlays ─────────────────────────────────────────────────────────
  {
    id: 'map-overlay',
    group: 'Map',
    title: 'Overlay',
    summary: 'A grid of per-cell summaries drawn under the points.',
    detail: [
      '48,000 overlapping dots show you where the data is dense. A grid shows what is actually in an area. Each cell aggregates the observations inside it and is shaded by the result.',
      'The overlays differ mainly in how much they are distorted by recording effort — how many people went looking there. That difference matters more than anything else on this page, so each mode below says where it stands.',
    ],
    also: ['map-overlay-density', 'map-overlay-richness', 'map-overlay-season',
      'map-overlay-hotspots', 'map-overlay-dominant', 'map-overlay-wind', 'map-cell-size'],
  },
  {
    id: 'map-overlay-density',
    group: 'Map',
    title: 'Overlay: Observation density',
    summary: 'How many records fall in each cell.',
    detail: [
      'A straight count. Useful for seeing the shape of the dataset — where the records are and where there are none.',
    ],
    caveat: 'This is a map of where people went, not where mushrooms are. Cities, trailheads and popular parks glow because they are visited, not because they are productive. Do not read it as habitat.',
    also: ['map-overlay-season'],
  },
  {
    id: 'map-overlay-richness',
    group: 'Map',
    title: 'Overlay: Species richness',
    summary: 'How many distinct species were recorded in each cell.',
    detail: [
      'Counts unique species rather than records. A cell with 200 observations of one species scores 1.',
    ],
    caveat: 'Still effort-sensitive, and in a specific direction: more visits find more species, so richness climbs with sampling long before it reflects real diversity. Compare cells with similar record counts, or use Seasonal activity instead.',
    also: ['map-overlay-density', 'map-overlay-season'],
  },
  {
    id: 'map-overlay-season',
    group: 'Map',
    title: 'Overlay: Seasonal activity',
    summary: "The share of a cell's own finds that fall inside your date window.",
    detail: [
      'For each cell: of all the observations ever recorded there, what fraction happened within ±window of your chosen day of the year, across all years.',
      'Dividing by the cell\'s own total is what makes this worth trusting. A cell visited 500 times and a cell visited 10 times are on the same scale, because effort cancels out of a ratio — the seasonal *shape* of a place does not depend on how many people went, only on when they found things.',
      'Cells with fewer than 3 records are left blank rather than shown at 0% or 100% on a single observation.',
    ],
    caveat: 'It describes when finds happened historically. It is not a forecast, and a warm or dry year will not match it.',
    also: ['map-season-day', 'map-season-window', 'map-overlay-hotspots'],
  },
  {
    id: 'map-overlay-hotspots',
    group: 'Map',
    title: 'Overlay: In-season hotspots',
    summary: 'Seasonal activity, weighted by how well-sampled the cell is.',
    detail: [
      'Takes the Seasonal activity share and weights it by how much evidence sits behind it, so a cell at 80% from 4 records does not outrank a cell at 60% from 200.',
      'This is the closest thing here to "where would I go this week", which is why it is weighted — a confident 60% is a better bet than a noisy 80%.',
    ],
    caveat: 'It is a record of past finds, not a prediction. Weighting by sample size also reintroduces some effort bias, since well-sampled cells are well-visited ones.',
    also: ['map-overlay-season', 'map-season-day', 'map-season-window'],
  },
  {
    id: 'map-overlay-dominant',
    group: 'Map',
    title: 'Overlay: Dominant species',
    summary: 'The most-recorded species in each cell, by colour.',
    detail: [
      'Shows regional character — where one species takes over. Colours match the rest of the app, so a cell can be read against the legend and the charts.',
    ],
    caveat: 'Only the winner is shown, so a cell split 51/49 between two species looks identical to one that is 100% a single species. Ties are broken arbitrarily.',
    also: ['map-color-by'],
  },
  {
    id: 'map-overlay-wind',
    group: 'Map',
    title: 'Overlay: Wind / aspect vectors',
    summary: 'Arrows for wind where it exists, and for the way slopes face where it does not.',
    detail: [
      'The observations carry terrain **aspect** — the compass direction a slope faces — and a wind-exposure index, but no measured wind. Until the pipeline samples ERA5 wind into `wind_u` / `wind_v`, this overlay draws mean terrain aspect: arrow direction is the way the ground faces, arrow length is how *consistently* the cell faces one way, and colour is wind exposure.',
      'When real wind data is present the overlay switches to it automatically and relabels its legend, so the arrows never silently change meaning.',
    ],
    caveat: 'A short arrow means mixed terrain, not calm air. Aspect is averaged as a vector, because averaging compass degrees numerically puts the mean of 350° and 10° at 180° — exactly backwards.',
  },
  {
    id: 'map-cell-size',
    group: 'Map',
    title: 'Cell size',
    summary: 'How coarse the overlay grid is, from about 2 km to about 28 km.',
    detail: [
      'Smaller cells resolve more detail and hold fewer records each, so they are noisier; larger cells are steadier but blur real boundaries.',
      'If an overlay looks like static, the cells are probably too small for the number of records in view.',
    ],
    also: ['map-overlay'],
  },
  {
    id: 'map-season-day',
    group: 'Map',
    title: 'Date',
    summary: 'The day of the year the seasonal overlays are centred on.',
    detail: [
      'Year is ignored — every year\'s records are pooled by day of the year, so this asks "what happens around this time of year", not "what happened on this date".',
    ],
    also: ['map-season-window', 'map-overlay-season'],
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
    also: ['map-season-day', 'map-overlay-season'],
  },

  // ── Map: other controls ───────────────────────────────────────────────────
  {
    id: 'map-basemaps',
    group: 'Map',
    title: 'Basemaps and layers',
    summary: 'Street, terrain or satellite underneath, plus USGS topo, imagery, hillshade and relief on top.',
    detail: [
      'One basemap at a time, any number of the overlay layers on top of it. Hillshade over street is a cheap way to read terrain without losing labels.',
    ],
  },
  {
    id: 'map-locate',
    group: 'Map',
    title: 'My location',
    summary: 'Puts a dot at your position, with a circle showing its accuracy.',
    detail: [
      'Uses the browser\'s geolocation, so it asks permission first. The circle is the accuracy the device reports — a wide circle means the fix is poor, not that the area is interesting.',
      'Your position is never sent anywhere; it is only drawn on your own map.',
    ],
  },
  {
    id: 'map-live-clusters',
    group: 'Map',
    title: 'Live clustering',
    summary: 'Group observations by k-means in the browser, by environment or by geography.',
    detail: [
      'Runs k-means over the observations currently in view. Clustering by *features* groups finds that share an environment — similar elevation, temperature, moisture — wherever they are. Clustering by *geography* groups them by location alone.',
      'The result becomes available as a colouring and as a chart field, so a cluster can be inspected rather than just admired.',
      'This is exploratory: k-means will always return the number of groups you ask for, whether or not the data has that many. Changing k and seeing whether the groups survive is the check.',
    ],
    caveat: 'Clusters are descriptions of this dataset, not species categories or habitat types. Two runs with different k are not comparable.',
  },
  {
    id: 'map-save-image',
    group: 'Map',
    title: 'Save image',
    summary: 'Flattens the map — basemap tiles, overlay and points — into a single PNG.',
    detail: [
      'Composites what is on screen, including the tile layers, rather than exporting only the vector layer.',
    ],
  },
  {
    id: 'data-show-filtered',
    group: 'Map',
    title: 'Include excluded water / non-terrestrial rows',
    summary: 'Show observations the pipeline flagged as falling on water, ice or built-up land.',
    detail: [
      'The enrichment pipeline looks up the land cover under every observation and flags the ones landing on open water, permanent snow and ice, or built-up surfaces (land-cover codes 50, 70 and 80). Those are almost always a GPS error or a coordinate rounded off a shoreline, not a mushroom growing in a lake — so they are excluded from the terrestrial dataset the analysis uses. This checkbox puts them back in view.',
      '**In the currently published dataset this control does nothing, and the reason is worth knowing.** The pipeline sets the `water_mask` flag and then drops the flagged rows before exporting, so the published file contains 48,233 observations of which zero are flagged. The toggle only has something to reveal if the pipeline is run with `FILTER_NON_PRODUCTIVE_LANDCOVER=0`, which keeps the flagged rows in the export and leaves the decision to the viewer.',
    ],
    caveat: 'Turning this on does not recover observations from the published dataset — they were removed upstream, not hidden downstream.',
  },

  // ── Appearance ────────────────────────────────────────────────────────────
  {
    id: 'appearance-palette',
    group: 'Appearance',
    title: 'Palette',
    summary: 'The set of colours categories are drawn from.',
    detail: [
      'Okabe–Ito is the safe default for colour-vision deficiency; the others trade some of that for contrast or for a more muted look on satellite imagery.',
      'A palette change applies everywhere at once — map, charts and legends — because category colours are shared.',
    ],
    also: ['appearance-overrides', 'map-color-by'],
  },
  {
    id: 'appearance-shapes',
    group: 'Appearance',
    title: 'Shape set',
    summary: 'Which marker shapes are used when a chart encodes a category by shape.',
    detail: [
      'Shape survives printing in black and white and is readable to anyone who cannot separate two hues, so it is worth using alongside colour rather than instead of it.',
    ],
  },
  {
    id: 'appearance-point-size',
    group: 'Appearance',
    title: 'Point size',
    summary: 'The radius of each map dot, from 1 to 10 pixels.',
    detail: [
      'Smaller points make dense areas readable; larger ones make sparse regions visible when zoomed out.',
    ],
    also: ['map-size-by'],
  },
  {
    id: 'appearance-point-opacity',
    group: 'Appearance',
    title: 'Opacity',
    summary: 'How solid the map dots are — lower values let density show through as shading.',
    detail: [
      'At low opacity, overlapping dots accumulate into darker patches, which turns the point layer into its own density map. This is often more informative than the density overlay because it keeps every point\'s colour.',
      'The outline follows this slider too, so fading the dots fades the whole marker rather than leaving a mesh of solid rings behind.',
    ],
    also: ['appearance-point-outline', 'map-overlay-density'],
  },
  {
    id: 'appearance-point-outline',
    group: 'Appearance',
    title: 'Outline map dots',
    summary: 'The dark ring around each dot — helpful when sparse, a grey mass when dense.',
    detail: [
      'The ring separates overlapping finds and makes pale colours readable against a light basemap. Over a dense patch the rings merge into grey and hide the colours they were drawn to separate, so it can be turned off.',
      'Turning it off is usually the right move when zoomed out over a well-sampled region; turning it back on helps when inspecting a handful of points.',
    ],
    also: ['appearance-point-opacity'],
  },
  {
    id: 'appearance-overrides',
    group: 'Appearance',
    title: 'Per-value colour and shape',
    summary: 'Pin a specific species (or other category) to a colour or shape of your choosing.',
    detail: [
      'Click a swatch to recolour that value. The override applies across the map and every chart, and persists — so the species you care about can keep the same colour across sessions and shared links.',
    ],
    also: ['appearance-palette', 'appearance-shapes'],
  },

  // ── Chart builder: types ──────────────────────────────────────────────────
  {
    id: 'chart-type',
    group: 'Charts',
    title: 'Chart type',
    summary: 'What kind of chart to draw — which then decides which fields you can set.',
    detail: [
      'The controls below the type change with it, because a scatter plot needs an X and a Y while a donut needs a grouping and a measure. Switching type keeps whatever fields still apply.',
    ],
    also: ['chart-scatter', 'chart-bar', 'chart-line', 'chart-box', 'chart-histogram',
      'chart-heatmap', 'chart-radar', 'chart-donut'],
  },
  {
    id: 'chart-scatter',
    group: 'Charts',
    title: 'Scatter',
    summary: 'One mark per observation, positioned by two measurements.',
    detail: [
      'The only chart here that shows individual records rather than aggregates, so it is where relationships and outliers actually appear. Click any point to open that observation.',
    ],
    caveat: 'With tens of thousands of points, overplotting hides density — a solid mass could be 500 records or 50,000. Lower the opacity to read it.',
    also: ['chart-x', 'chart-y', 'appearance-point-opacity'],
  },
  {
    id: 'chart-bar',
    group: 'Charts',
    title: 'Bar (aggregate)',
    summary: 'One bar per category, showing a count or a mean.',
    detail: [
      'The workhorse: how many records per species, or the mean elevation per land cover.',
    ],
    also: ['chart-group-by', 'chart-measure', 'chart-horizontal', 'chart-sort'],
  },
  {
    id: 'chart-line',
    group: 'Charts',
    title: 'Line and Area',
    summary: 'A measurement averaged along an axis, optionally split into series.',
    detail: [
      'X is bucketed into equal steps and Y is averaged within each bucket, so this shows a trend rather than individual records. Area is the same chart with the region under the line filled.',
      'Use it for anything against day of year or year, where the ordering of X carries meaning.',
    ],
    also: ['chart-granularity', 'chart-series'],
  },
  {
    id: 'chart-box',
    group: 'Charts',
    title: 'Box plot by category',
    summary: 'The spread of a measurement within each category, not just its average.',
    detail: [
      'Each box spans the middle half of the values with the median marked, and the whiskers reach the rest. Two species with the same mean elevation can have completely different ranges, and that difference is what a box plot is for.',
      'Grows a row per category, so it scrolls inside its card when there are many.',
    ],
    also: ['chart-group-by', 'chart-value'],
  },
  {
    id: 'chart-histogram',
    group: 'Charts',
    title: 'Histogram',
    summary: 'The distribution of a single measurement, in bins.',
    detail: [
      'Answers "what values are common" for one field — whether elevations cluster at one band or spread evenly.',
    ],
    also: ['chart-value', 'chart-bins'],
  },
  {
    id: 'chart-heatmap',
    group: 'Charts',
    title: 'Heatmap',
    summary: 'Two categories crossed, with each cell shaded by a count or a mean.',
    detail: [
      'Good for questions of the form "which species turn up in which land cover" — the pattern of filled and empty cells is usually more informative than any single number in it.',
    ],
    also: ['chart-rows', 'chart-columns', 'chart-measure'],
  },
  {
    id: 'chart-radar',
    group: 'Charts',
    title: 'Radar',
    summary: 'Values around a circle, for comparing a category across several measures.',
    detail: [
      'Readable for a handful of categories and unreadable beyond that. Bar charts compare lengths more accurately; radar is for shape at a glance.',
    ],
    caveat: 'The area enclosed grows with the square of the values and changes with the order of the axes, so it exaggerates. Read the vertices, not the area.',
  },
  {
    id: 'chart-donut',
    group: 'Charts',
    title: 'Donut',
    summary: 'Shares of a whole, as segments of a ring.',
    detail: [
      'Works when a few categories make up most of the total. Beyond about six segments the small ones become indistinguishable — a sorted bar chart says the same thing more clearly.',
    ],
    also: ['chart-bar'],
  },

  // ── Chart builder: encodings ──────────────────────────────────────────────
  {
    id: 'chart-x',
    group: 'Charts',
    title: 'X',
    summary: 'The measurement along the horizontal axis.',
    detail: ['Only fields with values in the current data are offered, so an empty list means the filters have excluded everything carrying that field.'],
  },
  {
    id: 'chart-y',
    group: 'Charts',
    title: 'Y',
    summary: 'The measurement along the vertical axis.',
    detail: ['Axes are clamped to what the quantity can physically be, so a padded axis never invents a 370° compass aspect or a negative rainfall.'],
  },
  {
    id: 'chart-color-field',
    group: 'Charts',
    title: 'Colour',
    summary: 'The category that decides each mark\'s colour.',
    detail: ['Shares the app-wide colours, so a species keeps its colour here, on the map and in the legend.'],
    also: ['appearance-palette'],
  },
  {
    id: 'chart-shape-field',
    group: 'Charts',
    title: 'Shape',
    summary: 'The category that decides each mark\'s shape.',
    detail: ['Encoding the same field by both colour and shape makes a chart readable in black and white and to viewers with colour-vision deficiency.'],
    also: ['appearance-shapes'],
  },
  {
    id: 'chart-size-field',
    group: 'Charts',
    title: 'Size',
    summary: 'A measurement that scales each mark.',
    detail: ['A third dimension on a scatter plot. Best kept for a field with a wide range — a narrow one produces marks that all look alike.'],
  },
  {
    id: 'chart-series',
    group: 'Charts',
    title: 'Series',
    summary: 'Split a line or area chart into one line per category.',
    detail: ['Turns "mean elevation by day of year" into one line per species, which is where year-on-year or species-by-species differences become visible.'],
  },
  {
    id: 'chart-group-by',
    group: 'Charts',
    title: 'Group by',
    summary: 'The category each bar, box or segment represents.',
    detail: ['Categories with no value for the field are dropped rather than collected into an "unknown" bar.'],
    also: ['filter-min-obs'],
  },
  {
    id: 'chart-measure',
    group: 'Charts',
    title: 'Measure',
    summary: 'What each bar or cell is: a count of records, or the mean of a field.',
    detail: [
      'Count answers "how many"; a mean answers "how much, typically". They can point in opposite directions — the species with the most records is often not the one found highest.',
    ],
    caveat: 'A mean over very few records is unstable. Use the minimum-records filter to hold it to a sample size worth reporting.',
    also: ['filter-min-obs'],
  },
  {
    id: 'chart-value',
    group: 'Charts',
    title: 'Value',
    summary: 'The measurement being distributed or summarised.',
    detail: ['For a histogram it is the field being binned; for a box plot it is the field whose spread each box describes.'],
  },
  {
    id: 'chart-rows',
    group: 'Charts',
    title: 'Rows and Columns',
    summary: 'The two categories a heatmap crosses.',
    detail: ['Rows are usually the many-valued field (species) and columns the few-valued one (land cover), which keeps the table readable.'],
  },
  {
    id: 'chart-columns',
    group: 'Charts',
    title: 'Columns',
    summary: 'The category across the top of a heatmap.',
    detail: ['See Rows and Columns.'],
    also: ['chart-rows'],
  },
  {
    id: 'chart-bins',
    group: 'Charts',
    title: 'Bins',
    summary: 'How many bars a histogram divides its range into, from 4 to 30.',
    detail: [
      'Bin count changes the story: too few hides structure, too many turns it into noise. If a peak appears or vanishes as you change this, it was never solid.',
    ],
  },
  {
    id: 'chart-granularity',
    group: 'Charts',
    title: 'Granularity',
    summary: 'How many steps a line or area chart divides its X axis into.',
    detail: ['The same trade-off as bins: fewer steps give a smoother, more confident-looking line built on more records each.'],
  },
  {
    id: 'chart-horizontal',
    group: 'Charts',
    title: 'Horizontal',
    summary: 'Lay bars across instead of up.',
    detail: ['Almost always the better choice when categories have long names, since horizontal bars leave room to write them out.'],
  },
  {
    id: 'chart-today',
    group: 'Charts',
    title: 'Today line',
    summary: 'Marks the current day of the year on a day-of-year axis.',
    detail: ['A reference line for reading a seasonal chart against the date now.'],
  },
  {
    id: 'chart-sort',
    group: 'Charts',
    title: 'Sort',
    summary: 'The order categories appear in — by value or by label.',
    detail: [
      'Sorting by value ranks; sorting by label makes two charts comparable side by side, since the categories stay in the same places.',
    ],
  },

  // ── Charts page ───────────────────────────────────────────────────────────
  {
    id: 'chart-save',
    group: 'Charts',
    title: 'Save to Charts',
    summary: 'Keeps the chart you built at the top of the Charts page.',
    detail: [
      'Saved charts persist locally, and sync to your account when one is connected, so they follow you to another device.',
    ],
    also: ['chart-edit', 'chart-share'],
  },
  {
    id: 'chart-edit',
    group: 'Charts',
    title: 'Edit a saved chart',
    summary: 'Reopens a saved chart in the builder, editing it rather than a copy.',
    detail: [
      'Saving then writes back in place, keeping the chart\'s position in the row. "Save as new" is there when you want a variant, and switches to editing the copy so a second save does not overwrite the original.',
    ],
    also: ['chart-save', 'chart-share'],
  },
  {
    id: 'chart-share',
    group: 'Charts',
    title: 'Share a chart',
    summary: 'A link that opens this chart over the same filtered data.',
    detail: [
      'The link carries the chart\'s configuration alongside the filters behind it, so the recipient sees the claim and its evidence rather than an empty builder. It is short enough to fit in a QR code.',
    ],
    also: ['share-link'],
  },
  {
    id: 'chart-arrange',
    group: 'Charts',
    title: 'Arrange charts',
    summary: 'Reorder or hide the preset gallery charts.',
    detail: ['Hidden charts are listed so they can be brought back, and the layout persists.'],
  },

  // ── Analysis ──────────────────────────────────────────────────────────────
  {
    id: 'analysis-drivers',
    group: 'Analysis',
    title: 'What relates to what',
    summary: 'Which environmental variables move together, by rank correlation.',
    detail: [
      'Spearman rank correlation between every pair of fields, computed pairwise-complete so a field with gaps still contributes where it has data. Rank rather than linear, so a relationship that is consistent but not straight still registers.',
      'Values run from −1 to +1. Around zero means no monotonic relationship — which is not the same as no relationship.',
    ],
    caveat: 'Correlation here is not cause, and the confound is usually season. Elevation and temperature correlate at +0.43 across the dataset, which looks like altitude warming things; hold the season still and it flattens to about zero. High finds happen in summer, low ones in spring and autumn.',
  },
  {
    id: 'analysis-species',
    group: 'Analysis',
    title: 'Species',
    summary: 'What each species prefers, and which species turn up together.',
    detail: [
      'The fingerprint shows how far each species sits from the dataset average on each field, in standard deviations, so fields with different units can be compared on one scale.',
      '"Found together" uses lift: how much more often two species share a place than they would if they were independent. Lift above 1 means they co-occur more than chance.',
    ],
    caveat: 'Co-occurrence at this resolution mostly means "the same person walked past both", so it reflects shared habitat and shared observers together.',
  },
  {
    id: 'analysis-seasons',
    group: 'Analysis',
    title: 'Year over year',
    summary: 'How season timing and elevation shift by year, shown against recording effort.',
    detail: [
      'Effort is plotted alongside deliberately: a year with more records will show a longer apparent season for no biological reason at all, and seeing the two together is the only way to tell a real shift from a sampling one.',
    ],
    caveat: 'Median timing is reported rather than mean, because a handful of very late finds drag a mean and leave a median alone.',
  },
  {
    id: 'analysis-quality',
    group: 'Analysis',
    title: 'Data quality',
    summary: 'How much of each field is actually filled in.',
    detail: [
      'Worth reading before any of the other views. A relationship computed over the 11,000 records that carry soil moisture is a different claim from one over all 48,000, and this is where you find out which you are looking at.',
    ],
    also: ['coverage'],
  },
  {
    id: 'analysis-scope',
    group: 'Analysis',
    title: 'Scope',
    summary: 'Which observations the analysis runs over — everything, or the current filters.',
    detail: ['Narrowing the scope narrows every statistic on the page, including the sample sizes reported next to them.'],
    also: ['filter-panel'],
  },

  // ── Data and filters ──────────────────────────────────────────────────────
  {
    id: 'filter-panel',
    group: 'Data',
    title: 'Filters',
    summary: 'Restrict the dataset — every view follows the same filters.',
    detail: [
      'Filters are shared across the map, charts, analysis and table, so a filter set once holds while you move between them. They are also carried in a shared link.',
    ],
    also: ['filter-location', 'filter-time', 'filter-min-obs', 'filter-saved'],
  },
  {
    id: 'filter-location',
    group: 'Data',
    title: 'Location',
    summary: 'Country, state and county, or a radius around a point.',
    detail: [
      'The radius filter takes a latitude, longitude and distance in kilometres, and can also be set from a Plus Code. It applies as a true great-circle distance rather than a bounding box.',
    ],
  },
  {
    id: 'filter-time',
    group: 'Data',
    title: 'Time',
    summary: 'Year, month, week of the year, or an explicit date range.',
    detail: [
      'Month and week ignore the year, so they select a season across every year at once — which is usually what you want for a phenology question. From/To select an actual span.',
    ],
  },
  {
    id: 'filter-min-obs',
    group: 'Data',
    title: 'Sample size',
    summary: 'Drop species or genera with fewer than N records in the current view.',
    detail: [
      'Counted *after* the other filters, so the threshold means "enough records in what you are actually looking at", not "enough somewhere in the dataset". That distinction matters: a species with 400 records nationally may have 2 in the county you filtered to, and a mean over those 2 should not be on the chart.',
    ],
    also: ['chart-measure'],
  },
  {
    id: 'filter-precise-only',
    group: 'Data',
    title: 'Precise coordinates only',
    summary: 'Drop records whose published location is deliberately vague or poorly measured.',
    detail: [
      'iNaturalist does not always publish where an observation actually was. A record is **obscured** when the observer asks for it, or automatically when the taxon is threatened — and obscuring is not a rounding: the published point is randomised inside a roughly 0.2° cell, about 20km across. Others are simply **coarse**: an honest location with an accuracy radius wider than a kilometre.',
      'That matters more here than in most places that show iNaturalist data. Every environmental value in this app is sampled *at the point* — elevation, slope, aspect, NDVI, soil moisture, the seven-day weather lead-up. For an obscured record those readings come from wherever the randomised point happened to land, which can be the far side of a ridge, a different watershed, or 1,000m up. The observation is real; the terrain attached to it describes somewhere else.',
      'The breakdown under the checkbox counts the whole loaded dataset, not the filtered view, so the numbers that justify the filter do not move when you switch it on. Records where iNaturalist reported no accuracy at all are counted separately as "not reported" — that is missing information, not a guarantee of precision, and the filter drops them too.',
    ],
    caveat: 'This is a strong filter and on some taxa it removes most of the data — obscuring is applied to whole species, so a threatened one may have no precise records at all. Turn it on when reading terrain relationships; leave it off when counting where and when people find things, which obscured records still answer honestly at a regional scale.',
    also: ['analysis-drivers', 'analysis-quality', 'data-show-filtered'],
  },
  {
    id: 'filter-saved',
    group: 'Data',
    title: 'Saved subsets',
    summary: 'Name and store a filter set to return to it later.',
    detail: ['Stored locally and synced to your account when one is connected.'],
  },
  {
    id: 'data-species',
    group: 'Data',
    title: 'Species selection',
    summary: 'Which species files are loaded, out of the per-species store.',
    detail: [
      'Observations are kept as one file per species, so a working set can be loaded without pulling everything. Fewer species loaded means a faster app.',
    ],
  },
  {
    id: 'data-table',
    group: 'Data',
    title: 'Table',
    summary: 'The filtered observations as rows, sortable and exportable.',
    detail: ['Shows the enrichment fields alongside the record, which is the quickest way to see what a given observation actually carries.'],
  },
  {
    id: 'data-fetch',
    group: 'Data',
    title: 'Fetch new',
    summary: 'Pull fresh observations for a species from iNaturalist.',
    detail: ['Newly fetched records carry only what iNaturalist provides; the environmental fields appear after the enrichment pipeline runs over them.'],
  },

  // ── Sharing and coverage ──────────────────────────────────────────────────
  {
    id: 'share-link',
    group: 'Sharing',
    title: 'Share',
    summary: 'A link that reproduces this view — filters, colouring, overlay and position.',
    detail: [
      'Everything that made the view worth showing goes into the query string, so the link opens the same thing rather than the front page. The same link is behind the QR code, the social buttons and the email and text options.',
      'The QR code is generated in the browser rather than through an image service, so the URL — filters and all — is not handed to a third party.',
    ],
    also: ['share-embed', 'chart-share'],
  },
  {
    id: 'share-embed',
    group: 'Sharing',
    title: 'Embed',
    summary: 'An iframe snippet that renders the view without the site header.',
    detail: ['The same link with `embed=1`, which drops the app chrome so it sits cleanly inside another page.'],
    also: ['share-link'],
  },
  {
    id: 'coverage',
    group: 'Sharing',
    title: 'Coverage',
    summary: 'What the enrichment pipeline has and has not filled in.',
    detail: [
      'Field-by-field completeness, plus how coverage varies over space and time. Read it as the honest limit on everything else in the app.',
    ],
    also: ['analysis-quality'],
  },
]

const BY_ID = new Map(OPTION_DOCS.map((d) => [d.id, d]))

/** One option's entry, or null. */
export function docFor(id) {
  return BY_ID.get(id) || null
}

/** The hover text for a control: its summary, or '' when undocumented. */
export function docSummary(id) {
  return BY_ID.get(id)?.summary || ''
}

// Reference anchors are namespaced because the guide's prose headings are
// anchored too, from the same page. "Coverage" is both a section of the prose
// and an option here, and without the prefix the two fight over #coverage —
// duplicate ids, and a tooltip that lands on the wrong thing.
const ANCHOR_PREFIX = 'opt-'

/** The element id the guide gives this option. */
export function docAnchor(id) {
  return `${ANCHOR_PREFIX}${id}`
}

/** Where the guide documents this option. */
export function docHref(id) {
  return `/guide#${docAnchor(id)}`
}

/** Entries grouped in declaration order, for rendering the reference. */
export function docGroups() {
  const groups = []
  const seen = new Map()
  for (const doc of OPTION_DOCS) {
    let group = seen.get(doc.group)
    if (!group) {
      group = { name: doc.group, items: [] }
      seen.set(doc.group, group)
      groups.push(group)
    }
    group.items.push(doc)
  }
  return groups
}
