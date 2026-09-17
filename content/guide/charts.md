# Charts

Each chart reads the observations that the filters select. Change a filter and
each chart changes.

## Gallery

The gallery holds nineteen preset charts. You can move a card, hide it, open it
full screen, or save it as a PNG. Hold `Shift` and click the save button to get
an SVG. An SVG stays sharp at any size.

Each card has a fixed height, so the grid stays regular. A chart that is taller
than its card is cut, with a fade at the edge. Full screen shows it complete.

## Build

The chart builder composes your own chart. The types are scatter, bar, line,
area, box plot, histogram, heatmap, radar and donut.

1. Select the chart type.
2. Select a field for each axis, and for the color, the shape, the size and the
   series.
3. Select **Save to Charts** to keep the chart.

**Sort** orders a grouped chart by value or by label. Largest-first answers
"which is biggest". A–Z answers "where is X". The app applies the size cap
before the sort, so a change to A–Z never pushes the biggest categories off the
chart.

## Style

The **Style** panel sets how the app draws everything. Your settings persist,
and they follow your account when you sign in.

### Colors for categories

Five palettes are available, and one of them is safe for color-blind readers.

**Shuffle colors** deals the same palette in a different order. Use it when two
species land on shades that you cannot separate. The shuffle is deterministic,
so a shuffled view looks the same to each person who opens your shared link.

**Per-value overrides** pin one value to one color. Pin a species to a color and
it holds on the map and in each chart.

### Colors for measurements

Two gradients are set here, and both use the same editor:

- **Point gradient** colors the points when you color them by a measurement.
  Read [Point colors](/guide/map#the-point-gradient).
- **Heatmap** colors the grid cells.

Each gradient holds between 2 and 8 stops. Add a stop with the **+** button
between two swatches. Remove one with the **×** button on a swatch. **Reverse**
turns the gradient end to end.

### Sizes and opacity

The panel also sets the point size, the point opacity, the point outline, the
heatmap opacity, the cell shape that the grid bins into, and the opacity of the
reference layers.
