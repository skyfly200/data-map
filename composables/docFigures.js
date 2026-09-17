// Line drawings for the guide.
//
// The guide explains a stack of things that are easier to see than to read: the
// order layers draw in, where a tile URL comes from, why a hexagon grid is not a
// square one. Each of those costs a paragraph in words and a glance as a
// picture, so the picture goes first and the paragraph says the part a picture
// cannot.
//
// They are SVG source rather than image files for one reason: the app has a
// dark theme and a light one, and a drawing in either is wrong in the other.
// Inline SVG can read the same CSS colour tokens as the text around it, so one
// drawing is correct in both. Every shape therefore carries a class, and
// MarkdownDoc gives that class its colour — no fill or stroke is written here.
//
// Marker ids are prefixed per figure. Several figures render into one page, and
// two <marker id="arrow"> in one document is one arrowhead, used by whichever
// figure the browser saw first.

/** An arrowhead definition, named uniquely so figures do not share one. */
const head = (id) => `<defs><marker id="${id}" viewBox="0 0 8 8" refX="7" refY="4"
  markerWidth="6" markerHeight="6" orient="auto-start-reverse">
  <path class="arw-head" d="M0 0 L8 4 L0 8 z" /></marker></defs>`

/** A labelled box. `sub` is the second, quieter line. */
const box = (x, y, w, h, label, sub = '', cls = 'box') => `
  <rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="6" />
  <text class="lbl" x="${x + w / 2}" y="${sub ? y + h / 2 - 2 : y + h / 2 + 4}"
        text-anchor="middle">${label}</text>
  ${sub ? `<text class="sub" x="${x + w / 2}" y="${y + h / 2 + 13}" text-anchor="middle">${sub}</text>` : ''}`

const arrow = (id, x1, y1, x2, y2, cls = 'arw') =>
  `<path class="${cls}" d="M${x1} ${y1} L${x2} ${y2}" marker-end="url(#${id})" />`

export const FIGURES = {
  // Where a value on the map came from, left to right.
  'data-flow': `<svg viewBox="0 0 700 130" role="img" aria-label="Observations and
  raster layers go through a pipeline job into a dataset, which the map, charts
  and analysis all read">
  ${head('ah-flow')}
  ${box(6, 10, 118, 42, 'iNaturalist', 'or your own points')}
  ${box(6, 72, 118, 42, 'Earth Engine', 'raster layers')}
  ${box(190, 41, 118, 42, 'Pipeline job', 'samples each layer')}
  ${box(374, 41, 118, 42, 'Dataset', 'one row per point', 'box accent')}
  ${box(558, 41, 136, 42, 'Map · Charts', 'and Analysis')}
  ${arrow('ah-flow', 128, 31, 186, 55)}
  ${arrow('ah-flow', 128, 93, 186, 69)}
  ${arrow('ah-flow', 312, 62, 370, 62)}
  ${arrow('ah-flow', 496, 62, 554, 62)}
</svg>`,

  // What is drawn on top of what.
  'map-stack': `<svg viewBox="0 0 520 210" role="img" aria-label="The map draws
  the basemap first, then reference layers, then the heatmap, then the points">
  <g class="plane"><path class="box" d="M60 30 L300 30 L360 60 L120 60 z" />
    <text class="lbl" x="380" y="49">Points</text>
    <text class="sub" x="380" y="62">one dot per record</text></g>
  <g class="plane"><path class="box" d="M60 80 L300 80 L360 110 L120 110 z" />
    <text class="lbl" x="380" y="99">Heatmap</text>
    <text class="sub" x="380" y="112">a grid of summaries</text></g>
  <g class="plane"><path class="box" d="M60 130 L300 130 L360 160 L120 160 z" />
    <text class="lbl" x="380" y="149">Layers</text>
    <text class="sub" x="380" y="162">stacked, topmost first</text></g>
  <g class="plane"><path class="box" d="M60 180 L300 180 L360 205 L120 205 z" />
    <text class="lbl" x="380" y="196">Basemap</text>
    <text class="sub" x="380" y="209">one choice</text></g>
  <text class="sub" x="14" y="24">top</text>
  <text class="sub" x="14" y="200">bottom</text>
</svg>`,

  // Why the grid uses hexagons.
  'hex-vs-square': `<svg viewBox="0 0 520 190" role="img" aria-label="A hexagon
  has six neighbours at one distance; a square has four edge neighbours and four
  diagonal neighbours that are further away">
  <g class="cells">
    <path class="cell fill" d="M92 60 L122 77 L122 111 L92 128 L62 111 L62 77 z" />
    <path class="cell" d="M92 0 L122 17 L122 51 L92 68 L62 51 L62 17 z" />
    <path class="cell" d="M92 120 L122 137 L122 171 L92 188 L62 171 L62 137 z" />
    <path class="cell" d="M144 30 L174 47 L174 81 L144 98 L114 81 L114 47 z" />
    <path class="cell" d="M144 90 L174 107 L174 141 L144 158 L114 141 L114 107 z" />
    <path class="cell" d="M40 30 L70 47 L70 81 L40 98 L10 81 L10 47 z" />
    <path class="cell" d="M40 90 L70 107 L70 141 L40 158 L10 141 L10 107 z" />
  </g>
  <text class="sub" x="92" y="182" text-anchor="middle">6 neighbours, all the same distance</text>

  <g class="cells">
    <rect class="cell fill" x="340" y="64" width="46" height="46" />
    <rect class="cell" x="294" y="64" width="46" height="46" />
    <rect class="cell" x="386" y="64" width="46" height="46" />
    <rect class="cell" x="340" y="18" width="46" height="46" />
    <rect class="cell" x="340" y="110" width="46" height="46" />
    <rect class="cell dim" x="294" y="18" width="46" height="46" />
    <rect class="cell dim" x="386" y="18" width="46" height="46" />
    <rect class="cell dim" x="294" y="110" width="46" height="46" />
    <rect class="cell dim" x="386" y="110" width="46" height="46" />
  </g>
  <text class="sub" x="363" y="182" text-anchor="middle">4 near, 4 corners at 1.41×</text>
</svg>`,

  // The asset stays; the URL does not.
  'asset-and-token': `<svg viewBox="0 0 700 120" role="img" aria-label="A
  permanent Earth Engine asset, a server function that mints a tile URL, and a
  tile URL that expires within hours">
  ${head('ah-token')}
  ${box(6, 36, 168, 48, 'Earth Engine asset', 'permanent', 'box accent')}
  ${box(266, 36, 168, 48, 'Tile function', 'signs in, calls getMapId')}
  ${box(526, 36, 168, 48, 'Tile URL', 'expires in hours')}
  ${arrow('ah-token', 178, 60, 262, 60)}
  ${arrow('ah-token', 438, 60, 522, 60)}
  <text class="sub" x="350" y="20" text-anchor="middle">once, by you</text>
  <text class="sub" x="610" y="20" text-anchor="middle">again about every hour</text>
  <path class="arw dash" d="M610 90 C610 116, 90 116, 90 92" marker-end="url(#ah-token)" />
</svg>`,

  // Two keys, two questions.
  'two-keys': `<svg viewBox="0 0 560 160" role="img" aria-label="The Points key
  explains the dots; the Map layers key explains the tiles under them">
  <rect class="box" x="6" y="8" width="256" height="144" rx="6" />
  <text class="lbl" x="22" y="30">Points</text>
  <text class="sub" x="22" y="44">what the dots mean</text>
  <circle class="dot d1" cx="28" cy="66" r="6" /><text class="sub" x="44" y="70">Amanita muscaria</text>
  <circle class="dot d2" cx="28" cy="90" r="6" /><text class="sub" x="44" y="94">Suillus brevipes</text>
  <circle class="dot d3" cx="28" cy="114" r="6" /><text class="sub" x="44" y="118">Boletus edulis</text>

  <rect class="box" x="298" y="8" width="256" height="144" rx="6" />
  <text class="lbl" x="314" y="30">Map layers</text>
  <text class="sub" x="314" y="44">what the tiles mean</text>
  <defs><linearGradient id="tk-ramp" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" class="rs0" /><stop offset="0.5" class="rs1" /><stop offset="1" class="rs2" />
  </linearGradient></defs>
  <rect class="ramp" x="314" y="60" width="224" height="14" rx="3" fill="url(#tk-ramp)" />
  <text class="sub" x="314" y="90">0°</text>
  <text class="sub" x="538" y="90" text-anchor="end">45°</text>
  <text class="sub" x="314" y="118">Slope · SRTM · 30 m</text>
</svg>`,

  // A ramp is a list of stops.
  'ramp-stops': `<svg viewBox="0 0 520 100" role="img" aria-label="A colour ramp
  with four evenly spaced stops, low value on the left and high on the right">
  <defs><linearGradient id="rs-ramp" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" class="rs0" /><stop offset="0.33" class="rs1" />
    <stop offset="0.67" class="rs2" /><stop offset="1" class="rs3" />
  </linearGradient></defs>
  <rect class="ramp" x="20" y="20" width="480" height="26" rx="4" fill="url(#rs-ramp)" />
  <g class="stops">
    <rect class="stop" x="13" y="52" width="14" height="14" rx="3" />
    <rect class="stop" x="171" y="52" width="14" height="14" rx="3" />
    <rect class="stop" x="329" y="52" width="14" height="14" rx="3" />
    <rect class="stop" x="487" y="52" width="14" height="14" rx="3" />
  </g>
  <text class="sub" x="20" y="82">low</text>
  <text class="sub" x="500" y="82" text-anchor="end">high</text>
  <text class="sub" x="260" y="82" text-anchor="middle">stops are evenly spaced</text>
</svg>`,

  // One job's result is the next job's input.
  'job-chain': `<svg viewBox="0 0 700 150" role="img" aria-label="A job writes a
  result, you save the result as a dataset, and a later job reads that dataset as
  its points">
  ${head('ah-chain')}
  ${box(6, 20, 150, 46, 'Job 1', 'samples 6 layers')}
  ${box(246, 20, 170, 46, 'Saved dataset', 'private to you', 'box accent')}
  ${box(506, 20, 188, 46, 'Job 2', 'samples 4 more layers')}
  ${arrow('ah-chain', 160, 43, 242, 43)}
  ${arrow('ah-chain', 420, 43, 502, 43)}
  <text class="sub" x="203" y="14" text-anchor="middle">Save result</text>
  <text class="sub" x="461" y="14" text-anchor="middle">Use as source</text>
  <path class="arw dash" d="M600 70 C600 122, 331 128, 331 72" marker-end="url(#ah-chain)" />
  <text class="sub" x="350" y="145" text-anchor="middle">save the wider result again, and repeat</text>
</svg>`,
}
