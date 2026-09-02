// Grid geometry for the map's heatmaps: which cell a point falls in, and the
// polygon to draw for that cell.
//
// Pure module — no framework imports — so it can be unit-tested directly.
//
// Everything works in raw (lon, lat) degrees, treating them as a plane. That is
// the same assumption the square grid has always made, and it is what keeps the
// cell-size labels ("~5 km") meaning the same thing for both shapes. It does
// mean a cell renders slightly taller than wide on a Mercator map, by 1/cos(lat)
// — unavoidable without giving each latitude band its own grid.

export const CELL_SHAPES = [
  { value: 'hex', label: 'Hexagons' },
  { value: 'square', label: 'Squares' },
]

// A hexagon of circumradius r has area (3√3/2)r². Sizing the hex to the same
// area as a size×size square keeps the two shapes interchangeable: switching
// shape re-bins the same observations at the same resolution rather than
// silently coarsening or sharpening the map.
const HEX_AREA = (3 * Math.sqrt(3)) / 2
export const HEX_RADIUS_FACTOR = 1 / Math.sqrt(HEX_AREA)   // ≈ 0.6204

/** Circumradius, in degrees, of the hex matching a square cell of `size`. */
export function hexRadius(size) { return size * HEX_RADIUS_FACTOR }

/**
 * Which hex a point falls in, as integer axial-ish coordinates.
 *
 * Pointy-top hexes on a staggered grid: columns are √3·r apart, rows 1.5·r, and
 * odd rows are offset half a column. Rounding to the nearest row/column lands in
 * the right hex over most of its area but not near the slanted edges, so the
 * candidate is compared against its diagonal neighbour and the closer centre
 * wins. (The naive rounding alone produces rectangles with zig-zag edges, not
 * hexagons.) Assigning each point to its nearest centre is exactly the hexagonal
 * partition — the Voronoi cells of a triangular lattice are hexagons.
 *
 * The comparison is in real distance, not in row/column units. d3-hexbin
 * compares px²+py² with px measured in columns and py in rows, but a column
 * (√3·r) and a row (1.5·r) are not the same length, so that ranking is stretched
 * and mis-assigns a thin band along each slanted edge to the neighbour across
 * it. Scaling the row term by dy/dx before squaring restores a true circle.
 */
const ROW_TO_COL = 1.5 / Math.sqrt(3)

export function hexIndex(lon, lat, size) {
  const r = hexRadius(size)
  const dx = r * Math.sqrt(3)
  const dy = r * 1.5

  const py = lat / dy
  let pj = Math.round(py)
  const px = lon / dx - (pj & 1 ? 0.5 : 0)
  let pi = Math.round(px)

  const py1 = (py - pj) * ROW_TO_COL
  if (Math.abs(py - pj) * 3 > 1) {
    const px1 = px - pi
    const pi2 = pi + (px < pi ? -1 : 1) / 2
    const pj2 = pj + (py < pj ? -1 : 1)
    const px2 = px - pi2
    const py2 = (py - pj2) * ROW_TO_COL
    if (px1 * px1 + py1 * py1 > px2 * px2 + py2 * py2) {
      pi = pi2 + (pj & 1 ? 1 : -1) / 2
      pj = pj2
    }
  }
  return [pi, pj]
}

/** Centre of the hex with the given index, in [lon, lat]. */
export function hexCentre(pi, pj, size) {
  const r = hexRadius(size)
  return [(pi + (pj & 1 ? 0.5 : 0)) * r * Math.sqrt(3), pj * r * 1.5]
}

/** The six corners of a hex, as Leaflet [lat, lon] pairs, starting at the top. */
export function hexPolygon(lon, lat, size) {
  const r = hexRadius(size)
  const pts = []
  for (let k = 0; k < 6; k += 1) {
    const a = ((90 + k * 60) * Math.PI) / 180
    pts.push([lat + r * Math.sin(a), lon + r * Math.cos(a)])
  }
  return pts
}

/**
 * The cell a coordinate falls in.
 *
 * Returns `{ key, lat, lon, lat0, lon0, lat1, lon1, polygon }` — the key to bin
 * on, the centre, the bounding box (which the arrow overlay sizes itself
 * against) and the outline to draw. Both shapes return all of it, so nothing
 * downstream has to branch on which grid is in use.
 */
export function cellAt(lon, lat, size, shape = 'hex') {
  if (shape === 'hex') {
    const [pi, pj] = hexIndex(lon, lat, size)
    const [cx, cy] = hexCentre(pi, pj, size)
    const r = hexRadius(size)
    const hw = (r * Math.sqrt(3)) / 2
    return {
      key: `h${pi}:${pj}`,
      lon: cx, lat: cy,
      lat0: cy - r, lat1: cy + r, lon0: cx - hw, lon1: cx + hw,
      polygon: hexPolygon(cx, cy, size),
    }
  }
  const gy = Math.floor(lat / size)
  const gx = Math.floor(lon / size)
  const lat0 = gy * size, lon0 = gx * size
  const lat1 = (gy + 1) * size, lon1 = (gx + 1) * size
  return {
    key: `${gy}:${gx}`,
    lon: (lon0 + lon1) / 2, lat: (lat0 + lat1) / 2,
    lat0, lon0, lat1, lon1,
    polygon: [[lat0, lon0], [lat0, lon1], [lat1, lon1], [lat1, lon0]],
  }
}

/** Just the bin key, for looking a coordinate up in an already-built grid. */
export function cellKeyAt(lon, lat, size, shape = 'hex') {
  return cellAt(lon, lat, size, shape).key
}
