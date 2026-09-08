/* Split the map GeoJSON into spatial chunks the browser fetches as it pans.
 *
 * The map was fetching all 48,233 observations before it could draw one of
 * them, 49.7 MB of them, and on a slow connection it simply never appeared.
 * Two things are wrong with that and this addresses both.
 *
 * **It fetched everything to show anything.** So the build also writes one file
 * per half-degree cell. The map loads the cells its viewport touches, and a pan
 * into new ground loads only the new ground.
 *
 * **Every observation carried 40 enrichment fields the map does not draw.** A
 * feature averages 1,030 bytes, of which the map needs the position, the name,
 * the date, and whatever is currently being coloured or sized by. The rest is
 * the drawer's, and the drawer opens one record at a time. So there are two
 * shapes:
 *
 *   overview.json  every observation, slim fields, thinned. Paints immediately.
 *   <cell>.json    everything in one cell, full fields. Fetched on demand.
 *
 * Neither alone is enough: without the overview a zoomed-out map still has to
 * fetch the world, and without the cells a click has nothing to open.
 *
 * This runs at build time (npm prebuild) rather than in the Python pipeline, so
 * the chunks are always derived from whatever observations.geojson is currently
 * committed, and 46 MB of derivable files stay out of git.
 */

import fs from 'node:fs'
import path from 'node:path'

// Half a degree is about 55km of latitude. Smaller means more requests per pan;
// larger means the busiest cell is too big to be worth splitting. At this size
// the heaviest cell in the shipped dataset holds ~5,100 observations and 40
// cells hold 90% of everything.
export const CELL_SIZE = 0.5

// What the map itself draws with. Everything else belongs to the drawer, which
// opens one record at a time out of that record's own cell file.
export const OVERVIEW_FIELDS = [
  'species', 'genus', 'date', 'day_of_year', 'cluster',
  'elevation', 'land_cover_label', 'inat_id',
]

// The overview exists to paint fast, so it has a budget.
export const OVERVIEW_MAX = 12000

/** The cell a coordinate falls in. Must match cellKeyFor in useMapChunks.js. */
export function cellKey(lat, lon, size = CELL_SIZE) {
  return `${Math.floor(lat / size)}_${Math.floor(lon / size)}`
}

function coordsOf(feature) {
  const co = feature?.geometry?.coordinates
  if (!co || co.length < 2) return null
  const lon = Number(co[0]); const lat = Number(co[1])
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null
  return [lat, lon]
}

/** Group features by cell. */
export function splitFeatures(features, size = CELL_SIZE) {
  const cells = new Map()
  for (const f of features) {
    const pos = coordsOf(f)
    if (!pos) continue
    const key = cellKey(pos[0], pos[1], size)
    if (!cells.has(key)) cells.set(key, [])
    cells.get(key).push(f)
  }
  return cells
}

/** A feature carrying only the fields the map draws with. */
export function slim(feature, fields = OVERVIEW_FIELDS) {
  const props = feature?.properties || {}
  const out = {}
  for (const k of fields) {
    const v = props[k]
    if (v !== null && v !== undefined && v !== '') out[k] = v
  }
  return { type: 'Feature', geometry: feature.geometry, properties: out }
}

/**
 * Every observation, slimmed, thinned to `limit` by taking every Nth.
 *
 * By stride rather than by region: taking the first N would draw one area in
 * full and leave the rest empty, which is a worse lie than drawing everywhere
 * at a fifth of the density.
 */
export function buildOverview(features, { limit = OVERVIEW_MAX, fields = OVERVIEW_FIELDS } = {}) {
  const usable = features.filter((f) => coordsOf(f))
  const stride = limit ? Math.max(1, Math.ceil(usable.length / limit)) : 1
  const out = []
  for (let i = 0; i < usable.length; i += stride) out.push(slim(usable[i], fields))
  return { features: out, stride }
}

/** Write the cell files, the overview and the index. Returns the index. */
export function writeChunks(features, { dataDir = path.join('public', 'data'), size = CELL_SIZE } = {}) {
  const outDir = path.join(dataDir, 'map')
  fs.rmSync(outDir, { recursive: true, force: true })
  fs.mkdirSync(outDir, { recursive: true })

  const cells = splitFeatures(features, size)
  const index = {}
  for (const [key, rows] of [...cells.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    const file = path.join(outDir, `${key}.json`)
    fs.writeFileSync(file, JSON.stringify({ type: 'FeatureCollection', features: rows }))
    index[key] = { n: rows.length, bytes: fs.statSync(file).size }
  }

  const overview = buildOverview(features)
  const overviewPath = path.join(outDir, 'overview.json')
  fs.writeFileSync(overviewPath,
    JSON.stringify({ type: 'FeatureCollection', features: overview.features }))

  const manifest = {
    cellSize: size,
    total: Object.values(index).reduce((a, c) => a + c.n, 0),
    overview: {
      n: overview.features.length,
      stride: overview.stride,
      bytes: fs.statSync(overviewPath).size,
      complete: overview.stride === 1,
    },
    cells: index,
  }
  fs.writeFileSync(path.join(outDir, 'index.json'), JSON.stringify(manifest))
  return manifest
}

export function rebuild(dataDir = path.join('public', 'data')) {
  const source = path.join(dataDir, 'observations.geojson')
  if (!fs.existsSync(source)) {
    console.log(`[map chunks] ${source} not found, nothing to split.`)
    return null
  }
  const { features = [] } = JSON.parse(fs.readFileSync(source, 'utf8'))
  const manifest = writeChunks(features, { dataDir })
  const ov = manifest.overview
  console.log(`[map chunks] ${Object.keys(manifest.cells).length} cells, `
    + `${manifest.total.toLocaleString()} observations; overview `
    + `${ov.n.toLocaleString()} features, ${(ov.bytes / 1e6).toFixed(1)} MB, `
    + `${ov.complete ? 'complete' : `every ${ov.stride}th`}`)
  return manifest
}

if (import.meta.url === `file://${process.argv[1]}`) rebuild()
