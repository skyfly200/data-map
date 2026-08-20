// Sample the precomputed terrain-exposure rasters (from terrain_pipeline.py)
// at a point, in Node, via geotiff.js. Optional: if the rasters are not bundled
// with the function, sampling is skipped and the fields stay null (the full
// Python pipeline fills them in on its next run).
//
// The rasters are EPSG:4326 GeoTIFFs, so lon/lat map directly onto the grid via
// the bounding box and dimensions.

import { fromFile } from 'geotiff'
import { existsSync } from 'node:fs'
import { join } from 'node:path'

const LAYERS = ['slope', 'aspect', 'solar_exposure', 'wind_exposure', 'water_retention']
const DECIMALS = { slope: 1, aspect: 1 }  // exposures default to 4 dp

export async function openTerrain(dir) {
  const readers = {}
  if (!dir) return readers
  for (const name of LAYERS) {
    const path = join(dir, `${name}.tif`)
    if (!existsSync(path)) continue
    const tiff = await fromFile(path)
    const image = await tiff.getImage()
    readers[name] = {
      image,
      bbox: image.getBoundingBox(), // [minX, minY, maxX, maxY]
      width: image.getWidth(),
      height: image.getHeight(),
    }
  }
  return readers
}

export async function sampleAt(reader, lon, lat) {
  const [minX, minY, maxX, maxY] = reader.bbox
  if (lon < minX || lon > maxX || lat < minY || lat > maxY) return null
  const col = clamp(Math.floor(((lon - minX) / (maxX - minX)) * reader.width), 0, reader.width - 1)
  const row = clamp(Math.floor(((maxY - lat) / (maxY - minY)) * reader.height), 0, reader.height - 1)
  const rasters = await reader.image.readRasters({ window: [col, row, col + 1, row + 1] })
  const value = rasters?.[0]?.[0]
  if (value === undefined || value === null || Number.isNaN(value)) return null
  return value
}

export async function enrichFeatureTerrain(feature, readers) {
  const [lon, lat] = feature.geometry.coordinates
  for (const [name, reader] of Object.entries(readers)) {
    const value = await sampleAt(reader, lon, lat)
    if (value !== null) {
      const dp = DECIMALS[name] ?? 4
      feature.properties[name] = Number(value.toFixed(dp))
    }
  }
  return feature
}

function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)) }
