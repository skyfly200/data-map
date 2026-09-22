// Mapping between observation fields and the palettes used to colour them.
//
// The map draws the same quantity twice: as a raster from Earth Engine, and asT
// a dot for each observation. Those had unrelated palettes, so a find on wet
// ground could be a dark dot over a pale pixel and the two views of one number
// argued with each other. Reading a point against its layer is the main thing
// this map is for, and it only works if green means the same in both.
//
// What a field takes from its layer is the PALETTE. Whether it can also take
// the layer's DOMAIN is a separate question, and the answer is no more often
// than it looks:
//
//   Same units — slope is degrees on both sides, NDVI is NDVI. The layer's
// domain applies, so a dot over a pixel is genuinely the same colour for the
// same value.
//
//   Different units — the enrichment pipeline rank-normalises solar exposure,
// wind exposure and wetness to 0..1 across the sampled points, while the
// layers render raw heat load, raw TPI in metres and raw TWI. The shades are
// comparable in rank and not in value, and pretending otherwise by stretching
// one onto the other would put a confident number on a coincidence.
//
// `comparable` is that distinction, declared per field so the legend can say
// which kind of match the viewer is looking at.

import { EE_TILE_LAYERS } from '../netlify/lib/ee-tile-layers.mjs'
import { mix, rampColor } from './ramps.js'

// Re-exported so a caller that colours by field does not have to know which
// module the interpolation lives in.
export { mix, rampColor }

/**
 * Observation field → the layer that draws the same thing.
 *
 * `domain` is the layer's own visualisation range, and is present only where
// the two sides measure in the same units. It is checked against what the
// layer's build() actually asks Earth Engine for, so a layer whose stretch
// changes cannot leave this table quietly wrong.
 */
export const FIELD_LAYERS: Record<string, { layer: string, comparable: boolean, domain?: number[] }> = {
  land_cover_label: { layer: 'land-cover', comparable: true },

  ndvi: { layer: 'ndvi-recent', comparable: true, domain: [-0.2, 0.9] },
  ndmi: { layer: 'ndmi-recent', comparable: true, domain: [-0.2, 0.9] },
  slope: { layer: 'slope', comparable: true, domain: [0, 45] },
  aspect: { layer: 'aspect', comparable: true, domain: [0, 360] },

  // Normalised on the observation side, raw on the layer's. Palette only.
  solar_exposure: { layer: 'solar-exposure', comparable: false },
  wind_exposure: { layer: 'wind-exposure', comparable: false },
  water_retention: { layer: 'twi', comparable: false },
}

/** Fields that have a layer to match. */
export const MATCHED_FIELDS = Object.keys(FIELD_LAYERS)

/**
 * The palette a field should be drawn with, or null when nothing matches.
 *
 * Returns the layer's own legend, so there is one definition of what a colour
// means and changing a layer's palette moves the points with it.
 */
export function paletteFor(field: string): any {
  const match = FIELD_LAYERS[field]
  if (!match) return null
  const layer = EE_TILE_LAYERS[match.layer]
  if (!layer?.legend) return null

  const { legend } = layer
  if (legend.type === 'classes') {
    return {
      kind: 'classes',
      items: legend.items,
      layer: match.layer,
      layerName: layer.name,
      comparable: match.comparable,
    }
  }
  return {
    kind: 'ramp',
    stops: legend.stops,
    // Null means "scale over whatever the data does", which is the honest
// treatment when the units differ.
    domain: match.comparable ? match.domain || null : null,
    layer: match.layer,
    layerName: layer.name,
    comparable: match.comparable,
  }
}

/** Where a value sits in a domain, 0..1. */
export function fraction(value: any, domain: [number, number]): number {
  const span = domain[1] - domain[0]
  if (!Number.isFinite(span) || span === 0) return 0
  return (Number(value) - domain[0]) / span
}

/**
 * The colour a class palette gives a label.
 *
 * Matched on the label and on the aliases beside it, because the two sides
// spell some classes differently, and the join that actually holds is the
// class code, which is also what the aliases stand in for.
 */
export function classColorFor(items: any[], label: string): string | null {
  if (!items) return null
  const want = String(label).trim().toLowerCase()
  for (const item of items) {
    if (String(item.label).trim().toLowerCase() === want) return item.color
    for (const alias of item.aliases || []) {
      if (String(alias).trim().toLowerCase() === want) return item.color
    }
  }
  return null
}

/**
 * How the legend should describe the match.
 *
 * Said rather than left to be inferred: a viewer comparing a dot to the ground
// under it deserves to know whether the shades mean the same number or only
// the same ranking.
 */
export function matchNote(palette: any): string {
  if (!palette) return ''
  if (palette.comparable) return `Coloured to match the ${palette.layerName} layer.`
  return `Palette matches the ${palette.layerName} layer. Values here are normalised, `
    + 'so shades compare with each other rather than with the layer.'
}
