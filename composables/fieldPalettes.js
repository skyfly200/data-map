// Colouring a point the way the layer under it is coloured.
//
// The map draws the same quantity twice: as a raster from Earth Engine, and as
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
//   domain applies, so a dot over a pixel is genuinely the same colour for the
//   same value.
//
//   Different units — the enrichment pipeline rank-normalises solar exposure,
//   wind exposure and wetness to 0..1 across the sampled points, while the
//   layers render raw heat load, raw TPI in metres and raw TWI. The shades are
//   comparable in rank and not in value, and pretending otherwise by stretching
//   one onto the other would put a confident number on a coincidence.
//
// `comparable` is that distinction, declared per field so the legend can say
// which kind of match the viewer is looking at.

import { EE_TILE_LAYERS, WORLDCOVER_CLASSES } from '../netlify/lib/ee-tile-layers.mjs'

/**
 * Observation field → the layer that draws the same thing.
 *
 * `domain` is the layer's own visualisation range, and is present only where
 * the two sides measure in the same units. It is checked against what the
 * layer's build() actually asks Earth Engine for, so a layer whose stretch
 * changes cannot leave this table quietly wrong.
 */
export const FIELD_LAYERS = {
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
 * means and changing a layer's palette moves the points with it.
 */
export function paletteFor(field) {
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

/** Clamped hex interpolation between two colours. */
export function mix(a, b, t) {
  const k = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0))
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  return `#${pa.map((v, i) => Math.round(v + (pb[i] - v) * k).toString(16).padStart(2, '0')).join('')}`
}

/**
 * A colour from anywhere along a multi-stop ramp.
 *
 * The layers declare ramps of six to ten stops, which is what gives them their
 * shape — a two-colour lerp between the ends would keep the extremes and throw
 * away everything that makes the middle readable.
 */
export function rampColor(stops, t) {
  if (!Array.isArray(stops) || !stops.length) return '#888888'
  if (stops.length === 1) return stops[0]
  const k = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0))
  const scaled = k * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(scaled))
  return mix(stops[i], stops[i + 1], scaled - i)
}

/** Where a value sits in a domain, 0..1. */
export function fraction(value, [lo, hi]) {
  const span = hi - lo
  if (!Number.isFinite(span) || span === 0) return 0
  return (Number(value) - lo) / span
}

/**
 * The colour a class palette gives a label.
 *
 * Matched on the label and on the aliases beside it, because the two sides
 * spell some classes differently: WorldCover's own name for code 60 is "Bare /
 * sparse" and the enrichment pipeline writes "Bare / sparse vegetation". The
 * join that actually holds is the class code, which is what the aliases stand
 * in for on data that carries the label and not the code.
 */
export function classColorFor(items, label) {
  if (!Array.isArray(items) || label === null || label === undefined) return null
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
 * under it deserves to know whether the shades mean the same number or only the
 * same ranking.
 */
export function matchNote(palette) {
  if (!palette) return ''
  if (palette.comparable) return `Coloured to match the ${palette.layerName} layer.`
  return `Palette matches the ${palette.layerName} layer. Values here are normalised, `
    + 'so shades compare with each other rather than with the layer.'
}
