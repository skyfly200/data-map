// Layers an administrator registers, pointing at their own Earth Engine assets.
//
// The built-in catalogue in ee-tile-layers.mjs is a fixed set of recipes. This
// is the open half: compute whatever you like in the Earth Engine Code Editor,
// `Export.image.toAsset()` it into your project, and register the asset here.
// The app then renders it through exactly the same path as the built-ins —
// getMapId, the shared cache, the same error reporting.
//
// WHY AN ASSET ID AND NOT A SCRIPT. The rule that runs through the rest of this
// codebase is that Earth Engine code never arrives over the wire. An asset id is
// a REFERENCE, not an expression: it names something already computed and stored
// under an account we control. Accepting a script would be arbitrary compute on
// the FRMS billing account, and a code-injection surface, however trusted the
// person pasting it is meant to be. Exporting to an asset first also makes the
// layer cheap to draw, because the heavy computation happened once.
//
// Everything here is pure. The validation is the security boundary — an asset id
// is interpolated into an Earth Engine call — so it is checked against a strict
// shape and tested rather than trusted.

export class CustomLayerError extends Error {}

/** Reducers offered for an ImageCollection, and what each is for. */
export const REDUCERS = {
  mosaic: 'Most recent pixel (mosaic)',
  mean: 'Mean',
  median: 'Median',
  max: 'Maximum',
  min: 'Minimum',
  first: 'First',
}

export const ASSET_TYPES = {
  image: 'Single image',
  image_collection: 'Image collection',
}

/**
 * An Earth Engine asset id.
 *
 * Three shapes are legal: a Cloud project asset
 * (projects/<project>/assets/<path>), a legacy user asset (users/<name>/<path>),
 * and a public catalogue id (MODIS/061/MCD64A1). All three are slash-separated
 * segments of letters, digits, underscore, dash and dot.
 *
 * The pattern is the point. This string goes straight into ee.Image(...), so it
 * must not be able to carry anything but a name — no spaces, no quotes, no
 * parentheses, and no `..` that could climb a path.
 */
const SEGMENT = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/
const MAX_SEGMENTS = 24
const MAX_SEGMENT_LEN = 128

export function validateAssetId(raw) {
  const id = String(raw ?? '').trim()
  if (!id) throw new CustomLayerError('An Earth Engine asset ID is required.')
  if (id.length > 512) throw new CustomLayerError('That asset ID is too long.')
  if (id.startsWith('/') || id.endsWith('/')) {
    throw new CustomLayerError('An asset ID does not start or end with a slash.')
  }
  const parts = id.split('/')
  if (parts.length > MAX_SEGMENTS) throw new CustomLayerError('That asset ID has too many parts.')
  for (const part of parts) {
    if (!part) throw new CustomLayerError('An asset ID cannot contain an empty part.')
    if (part === '.' || part === '..') throw new CustomLayerError('An asset ID cannot contain . or ..')
    if (part.length > MAX_SEGMENT_LEN) throw new CustomLayerError('One part of that asset ID is too long.')
    if (!SEGMENT.test(part)) {
      throw new CustomLayerError(
        `“${part}” is not a valid part of an asset ID. Use letters, digits, dot, dash and underscore.`,
      )
    }
  }
  return id
}

/** A band name, which is also interpolated into an Earth Engine call. */
export function validateBand(raw, { required = false } = {}) {
  const band = String(raw ?? '').trim()
  if (!band) {
    if (required) throw new CustomLayerError('A band name is required.')
    return ''
  }
  if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$/.test(band)) {
    throw new CustomLayerError('A band name uses letters, digits, dot, dash and underscore.')
  }
  return band
}

/** A slug the app addresses the layer by, and which becomes part of a cache key. */
export function validateSlug(raw) {
  const slug = String(raw ?? '').trim().toLowerCase()
  if (!/^[a-z0-9][a-z0-9-]{1,48}$/.test(slug)) {
    throw new CustomLayerError('A short name uses letters, digits and dashes, 2 to 49 characters.')
  }
  return slug
}

const HEX = /^#[0-9a-fA-F]{6}$/

export function validatePalette(raw) {
  const list = Array.isArray(raw)
    ? raw
    : String(raw ?? '').split(/[,\s]+/).filter(Boolean)
  const colors = list.map((c) => {
    const s = String(c).trim()
    // Earth Engine takes bare hex without the hash; the hash is accepted here
    // because that is how every palette is written everywhere else.
    const withHash = s.startsWith('#') ? s : `#${s}`
    if (!HEX.test(withHash)) throw new CustomLayerError(`“${s}” is not a 6-digit hex color.`)
    return withHash.toLowerCase()
  })
  if (colors.length > 32) throw new CustomLayerError('A palette of more than 32 colors is not readable.')
  return colors
}

function num(raw, name, { required = false } = {}) {
  if (raw === null || raw === undefined || raw === '') {
    if (required) throw new CustomLayerError(`${name} is required.`)
    return null
  }
  const n = Number(raw)
  if (!Number.isFinite(n)) throw new CustomLayerError(`${name} must be a number.`)
  return n
}

function isoDate(raw, name) {
  if (raw === null || raw === undefined || raw === '') return null
  const s = String(raw).slice(0, 10)
  if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) throw new CustomLayerError(`${name} must be a date like 2026-09-01.`)
  return s
}

/**
 * Validate and normalise a layer an administrator submitted.
 *
 * Throws CustomLayerError with a message written for the person filling the
 * form. Everything that reaches Earth Engine is checked against a shape;
 * everything else is clamped or trimmed.
 */
export function normaliseCustomLayer(input = {}) {
  const assetType = String(input.asset_type || 'image')
  if (!ASSET_TYPES[assetType]) throw new CustomLayerError('Choose a single image or an image collection.')

  const reducer = String(input.reducer || 'mosaic')
  if (assetType === 'image_collection' && !REDUCERS[reducer]) {
    throw new CustomLayerError('Choose how to combine the collection.')
  }

  const name = String(input.name || '').trim().slice(0, 80)
  if (!name) throw new CustomLayerError('A name is required — it is what the layer is called on the map.')

  const min = num(input.vis_min, 'Minimum')
  const max = num(input.vis_max, 'Maximum')
  if (min !== null && max !== null && min >= max) {
    throw new CustomLayerError('The maximum must be above the minimum.')
  }

  const palette = validatePalette(input.palette)
  if (palette.length === 1) {
    throw new CustomLayerError('A palette needs either no colors or at least two to ramp between.')
  }

  const from = isoDate(input.date_from, 'Start date')
  const to = isoDate(input.date_to, 'End date')
  if (from && to && from > to) throw new CustomLayerError('The start date is after the end date.')

  const opacity = num(input.opacity, 'Opacity')

  return {
    slug: validateSlug(input.slug || name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')),
    name,
    group: String(input.group || 'Custom').trim().slice(0, 40) || 'Custom',
    asset_id: validateAssetId(input.asset_id),
    asset_type: assetType,
    // A single image may legitimately have one band and not need naming; a
    // collection is reduced first, so the band is selected before that.
    band: validateBand(input.band),
    reducer: assetType === 'image_collection' ? reducer : null,
    date_from: assetType === 'image_collection' ? from : null,
    date_to: assetType === 'image_collection' ? to : null,
    vis_min: min,
    vis_max: max,
    palette,
    // Values at or below this are hidden. Without it a layer whose no-data is
    // zero paints the whole world the bottom of the ramp.
    mask_below: num(input.mask_below, 'Hide values below'),
    opacity: opacity === null ? 0.8 : Math.min(1, Math.max(0.05, opacity)),
    // The gate a viewer has to clear, which is not the same vocabulary as the
    // tiers an account can hold — deliberately not TIERS from tiers.mjs. A
    // layer gated on 'member' is visible to perpetual members and admins
    // through atLeast(); a layer gated on 'perpetual' would mean nothing, and
    // the check constraint in migration 003 would reject it.
    tier: ['free', 'member', 'admin'].includes(input.tier) ? input.tier : 'member',
    attribution: String(input.attribution || '').trim().slice(0, 200),
    note: String(input.note || '').trim().slice(0, 600),
  }
}

/**
 * Turn a stored layer into an Earth Engine image and its visualisation.
 *
 * Mirrors the shape the built-in catalogue's build() returns, so the tile
 * function does not have to know which kind of layer it is rendering.
 */
export function buildCustomLayer(ee, layer) {
  let image
  if (layer.asset_type === 'image_collection') {
    let collection = ee.ImageCollection(layer.asset_id)
    if (layer.date_from || layer.date_to) {
      // Open-ended either way: a start with no end means "since", and Earth
      // Engine is content with dates far outside the collection.
      collection = collection.filterDate(layer.date_from || '1970-01-01',
                                         layer.date_to || '2100-01-01')
    }
    if (layer.band) collection = collection.select(layer.band)
    const reduce = layer.reducer || 'mosaic'
    image = collection[reduce]()
  } else {
    image = ee.Image(layer.asset_id)
    if (layer.band) image = image.select(layer.band)
  }

  if (layer.mask_below !== null && layer.mask_below !== undefined) {
    image = image.updateMask(image.gt(layer.mask_below))
  }

  const vis = {}
  if (layer.vis_min !== null && layer.vis_min !== undefined) vis.min = layer.vis_min
  if (layer.vis_max !== null && layer.vis_max !== undefined) vis.max = layer.vis_max
  // Earth Engine wants bare hex. A palette on a multi-band image is refused
  // outright, which is why the band field matters and why the form says so.
  if (layer.palette?.length) vis.palette = layer.palette.map((c) => c.replace('#', ''))

  return { image, vis }
}

/** The shape the map's layer list and legend expect, matching describeLayer. */
export function describeCustomLayer(layer) {
  return {
    key: `custom:${layer.slug}`,
    name: layer.name,
    group: layer.group,
    attribution: layer.attribution || 'Earth Engine',
    opacity: layer.opacity,
    note: layer.note,
    tier: layer.tier,
    slow: false,
    custom: true,
    legend: layer.palette?.length
      ? {
        type: 'ramp',
        unit: '',
        min: layer.vis_min === null || layer.vis_min === undefined ? 'low' : String(layer.vis_min),
        max: layer.vis_max === null || layer.vis_max === undefined ? 'high' : String(layer.vis_max),
        stops: layer.palette,
      }
      : null,
    params: {},
  }
}

/** Is this key one of ours? Custom layers are namespaced so they cannot collide. */
export const CUSTOM_PREFIX = 'custom:'
export const isCustomKey = (key) => String(key || '').startsWith(CUSTOM_PREFIX)
export const slugFromKey = (key) => String(key || '').slice(CUSTOM_PREFIX.length)

// ─────────────────────────────────────────────────────────────────────────────
// Presets
// ─────────────────────────────────────────────────────────────────────────────
//
// The registration form asks for eleven things, and ten of them are the same
// every time for a given KIND of raster. A percent-cover layer always wants 0
// to 100, a green ramp and zero masked out; a classified grid always wants
// discrete colours and no ramp at all. Getting those wrong is not an error
// anybody sees — the layer renders, it just renders badly, as a uniform wash or
// an all-black square, and the asset gets blamed.
//
// Deliberately no asset IDs. A preset describes how to PAINT a raster, which is
// knowledge worth shipping; which asset you have is yours, and a guessed ID
// that does not exist is worse than an empty field.

export const LAYER_PRESETS = [
  {
    id: 'cover-percent',
    label: 'Cover, percent (0–100)',
    hint: 'Tree or canopy cover as a percentage. Exported from a classification '
      + 'or taken from a global product.',
    assetHint: 'projects/your-project/assets/tree-cover-2026',
    layer: {
      group: 'Cover',
      asset_type: 'image',
      band: 'cover',
      vis_min: 0,
      vis_max: 100,
      // Zero is "no trees" and would otherwise paint every field, road and lake
      // the bottom of the ramp, which reads as data rather than as absence.
      mask_below: 0,
      palette: '#f7fcb9, #addd8e, #41ab5d, #238443, #005a32',
      opacity: 0.75,
      note: 'Percent canopy cover. Ground with no cover is left unpainted rather '
        + 'than drawn as the low end, so blank means none rather than unknown.',
    },
  },
  {
    id: 'cover-classes',
    label: 'Classified cover (discrete classes)',
    hint: 'A remapped grid where each value is a class rather than a quantity — '
      + 'the usual output of an Export.image.toAsset() over a classification.',
    assetHint: 'projects/your-project/assets/cover-classes',
    layer: {
      group: 'Cover',
      asset_type: 'image',
      band: 'classification',
      // Classes are counted from 1, and 0 is left as the nodata value. A ramp
      // across ten classes is meaningless, so the palette is ten distinct
      // colours and Earth Engine assigns them in order.
      vis_min: 1,
      vis_max: 10,
      mask_below: 0,
      palette: '#1b7837, #5aae61, #a6dba0, #d9f0d3, #e7d4e8, '
        + '#c2a5cf, #9970ab, #762a83, #b35806, #542788',
      opacity: 0.8,
      note: 'Each colour is a class, not a level: the order carries no magnitude. '
        + 'Set the maximum to however many classes the asset actually has.',
    },
  },
  {
    id: 'canopy-height',
    label: 'Canopy height, metres (0–40)',
    hint: 'A height raster, from lidar or a global canopy height model.',
    assetHint: 'projects/your-project/assets/canopy-height',
    layer: {
      group: 'Cover',
      asset_type: 'image',
      band: 'height',
      vis_min: 0,
      vis_max: 40,
      mask_below: 0,
      palette: '#ffffcc, #c2e699, #78c679, #31a354, #006837',
      opacity: 0.75,
      note: 'Modelled canopy height in metres. Tall closed canopy saturates near '
        + 'the top of the range, so it separates open from forested far better '
        + 'than it separates old growth from mature.',
    },
  },
  {
    id: 'index',
    label: 'Index, −1 to 1 (NDVI and similar)',
    hint: 'A normalised difference index, where the sign means something.',
    assetHint: 'projects/your-project/assets/ndvi-summer',
    layer: {
      group: 'Vegetation',
      asset_type: 'image_collection',
      band: 'NDVI',
      reducer: 'median',
      vis_min: -0.2,
      vis_max: 1,
      // No mask: on a signed index, zero and below are real readings — bare
      // ground, water — not absence.
      palette: '#bfa06a, #dfd39a, #c3d17a, #7fb04a, #3d8228, #14520f',
      opacity: 0.7,
      note: 'Dense conifer and dense broadleaf both saturate near the top, so this '
        + 'separates bare from green far better than it separates forest types.',
    },
  },
  {
    id: 'probability',
    label: 'Probability or suitability (0–1)',
    hint: 'A model output where every pixel is a likelihood.',
    assetHint: 'projects/your-project/assets/habitat-suitability',
    layer: {
      group: 'Custom',
      asset_type: 'image',
      band: 'probability',
      vis_min: 0,
      vis_max: 1,
      // Low probability is a result, but painting the whole world pale is not
      // informative; a floor keeps the layer readable over the observations.
      mask_below: 0.1,
      palette: '#fee5d9, #fcae91, #fb6a4a, #de2d26, #a50f15',
      opacity: 0.7,
      note: 'A model output, not an observation. Values below 0.1 are hidden so the '
        + 'layer does not wash out the map; that is a display choice, not a threshold '
        + 'the model endorses.',
    },
  },
]

/** Apply a preset over a draft, leaving anything the admin has already typed. */
export function applyPreset(draft, presetId) {
  const preset = LAYER_PRESETS.find((p) => p.id === presetId)
  if (!preset) return draft
  const out = { ...draft }
  for (const [key, value] of Object.entries(preset.layer)) {
    // The asset ID, the name and the slug identify a particular layer and are
    // never a preset's business — overwriting a half-typed name to apply a
    // palette is the kind of help nobody asked for.
    if (['asset_id', 'name', 'slug', 'id', 'tier'].includes(key)) continue
    out[key] = value
  }
  return out
}
