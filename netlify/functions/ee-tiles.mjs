// Mint a tile URL for an Earth Engine layer.
//
//   GET  /.netlify/functions/ee-tiles                     the catalogue
//   GET  /.netlify/functions/ee-tiles?layer=…&through=…    a tile template
//   POST /.netlify/functions/ee-tiles  {layer, …}          the same, for big
//                                                          parameters
//
// The POST form exists for one parameter: the soil taxonomy layer's list of
// class codes. Every class in that raster at once is a few thousand characters,
// which is past what a query string can be relied on to carry, and a selection
// that fails at a size nobody can predict is worse than one that cannot be made
// at all. Same handler, same validation, same cache — only where the parameters
// were read from differs.
//
// Earth Engine renders tiles on demand behind a map id. Asking for one is a
// single API call, and the answer is a plain XYZ template the browser fetches
// directly — so the tiles themselves never pass through this function, and one
// mint serves every viewer looking at the same layer.
//
// That is also why it is cached. A map id costs an Earth Engine call to create
// and is identical for everyone at the same parameters, so two members opening
// last year's burn severity should share one rather than spending twice.
//
// Map ids expire. The cache TTL is deliberately well short of Earth Engine's
// own expiry, because a stale template does not error — it serves blank tiles,
// which on a fire map reads as "nothing burned here". Expiring early costs one
// API call; expiring late tells a lie.

import { createHash } from 'node:crypto'

import { getStore } from '@netlify/blobs'

import { requireTier } from '../lib/auth.mjs'
import {
  LayerError, EE_LAYER_CATALOGUE, EE_TILE_LAYERS,
  cacheKey, describeLayer, resolveLayer, tierFor, visParams,
} from '../lib/ee-tile-layers.mjs'
import {
  MATSUTAKE_GREAT_GROUPS, SOIL_ORDERS, SOIL_TAXONOMY_WIKI, describeGreatGroup,
} from '../lib/soil-taxonomy.mjs'
import { earthEngineConfigured, initEarthEngine } from '../lib/ee-runner.mjs'
import {
  buildCustomLayer, describeCustomLayer, isCustomKey, slugFromKey,
} from '../lib/ee-custom-layers.mjs'
import { adminClient } from '../lib/auth.mjs'

/**
 * The layers an administrator has registered, for the catalogue and for
 * rendering. Read past RLS so the catalogue is one query rather than one per
 * tier — the tile path checks the tier itself before spending anything.
 */
async function customLayers() {
  const client = adminClient()
  if (!client) return []
  const { data, error } = await client.from('ee_custom_layers').select('*').order('name')
  if (error) return []
  return data || []
}

/** How long a minted template is reused. Well inside Earth Engine's own expiry. */
const TTL_MS = 6 * 60 * 60 * 1000

/**
 * How long a layer's class table is reused.
 *
 * Much longer than a map id, because it is not one: it is the code → name table
 * carried on a published asset, and it changes when the publisher issues a new
 * version — which is a new asset id, and therefore a new cache key anyway.
 */
const CLASS_TTL_MS = 30 * 24 * 60 * 60 * 1000

/**
 * A blob key for a cache key that may be arbitrarily long.
 *
 * A selection of every soil class is a cache key of a few thousand characters,
 * which is past what a blob store will accept as a name. Hashing keeps it one
 * key per distinct selection — the property the cache depends on — and the
 * readable prefix is kept so a key in the store still says which layer it
 * belongs to.
 */
export function blobId(key) {
  if (key.length <= 120) return key
  const digest = createHash('sha256').update(key).digest('hex').slice(0, 32)
  return `${key.slice(0, 48)}|${digest}`
}

/**
 * The parameters of a request, from the query string and, for a POST, the body.
 *
 * The body wins where both name the same thing. Nothing here trusts either:
 * every value still goes through resolveLayer, which is the only thing that
 * decides what reaches Earth Engine.
 */
export async function readInput(request, url) {
  const query = Object.fromEntries(url.searchParams)
  if (request.method !== 'POST') return query
  let body
  try {
    body = await request.json()
  } catch {
    throw new LayerError('The request body could not be read as JSON.')
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) return query
  return { ...query, ...body }
}

/** Resolve an Earth Engine value. `evaluate` is callback-shaped in the client. */
function evaluate(obj) {
  return new Promise((resolve, reject) => {
    obj.evaluate((value, err) => (err ? reject(new Error(String(err))) : resolve(value)))
  })
}

// Within one warm process, a layer's class table is read once. The blob cache
// below survives a cold start; this saves the read on every request after the
// first in the same process.
const preparedHere = new Map()

/**
 * The table a layer's `prepare` reads off its asset.
 *
 * Cached hard, and cached by layer key: an unrelated layer's table must not be
 * served for this one, and a table that fails to load has to fail the render
 * rather than let build() paint from an empty list — which for a remap means
 * every pixel takes the default and the layer comes back uniformly blank.
 */
async function prepareLayer(ee, key, layer) {
  if (!layer.prepare) return null
  if (preparedHere.has(key)) return preparedHere.get(key)

  const id = `classes|${key}`
  const blobs = store()
  if (blobs) {
    try {
      const hit = await blobs.get(id, { type: 'json' })
      if (hit && hit.expires > Date.now() && hit.table) {
        preparedHere.set(key, hit.table)
        return hit.table
      }
    } catch { /* a cache that cannot be read is a cache miss */ }
  }

  const table = await evaluate(layer.prepare(ee))
  if (!table || !Array.isArray(table.values) || !table.values.length) {
    throw new Error(`${layer.name} could not read its class table from the asset.`)
  }
  preparedHere.set(key, table)
  if (blobs) {
    try { await blobs.setJSON(id, { table, expires: Date.now() + CLASS_TTL_MS }) } catch { /* not fatal */ }
  }
  return table
}

const json = (body, status = 200, extraHeaders = {}) => new Response(JSON.stringify(body), {
  status,
  headers: {
    'content-type': 'application/json',
    // The catalogue is static; a minted template is shared and short-lived.
    'cache-control': status === 200 ? 'private, max-age=300' : 'no-store',
    ...extraHeaders,
  },
})

function store() {
  try {
    return getStore('ee-tiles')
  } catch {
    // Blobs are not available in every context. Losing the cache costs an extra
    // Earth Engine call, which is not a reason to fail the request.
    return null
  }
}

/**
 * Ask Earth Engine for a tile template.
 *
 * getMapId is callback-shaped in the Node client. Newer Earth Engine returns
 * urlFormat directly; the fallback builds the same URL from the map id for
 * older responses.
 */
function getMapTemplate(ee, image, vis) {
  return new Promise((resolve, reject) => {
    // visParams, not vis: the client wants min/max/gamma as strings and throws
    // "csv.split is not a function" on the numbers every layer declares.
    ee.data.getMapId({ image, ...visParams(vis) }, (result, error) => {
      if (error) return reject(new Error(String(error)))
      if (!result) return reject(new Error('Earth Engine returned no map id.'))
      const template = result.urlFormat
        || (result.mapid
          ? `https://earthengine.googleapis.com/v1/${result.mapid}/tiles/{z}/{x}/{y}`
          : null)
      if (!template) return reject(new Error('Earth Engine returned no tile URL.'))
      return resolve(template)
    })
  })
}

async function renderCustom(request, key) {
  const rows = await customLayers()
  const layer = rows.find((l) => l.slug === slugFromKey(key))
  if (!layer) return json({ ok: false, error: 'That layer no longer exists.' }, 404)

  const auth = await requireTier(request, layer.tier, {
    message: `"${layer.name}" is a members' layer.`,
  })
  if (!auth.ok) return auth.response

  if (!earthEngineConfigured()) {
    return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
  }

  const described = describeCustomLayer(layer)
  // The updated_at timestamp is part of the key, so editing a layer's palette
  // or band shows the change immediately instead of serving the cached render
  // of the previous settings for up to six hours.
  const id = `${key}|${layer.updated_at}`
  const blobs = store()

  if (blobs) {
    try {
      const hit = await blobs.get(id, { type: 'json' })
      if (hit && hit.expires > Date.now()) {
        return json({ ok: true, ...described, template: hit.template, cached: true })
      }
    } catch { /* a cache that cannot be read is a cache miss */ }
  }

  try {
    const ee = await initEarthEngine()
    const { image, vis } = buildCustomLayer(ee, layer)
    const template = await getMapTemplate(ee, image, vis)
    if (blobs) {
      try { await blobs.setJSON(id, { template, expires: Date.now() + TTL_MS }) } catch { /* not fatal */ }
    }
    return json({ ok: true, ...described, template, cached: false })
  } catch (err) {
    // The likeliest faults are an asset that does not exist, one the service
    // account cannot read, or a palette on a multi-band image. Earth Engine
    // says which, so its words are passed through rather than summarised.
    const detail = String(err?.message || err).slice(0, 300)
    return json({
      ok: false,
      error: `Earth Engine could not render "${layer.name}": ${detail}`,
      hint: 'Check the asset ID, that the service account has read access to it, '
        + 'and that a band is named if the asset has more than one.',
      layer: key,
    }, 502)
  }
}

/**
 * A layer's class table, described.
 *
 *   GET /.netlify/functions/ee-tiles?classes=soil-taxonomy
 *
 * Four hundred great groups is too much to put in the catalogue every viewer
 * loads, and too much to ask Earth Engine for again each time somebody types in
 * a search box. So it is its own request, made once when the layer is switched
 * on, and each class arrives with the description already assembled — the
 * decoding rules live on the server next to the palette that uses them, and a
 * second copy in the browser is a second copy to disagree.
 */
async function classTable(request, key) {
  const layer = EE_TILE_LAYERS[key]
  if (!layer?.prepare) {
    return json({ ok: false, error: `"${key}" has no class table.` }, 404)
  }

  // Same gate as rendering it. The names are not a secret, but asking for them
  // spends an Earth Engine call on a cold cache.
  const auth = await requireTier(request, tierFor(key), {
    message: `"${layer.name}" is a members' layer.`,
  })
  if (!auth.ok) return auth.response

  if (!earthEngineConfigured()) {
    return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
  }

  try {
    const ee = await initEarthEngine()
    const table = await prepareLayer(ee, key, layer)
    const classes = (table.names || []).map((name, i) => ({
      code: table.values[i],
      ...describeGreatGroup(name),
    }))
    return json({
      ok: true,
      layer: key,
      orders: SOIL_ORDERS,
      wiki: SOIL_TAXONOMY_WIKI,
      flagged: MATSUTAKE_GREAT_GROUPS,
      classes,
    })
  } catch (err) {
    const detail = String(err?.message || err).slice(0, 300)
    return json({ ok: false, error: `Could not read the classes of "${layer.name}": ${detail}` }, 502)
  }
}

export default async function handler(request) {
  const url = new URL(request.url)

  let input
  try {
    input = await readInput(request, url)
  } catch (err) {
    if (err instanceof LayerError) return json({ ok: false, error: err.message }, 400)
    throw err
  }

  const key = input.layer || null

  const wantsClasses = input.classes
  if (wantsClasses) return classTable(request, String(wantsClasses))

  // The catalogue is not gated: the map needs it to know what to offer, and
  // knowing a layer exists is not the same as being able to render it.
  if (!key) {
    const custom = (await customLayers()).map(describeCustomLayer)
    // The catalogue is the same for every viewer and only changes when an admin
    // registers a layer, so it is cached at the CDN edge as well as in the
    // browser: the function runs once an hour per edge node to build this list
    // rather than once per page load. Only the catalogue is shared-cached — a
    // minted tile template carries a token and stays private (see json above).
    return json({ ok: true, layers: [...EE_LAYER_CATALOGUE, ...custom] }, 200, {
      'cache-control': 'public, max-age=300',
      'netlify-cdn-cache-control': 'public, s-maxage=3600, stale-while-revalidate=86400, durable',
    })
  }

  // A layer an administrator registered, pointing at their own Earth Engine
  // asset. Same gate, same cache, same error reporting as a built-in — only
  // where the recipe comes from differs.
  if (isCustomKey(key)) return renderCustom(request, key)

  // Gated per layer, not globally: years-since-fire and burn severity answer
  // the question FRMS exists to help with and are open to everyone,
  // while the layers that spend real compute per tile are what membership buys.
  // See DEFAULT_TIER in ../lib/ee-tile-layers.mjs.
  const auth = await requireTier(request, tierFor(key), {
    message: `"${describeLayer(key)?.name || key}" is a members' layer. `
      + 'Years since fire and burn severity are open to everyone.',
  })
  if (!auth.ok) return auth.response

  if (!earthEngineConfigured()) {
    return json({
      ok: false,
      error: 'Earth Engine is not configured on this deployment. '
        + 'Set EARTHENGINE_SERVICE_ACCOUNT_KEY and EARTHENGINE_PROJECT.',
    }, 503)
  }

  let resolved
  try {
    resolved = resolveLayer(key, input)
  } catch (err) {
    if (err instanceof LayerError) return json({ ok: false, error: err.message }, 400)
    throw err
  }

  const { layer, params } = resolved
  // Hashed when long: a selection of every soil class is a cache key of a few
  // thousand characters, which is more than a blob store will take as a name.
  const id = blobId(cacheKey(key, params))
  const blobs = store()

  if (blobs) {
    try {
      const hit = await blobs.get(id, { type: 'json' })
      if (hit && hit.expires > Date.now()) {
        return json({ ok: true, ...describeLayer(key), params, template: hit.template, cached: true })
      }
    } catch { /* a cache that cannot be read is a cache miss */ }
  }

  try {
    const ee = await initEarthEngine()

    // Is there anything published for the window being asked for?
    //
    // Reducing an empty collection gives an image with NO bands, and Earth
    // Engine refuses a palette on that — so a year the dataset has not reached
    // yet failed with a message about bands, which tells the reader nothing
    // about the actual problem. One extra call on a cache miss buys an answer
    // they can act on.
    if (layer.count) {
      const n = await evaluate(layer.count(ee, params))
      if (!n) {
        const when = params.year ?? params.through
        const period = when !== undefined ? String(when)
          : params.days !== undefined ? `the last ${params.days} days`
            : ''
        return json({
          ok: false,
          // A layer with no period cannot be empty because of publication lag,
          // so telling its reader to try an earlier one sends them looking for
          // a control that does not exist. For those the count is zero because
          // the asset or the property name is wrong, which is ours to fix.
          error: period
            ? `No ${layer.name} data has been published for ${period} yet. `
              + 'These products lag real time; try an earlier period.'
            : `${layer.name} could not be found in its source dataset. `
              + 'That is a fault in this layer rather than in what you asked for; '
              + 'please report it.',
          layer: key,
          params,
          empty: true,
        }, 404)
      }
    }

    const prepared = await prepareLayer(ee, key, layer)
    const { image, vis } = layer.build(ee, params, prepared)
    const template = await getMapTemplate(ee, image, vis)

    if (blobs) {
      try {
        await blobs.setJSON(id, { template, expires: Date.now() + TTL_MS })
      } catch { /* not caching is not failing */ }
    }
    return json({ ok: true, ...describeLayer(key), params, template, cached: false })
  } catch (err) {
    // Named and specific. A wrong asset id or band is the likeliest fault here
    // and it must not surface as an empty layer: on a fire map, blank ground
    // reads as ground that never burned.
    const detail = String(err?.message || err).slice(0, 300)
    return json({
      ok: false,
      error: `Earth Engine could not render "${layer.name}": ${detail}`,
      layer: key,
      params,
    }, 502)
  }
}
