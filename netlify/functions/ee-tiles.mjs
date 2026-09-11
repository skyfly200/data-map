// Mint a tile URL for an Earth Engine layer.
//
//   GET /.netlify/functions/ee-tiles                     the catalogue
//   GET /.netlify/functions/ee-tiles?layer=…&through=…    a tile template
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

import { getStore } from '@netlify/blobs'

import { requireTier } from '../lib/auth.mjs'
import {
  LayerError, EE_LAYER_CATALOGUE, cacheKey, describeLayer, resolveLayer, tierFor,
} from '../lib/ee-tile-layers.mjs'
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

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: {
    'content-type': 'application/json',
    // The catalogue is static; a minted template is shared and short-lived.
    'cache-control': status === 200 ? 'private, max-age=300' : 'no-store',
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
    ee.data.getMapId({ image, ...vis }, (result, error) => {
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
    message: `“${layer.name}” is a members' layer.`,
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
      error: `Earth Engine could not render “${layer.name}”: ${detail}`,
      hint: 'Check the asset ID, that the service account has read access to it, '
        + 'and that a band is named if the asset has more than one.',
      layer: key,
    }, 502)
  }
}

export default async function handler(request) {
  const url = new URL(request.url)
  const key = url.searchParams.get('layer')

  // The catalogue is not gated: the map needs it to know what to offer, and
  // knowing a layer exists is not the same as being able to render it.
  if (!key) {
    const custom = (await customLayers()).map(describeCustomLayer)
    return json({ ok: true, layers: [...EE_LAYER_CATALOGUE, ...custom] })
  }

  // A layer an administrator registered, pointing at their own Earth Engine
  // asset. Same gate, same cache, same error reporting as a built-in — only
  // where the recipe comes from differs.
  if (isCustomKey(key)) return renderCustom(request, key)

  // Gated per layer, not globally: years-since-fire and burn severity answer
  // the question the society exists to help with and are open to everyone,
  // while the layers that spend real compute per tile are what membership buys.
  // See DEFAULT_TIER in ../lib/ee-tile-layers.mjs.
  const auth = await requireTier(request, tierFor(key), {
    message: `“${describeLayer(key)?.name || key}” is a members' layer. `
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
    resolved = resolveLayer(key, Object.fromEntries(url.searchParams))
  } catch (err) {
    if (err instanceof LayerError) return json({ ok: false, error: err.message }, 400)
    throw err
  }

  const { layer, params } = resolved
  const id = cacheKey(key, params)
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
      const n = await new Promise((resolve, reject) => {
        layer.count(ee, params).evaluate((v, err) => (err ? reject(new Error(String(err))) : resolve(v)))
      })
      if (!n) {
        const when = params.year ?? params.through
        return json({
          ok: false,
          error: `No ${layer.name} data has been published for `
            + `${when !== undefined ? when : `the last ${params.days} days`} yet. `
            + 'These products lag real time; try an earlier period.',
          layer: key,
          params,
          empty: true,
        }, 404)
      }
    }

    const { image, vis } = layer.build(ee, params)
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
      error: `Earth Engine could not render “${layer.name}”: ${detail}`,
      layer: key,
      params,
    }, 502)
  }
}
