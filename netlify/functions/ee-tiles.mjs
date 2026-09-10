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
import { LayerError, EE_LAYER_CATALOGUE, cacheKey, describeLayer, resolveLayer } from '../lib/ee-tile-layers.mjs'
import { earthEngineConfigured, initEarthEngine } from '../lib/ee-runner.mjs'

/** How long a minted template is reused. Well inside Earth Engine's own expiry. */
const TTL_MS = 6 * 60 * 60 * 1000

/**
 * Which tier may mint one.
 *
 * Rendering costs the society's Earth Engine quota, and the computed layers
 * cost real time, so this follows the same rule as the pipeline: it is what
 * membership buys. Change to 'free' to put the fire layers on the public map.
 */
const REQUIRED_TIER = 'member'

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

export default async function handler(request) {
  const url = new URL(request.url)
  const key = url.searchParams.get('layer')

  // The catalogue is not gated: the map needs it to know what to offer, and
  // knowing a layer exists is not the same as being able to render it.
  if (!key) return json({ ok: true, layers: EE_LAYER_CATALOGUE })

  const auth = await requireTier(request, REQUIRED_TIER)
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
