// Sample the active Earth Engine layers at one point.
//
//   POST /.netlify/functions/ee-sample
//   body: { lat, lon, layers: [{ key, params }] }
//
// The tile path (ee-tiles) draws a layer everywhere; this reads one pixel of it
// where the viewer dropped a pin, so the map can say "here, soil moisture is
// 0.23" rather than only shade it. One reduceRegion per layer, evaluated
// server-side, so the browser never speaks to Earth Engine directly.
//
// Unlike a tile mint, a sample is NOT cached: it is a single cheap read, it is
// only issued when the viewer presses a button, and the point is different every
// time. The per-layer tier gate still applies — reading a member's layer is the
// same spend whether it is drawn or sampled.

import { requireTier } from '../lib/auth.mjs'
import {
  LayerError, describeLayer, resolveLayer, tierFor,
} from '../lib/ee-tile-layers.mjs'
import { earthEngineConfigured, initEarthEngine } from '../lib/ee-runner.mjs'
import { isCustomKey } from '../lib/ee-custom-layers.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

/** Read one pixel of an image at a point, as a plain object of band→value. */
function sampleImage(ee, image, lon, lat) {
  const point = ee.Geometry.Point([lon, lat])
  // A small scale samples the pixel under the point for fine layers and still
  // returns the covering pixel for coarse ones; bestEffort keeps a coarse layer
  // from tripping the pixel budget. first() is the pixel value, not an average.
  const dict = image.reduceRegion({
    reducer: ee.Reducer.first(),
    geometry: point,
    scale: 100,
    maxPixels: 1e9,
    bestEffort: true,
  })
  return new Promise((resolve, reject) => {
    dict.evaluate((v, err) => (err ? reject(new Error(String(err))) : resolve(v || {})))
  })
}

/**
 * Turn the raw band values into something a reader can use: the number for a
 * continuous layer, the class label for a categorical one, the three channels
 * for an RGB composite.
 */
function formatSample(key, values) {
  const described = describeLayer(key)
  const legend = described?.legend
  const bands = Object.keys(values || {})
  if (!bands.length) return { key, name: described?.name || key, empty: true }

  // The single band a one-band layer reduces to, whatever it is named.
  const first = values[bands[0]]

  if (legend?.type === 'classes' && Array.isArray(legend.items) && typeof first === 'number') {
    // Classes are painted min..max over the palette, so the value is a 1-based
    // index into the item list (see the remap layers). Clamp rather than trust.
    const item = legend.items[Math.round(first) - 1]
    return {
      key, name: described.name, value: first,
      label: item?.label || `Class ${first}`, color: item?.color,
    }
  }

  if (bands.length > 1) {
    return {
      key, name: described?.name || key,
      channels: bands.map((b) => ({ band: b, value: values[b] })),
    }
  }

  const num = typeof first === 'number' ? Number(first.toFixed(3)) : first
  return { key, name: described?.name || key, value: num, unit: legend?.unit || '' }
}

export default async function handler(request) {
  if (request.method !== 'POST') return json({ ok: false, error: 'Use POST.' }, 405)

  let body
  try {
    body = await request.json()
  } catch {
    return json({ ok: false, error: 'Send the request as JSON.' }, 400)
  }

  const lat = Number(body?.lat)
  const lon = Number(body?.lon)
  if (!Number.isFinite(lat) || !Number.isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    return json({ ok: false, error: 'A valid lat and lon are required.' }, 400)
  }
  const requested = Array.isArray(body?.layers) ? body.layers.slice(0, 12) : []
  if (!requested.length) return json({ ok: false, error: 'Name at least one layer to sample.' }, 400)

  if (!earthEngineConfigured()) {
    return json({ ok: false, error: 'Earth Engine is not configured on this deployment.' }, 503)
  }

  // The strictest tier among the requested layers gates the whole request, so a
  // free viewer sampling one free layer is never turned away, but a members'
  // layer in the list needs membership. Custom (admin-registered) layers are not
  // sampled here — their recipe lives in the database, not the catalogue.
  const keys = requested.map((r) => String(r?.key || '')).filter((k) => k && !isCustomKey(k))
  if (!keys.length) return json({ ok: false, error: 'None of those layers can be sampled.' }, 400)

  for (const key of keys) {
    const auth = await requireTier(request, tierFor(key), {
      message: `“${describeLayer(key)?.name || key}” is a members' layer.`,
    })
    if (!auth.ok) return auth.response
  }

  try {
    const ee = await initEarthEngine()
    const results = []
    for (const r of requested) {
      const key = String(r?.key || '')
      if (!key || isCustomKey(key)) continue
      try {
        const { layer, params } = resolveLayer(key, r?.params || {})
        const { image } = layer.build(ee, params)
        const values = await sampleImage(ee, image, lon, lat)
        results.push(formatSample(key, values))
      } catch (err) {
        // One layer failing does not fail the rest: the panel shows what it
        // could read and marks what it could not.
        const detail = err instanceof LayerError ? err.message : String(err?.message || err).slice(0, 160)
        results.push({ key, name: describeLayer(key)?.name || key, error: detail })
      }
    }
    return json({ ok: true, lat, lon, results })
  } catch (err) {
    return json({ ok: false, error: String(err?.message || err).slice(0, 300) }, 502)
  }
}
