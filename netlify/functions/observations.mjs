// Serving Netlify Function: returns the observations GeoJSON for the map.
// Prefers the fresh copy written to Netlify Blobs by refresh-observations.mjs;
// falls back to the committed baseline file if the blob is not present yet.
//
// The frontend calls /.netlify/functions/observations and itself falls back to
// the static /data/observations.geojson if this function is unavailable.

import { getStore } from '@netlify/blobs'
import { loadBaseline } from '../lib/baseline.mjs'

export default async () => {
  let collection

  try {
    const store = getStore('observations')
    collection = await store.get('latest', { type: 'json' })
  } catch {
    // Blobs unavailable — fall through to the baseline file.
  }

  if (!collection) collection = await loadBaseline()

  return new Response(JSON.stringify(collection), {
    headers: {
      'content-type': 'application/geo+json',
      'cache-control': 'public, max-age=300',
    },
  })
}
