// Serving Netlify Function: returns the observations GeoJSON for the map.
// The committed baseline is authoritative (fully enriched + clustered); the
// scheduled function's blob only adds genuinely-new interim sightings on top.
// This guarantees a newer baseline always wins over an older blob.
//
// The frontend calls /.netlify/functions/observations and itself falls back to
// the static /data/observations.geojson if this function is unavailable.

import { getStore } from '@netlify/blobs'
import { overlay } from '../lib/observations.mjs'
import { loadBaseline } from '../lib/baseline.mjs'

export default async () => {
  const baseline = await loadBaseline()

  let extras = []
  try {
    const store = getStore('observations')
    const blob = await store.get('new-observations', { type: 'json' })
    extras = blob?.features || []
  } catch {
    // Blobs unavailable — serve the baseline alone.
  }

  const collection = overlay(baseline, extras)

  return new Response(JSON.stringify(collection), {
    headers: {
      'content-type': 'application/geo+json',
      'cache-control': 'public, max-age=300',
    },
  })
}
