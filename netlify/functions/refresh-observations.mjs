// Scheduled Netlify Function: refresh the observations between full pipeline
// runs. Fetches recent iNaturalist sightings, merges any new ones onto the
// committed baseline, samples the terrain rasters for those new points (if the
// rasters are bundled), and stores the result in Netlify Blobs. The serving
// function (observations.mjs) then returns this fresh copy.
//
// The heavy satellite/terrain enrichment and clustering stay in the offline
// Python pipeline (run by the GitHub Action), which regenerates the baseline;
// this function only surfaces new finds quickly in between.

import { getStore } from '@netlify/blobs'
import { join } from 'node:path'
import { fetchInatFeatures, mergeByUuid } from '../lib/observations.mjs'
import { openTerrain, enrichFeatureTerrain } from '../lib/terrain.mjs'
import { loadBaseline } from '../lib/baseline.mjs'

// Run every 6 hours. Adjust the cron as needed.
export const config = { schedule: '0 */6 * * *' }

export default async () => {
  try {
    const baseline = await loadBaseline()

    const opts = {
      taxonName: process.env.INAT_TAXON || 'morchella',
      lat: Number(process.env.INAT_LAT ?? 40.0),
      lng: Number(process.env.INAT_LNG ?? -105.0),
      radius: Number(process.env.INAT_RADIUS ?? 500),
      perPage: Number(process.env.INAT_PER_PAGE ?? 100),
    }

    const fresh = await fetchInatFeatures(opts)
    const { collection, added, addedUuids } = mergeByUuid(baseline, fresh)

    // Best-effort terrain context for the newly added points.
    if (added > 0) {
      const readers = await openTerrain(join(process.cwd(), 'data/terrain')).catch(() => ({}))
      if (Object.keys(readers).length) {
        const isNew = new Set(addedUuids)
        for (const f of collection.features) {
          if (isNew.has(f.properties?.uuid)) await enrichFeatureTerrain(f, readers)
        }
      }
    }

    const store = getStore('observations')
    await store.setJSON('latest', collection)

    return new Response(
      JSON.stringify({ ok: true, total: collection.features.length, added }),
      { headers: { 'content-type': 'application/json' } },
    )
  } catch (err) {
    return new Response(JSON.stringify({ ok: false, error: String(err) }), {
      status: 500,
      headers: { 'content-type': 'application/json' },
    })
  }
}
