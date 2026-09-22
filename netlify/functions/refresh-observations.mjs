// Scheduled Netlify Function: refresh the observations between full pipeline
// runs. Fetches recent iNaturalist sightings, keeps only those NOT already in
// the current dataset, samples the terrain rasters for them (if the rasters
// are bundled), and writes the merged result back as a single authoritative
// observations.geojson in Supabase Storage (or stores only the new features in
// Netlify Blobs when Supabase is not configured — the serving function overlays
// them on the committed baseline in that case).

import { getStore } from '@netlify/blobs'
import { join } from 'node:path'
import { fetchInatFeatures, newFeatures, overlay } from '../lib/observations.mjs'
import { openTerrain, enrichFeatureTerrain } from '../lib/terrain.mjs'
import { loadBaseline } from '../lib/baseline.mjs'
import { supabaseConfigured } from '../lib/supabase-storage.mjs'
import { uploadJson, readJson } from '../lib/datasets-store.mjs'

// Run every 6 hours. Adjust the cron as needed.
export const config = { schedule: '0 */6 * * *' }

export default async () => {
  try {
    let baseline = null
    if (supabaseConfigured()) baseline = await readJson('observations.geojson')
    if (!baseline) baseline = await loadBaseline()

    const opts = {
      taxonName: process.env.INAT_TAXON || 'morchella',
      lat: Number(process.env.INAT_LAT ?? 40.0),
      lng: Number(process.env.INAT_LNG ?? -105.0),
      radius: Number(process.env.INAT_RADIUS ?? 500),
      perPage: Number(process.env.INAT_PER_PAGE ?? 200),
    }

    const fresh = await fetchInatFeatures(opts)
    const news = newFeatures(baseline, fresh)

    // Best-effort terrain context for the new points.
    if (news.length) {
      const readers = await openTerrain(join(process.cwd(), 'data/terrain')).catch(() => ({}))
      if (Object.keys(readers).length) {
        for (const f of news) await enrichFeatureTerrain(f, readers)
      }
    }

    let sink = 'netlify-blob'
    if (supabaseConfigured()) {
      // Merge new features into the existing dataset and write back a single
      // authoritative file. The serving function reads this directly with no
      // overlay needed.
      const merged = overlay(baseline, news)
      await uploadJson('observations.geojson', merged)
      sink = 'supabase'
    } else {
      // Non-Supabase path: store only the new features; the serving function
      // overlays them on the committed baseline at request time.
      const store = getStore('observations')
      await store.setJSON('new-observations', { type: 'FeatureCollection', features: news })
    }

    return new Response(
      JSON.stringify({ ok: true, sink, baseline: baseline.features?.length ?? 0, new: news.length }),
      { headers: { 'content-type': 'application/json' } },
    )
  } catch (err) {
    return new Response(JSON.stringify({ ok: false, error: String(err) }), {
      status: 500,
      headers: { 'content-type': 'application/json' },
    })
  }
}
