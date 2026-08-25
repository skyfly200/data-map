// On-demand fetch of a new species from iNaturalist (Node). Returns the
// GeoJSON so the UI can show it immediately, and — when Supabase Storage is
// configured — persists it as species/<slug>.geojson and adds it to the
// dataset manifest so it survives reloads.
//
//   GET/POST /.netlify/functions/fetch-species?species=Amanita%20muscaria

import { fetchInatFeatures } from '../lib/observations.mjs'
import { supabaseConfigured, publicUrl } from '../lib/supabase-storage.mjs'
import { uploadJson, readJson } from '../lib/datasets-store.mjs'
import { clusterFeatures } from '../lib/cluster.mjs'

export const config = { timeout: 60 }

function slugify(s) {
  return String(s).trim().toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'species'
}
function json(obj, status = 200) {
  return new Response(JSON.stringify(obj), { status, headers: { 'content-type': 'application/json' } })
}

export default async (request) => {
  try {
    const url = new URL(request.url)
    let species = url.searchParams.get('species')
    if (!species && request.method === 'POST') {
      try { species = (await request.json())?.species } catch { /* ignore */ }
    }
    if (!species || !species.trim()) return json({ ok: false, error: 'Missing "species"' }, 400)

    const opts = {
      taxonName: species.trim(),
      lat: Number(url.searchParams.get('lat') ?? process.env.INAT_LAT ?? 40.0),
      lng: Number(url.searchParams.get('lng') ?? process.env.INAT_LNG ?? -105.0),
      radius: Number(url.searchParams.get('radius') ?? process.env.INAT_RADIUS ?? 500),
      perPage: 200,
      qualityGrade: 'research',
    }

    const rawFeatures = await fetchInatFeatures(opts)
    // Give freshly-fetched species meaningful groups now (spatial/temporal),
    // until the full Python pipeline re-runs and clusters on enriched rasters.
    const features = clusterFeatures(rawFeatures)
    const geojson = { type: 'FeatureCollection', features }
    const slug = slugify(species)
    let path = null

    // Persist to Supabase Storage + manifest (best effort) when configured.
    if (supabaseConfigured() && features.length) {
      try {
        await uploadJson(`species/${slug}.geojson`, geojson)
        path = publicUrl(`species/${slug}.geojson`)
        const manifest = (await readJson('datasets.json')) || []
        const entry = { id: slug, label: `${species.trim()} (${features.length})`, path, count: features.length }
        const idx = manifest.findIndex((d) => d.id === slug)
        if (idx >= 0) manifest[idx] = entry
        else manifest.push(entry)
        await uploadJson('datasets.json', manifest, 'application/json')
      } catch (e) {
        console.warn('Supabase persist skipped:', String(e))
      }
    }

    return json({ ok: true, species: species.trim(), slug, count: features.length, path, geojson })
  } catch (err) {
    return json({ ok: false, error: String(err) }, 500)
  }
}
