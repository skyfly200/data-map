// Serving Netlify Function: returns the observations GeoJSON for the map.
//
// When Supabase is configured, observations.geojson in Supabase Storage is the
// single authoritative dataset — the refresh function merges new sightings in
// on each run, so no overlay is needed at serve time.
//
// Without Supabase the committed baseline (bundled with the deployment) is served
// with any new-observations blob from Netlify overlaid on top at request time.
//
// The frontend calls /.netlify/functions/observations and falls back to the
// static /data/observations.geojson if this function is unavailable.

import { getStore } from '@netlify/blobs'
import { overlay } from '../lib/observations.mjs'
import { loadBaseline } from '../lib/baseline.mjs'
import { supabaseConfigured } from '../lib/supabase-storage.mjs'
import { readJson } from '../lib/datasets-store.mjs'
import { taxaInFeatures } from '../lib/dataset-taxa.mjs'

// The taxon summary, computed once per warm process. Parsing fifty megabytes of
// GeoJSON to count names is not something to do per request, and the dataset
// changes at most once per refresh run (every 6 hours), so a process-scoped
// cache is an acceptable trade. A cold start after a refresh picks up the
// latest counts on its first ?summary=taxa request.
let taxaCache = null

/**
 * Which taxa the baseline carries, by rank, with counts.
 *
 *   GET /.netlify/functions/observations?summary=taxa
 *
 * This exists so the job runner's taxon field can offer what is there rather
 * than accept anything typed. The page cannot compute it for itself: it would
 * have to download the whole dataset to count it, which is fifty megabytes to
 * populate a dropdown.
 *
 * Read from the same baseline a bbox job is matched against, which is the point
 * — a name offered here is a name that selects records there.
 */
async function taxaSummary() {
  if (!taxaCache) {
    let dataset = supabaseConfigured() ? await readJson('observations.geojson') : null
    if (!dataset) dataset = await loadBaseline()
    const features = dataset?.features || []
    taxaCache = { ranks: taxaInFeatures(features), total: features.length }
  }
  return new Response(JSON.stringify({ ok: true, ...taxaCache }), {
    headers: {
      'content-type': 'application/json',
      // Static per deployment, so it is worth caching hard at the edge too.
      'cache-control': 'public, max-age=3600, stale-while-revalidate=86400',
    },
  })
}

export default async (request) => {
  const url = new URL(request.url)
  if (url.searchParams.get('summary') === 'taxa') return taxaSummary()

  // When Supabase is configured, observations.geojson is the single authoritative
  // file — the refresh function merges new sightings in on each run, so no
  // overlay is needed here. Fall back to the committed baseline + Netlify Blob
  // overlay for deployments without Supabase.
  let collection = null
  if (supabaseConfigured()) {
    collection = await readJson('observations.geojson')
  }

  if (!collection) {
    const baseline = await loadBaseline()
    let extras = []
    try {
      const store = getStore('observations')
      const blob = await store.get('new-observations', { type: 'json' })
      extras = blob?.features || []
    } catch {
      // Blobs unavailable — serve the baseline alone.
    }
    collection = overlay(baseline, extras)
  }

  return new Response(JSON.stringify(collection), {
    headers: {
      'content-type': 'application/geo+json',
      // Two caches, on purpose. The browser holds it five minutes so a reload
      // does not re-fetch. The CDN holds it at the edge for an hour and serves a
      // stale copy for a day while it refreshes in the background, so the
      // function itself runs at most once an hour per edge node rather than once
      // per visitor — the observations only change when the scheduled refresh
      // writes a new blob, so an hour-stale copy is never wrong enough to matter.
      'cache-control': 'public, max-age=300',
      'netlify-cdn-cache-control': 'public, s-maxage=3600, stale-while-revalidate=86400, durable',
    },
  })
}
