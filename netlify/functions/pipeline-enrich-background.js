import { supabase } from './_shared/supabaseAdmin.js'
import { fetchElevationBatch } from '../../lib/sources/openElevation.js'
import { fetchWeatherAndPrecip } from '../../lib/sources/openMeteo.js'
import { fetchNDVI } from '../../lib/sources/ndvi.js'
import pLimit from 'p-limit'

export const handler = async (event) => {
  let dataset_id
  try {
    ;({ dataset_id } = JSON.parse(event.body))
  } catch {
    return { statusCode: 400, body: 'Invalid body' }
  }

  try {
    await supabase
      .from('datasets')
      .update({ stage: 'enrich' })
      .eq('id', dataset_id)

    const { data: observations, error: fetchErr } = await supabase
      .from('observations')
      .select('id, lat, lon, observed_on')
      .eq('dataset_id', dataset_id)

    if (fetchErr) throw fetchErr
    if (!observations.length) {
      return chainToCluster(dataset_id)
    }

    // ── Elevation (single batch POST) ────────────────────────────────────────
    const elevations = await fetchElevationBatch(observations)

    // ── Weather + precip (concurrent, capped at 10) ──────────────────────────
    const limit = pLimit(10)
    const weatherResults = await Promise.all(
      observations.map((obs, i) =>
        limit(async () => {
          const weather = await fetchWeatherAndPrecip(obs.lat, obs.lon, obs.observed_on)
          const ndvi = await fetchNDVI(obs.lat, obs.lon, obs.observed_on)
          return {
            id: obs.id,
            elevation: elevations[i] ?? null,
            ndvi,
            ...weather
          }
        })
      )
    )

    // ── Write enriched rows back (batches of 100) ────────────────────────────
    for (let i = 0; i < weatherResults.length; i += 100) {
      const batch = weatherResults.slice(i, i + 100)
      await Promise.all(
        batch.map(row => {
          const { id, ...fields } = row
          return supabase.from('observations').update(fields).eq('id', id)
        })
      )
    }

    return chainToCluster(dataset_id)
  } catch (err) {
    console.error('[pipeline-enrich] error:', err)
    await supabase
      .from('datasets')
      .update({ status: 'error', stage: 'error', error_message: String(err) })
      .eq('id', dataset_id)
    return { statusCode: 500, body: String(err) }
  }
}

async function chainToCluster(dataset_id) {
  await supabase
    .from('datasets')
    .update({ stage: 'cluster' })
    .eq('id', dataset_id)

  const baseUrl = process.env.URL || 'http://localhost:8888'
  await fetch(`${baseUrl}/.netlify/functions/pipeline-cluster-background`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_id })
  })

  return { statusCode: 202 }
}
