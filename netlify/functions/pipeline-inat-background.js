import { supabase } from './_shared/supabaseAdmin.js'
import { fetchInatObservations } from './_shared/inat.js'

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
      .update({ status: 'running', stage: 'inat' })
      .eq('id', dataset_id)

    const { data: dataset, error: fetchErr } = await supabase
      .from('datasets')
      .select('config')
      .eq('id', dataset_id)
      .single()

    if (fetchErr) throw fetchErr

    const observations = await fetchInatObservations(dataset.config)

    if (observations.length > 0) {
      const rows = observations.map(obs => ({ ...obs, dataset_id }))

      // Upsert in batches of 500 to avoid payload limits
      for (let i = 0; i < rows.length; i += 500) {
        const { error } = await supabase
          .from('observations')
          .upsert(rows.slice(i, i + 500), { onConflict: 'inat_uuid,dataset_id' })
        if (error) throw error
      }
    }

    await supabase
      .from('datasets')
      .update({ observation_count: observations.length, stage: 'enrich' })
      .eq('id', dataset_id)

    // Chain to enrich step
    const baseUrl = process.env.URL || 'http://localhost:8888'
    await fetch(`${baseUrl}/.netlify/functions/pipeline-enrich-background`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset_id })
    })

    return { statusCode: 202 }
  } catch (err) {
    console.error('[pipeline-inat] error:', err)
    await supabase
      .from('datasets')
      .update({ status: 'error', stage: 'error', error_message: String(err) })
      .eq('id', dataset_id)
    return { statusCode: 500, body: String(err) }
  }
}
