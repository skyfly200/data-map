import { supabase } from './_shared/supabaseAdmin.js'
import { clusterObservations } from '../../lib/cluster.js'

// Features used for clustering. soil_moisture excluded until GEE/ERA5 is wired up.
const FEATURES = ['ndvi', 'prcp_d0', 'prcp_d1', 'prcp_d2', 'prcp_d3', 'prcp_d4', 'prcp_d5', 'prcp_d6']

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
      .update({ stage: 'cluster' })
      .eq('id', dataset_id)

    const { data: dataset, error: dsErr } = await supabase
      .from('datasets')
      .select('config')
      .eq('id', dataset_id)
      .single()
    if (dsErr) throw dsErr

    const n_clusters = dataset.config.n_clusters ?? 4

    const { data: observations, error: obsErr } = await supabase
      .from('observations')
      .select(`id, ${FEATURES.join(', ')}`)
      .eq('dataset_id', dataset_id)
    if (obsErr) throw obsErr

    const assignments = clusterObservations(observations, FEATURES, n_clusters)

    // ── Write cluster IDs back in batches ─────────────────────────────────────
    for (let i = 0; i < assignments.length; i += 100) {
      const batch = assignments.slice(i, i + 100)
      await Promise.all(
        batch.map(a =>
          supabase
            .from('observations')
            .update({ cluster: a.cluster })
            .eq('id', a.id)
        )
      )
    }

    await supabase
      .from('datasets')
      .update({ status: 'complete', stage: 'done' })
      .eq('id', dataset_id)

    return { statusCode: 200 }
  } catch (err) {
    console.error('[pipeline-cluster] error:', err)
    await supabase
      .from('datasets')
      .update({ status: 'error', stage: 'error', error_message: String(err) })
      .eq('id', dataset_id)
    return { statusCode: 500, body: String(err) }
  }
}
