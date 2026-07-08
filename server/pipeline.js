import { randomUUID } from 'crypto'
import pLimit from 'p-limit'
import db from './db.js'
import { fetchInatObservations } from '../lib/sources/inat.js'
import { fetchElevationBatch } from '../lib/sources/openElevation.js'
import { fetchWeatherAndPrecip } from '../lib/sources/openMeteo.js'
import { fetchNDVI } from '../lib/sources/ndvi.js'
import { clusterObservations } from '../lib/cluster.js'

const CLUSTER_FEATURES = ['ndvi', 'prcp_d0', 'prcp_d1', 'prcp_d2', 'prcp_d3', 'prcp_d4', 'prcp_d5', 'prcp_d6']

function updateDataset(id, fields) {
  const keys = Object.keys(fields)
  const setClause = keys.map(k => `${k} = @${k}`).join(', ')
  db.prepare(`UPDATE datasets SET ${setClause}, updated_at = datetime('now') WHERE id = @id`)
    .run({ ...fields, id })
}

// Runs the full 3-stage pipeline in-process, updating the dataset row as it
// advances so the frontend can poll status. Designed to be called fire-and-forget.
export async function runPipeline(datasetId) {
  try {
    // ── Stage 1: iNaturalist fetch ────────────────────────────────────────────
    updateDataset(datasetId, { status: 'running', stage: 'inat' })

    const { config: configJson } = db
      .prepare('SELECT config FROM datasets WHERE id = ?')
      .get(datasetId)
    const config = JSON.parse(configJson)

    const observations = await fetchInatObservations(config)

    const insert = db.prepare(`
      INSERT INTO observations
        (id, dataset_id, inat_uuid, taxon_name, lat, lon, observed_on, quality_grade, place_guess)
      VALUES
        (@id, @dataset_id, @inat_uuid, @taxon_name, @lat, @lon, @observed_on, @quality_grade, @place_guess)
      ON CONFLICT(inat_uuid, dataset_id) DO UPDATE SET
        taxon_name    = excluded.taxon_name,
        lat           = excluded.lat,
        lon           = excluded.lon,
        observed_on   = excluded.observed_on,
        quality_grade = excluded.quality_grade,
        place_guess   = excluded.place_guess
    `)
    const insertMany = db.transaction((rows) => {
      for (const o of rows) insert.run({ id: randomUUID(), dataset_id: datasetId, ...o })
    })
    insertMany(observations)

    updateDataset(datasetId, { observation_count: observations.length, stage: 'enrich' })

    // ── Stage 2: enrichment ───────────────────────────────────────────────────
    const obsRows = db
      .prepare('SELECT id, lat, lon, observed_on FROM observations WHERE dataset_id = ?')
      .all(datasetId)

    if (obsRows.length) {
      const elevations = await fetchElevationBatch(obsRows)

      const limit = pLimit(10)
      const enriched = await Promise.all(
        obsRows.map((obs, i) =>
          limit(async () => {
            const weather = await fetchWeatherAndPrecip(obs.lat, obs.lon, obs.observed_on)
            const ndvi = await fetchNDVI(obs.lat, obs.lon, obs.observed_on)
            return { id: obs.id, elevation: elevations[i] ?? null, ndvi, ...weather }
          })
        )
      )

      const update = db.prepare(`
        UPDATE observations SET
          elevation = @elevation, ndvi = @ndvi, temp_avg = @temp_avg,
          precip_total = @precip_total, wind_speed = @wind_speed,
          prcp_d0 = @prcp_d0, prcp_d1 = @prcp_d1, prcp_d2 = @prcp_d2, prcp_d3 = @prcp_d3,
          prcp_d4 = @prcp_d4, prcp_d5 = @prcp_d5, prcp_d6 = @prcp_d6
        WHERE id = @id
      `)
      const updateMany = db.transaction((rows) => { for (const r of rows) update.run(r) })
      updateMany(enriched)
    }

    updateDataset(datasetId, { stage: 'cluster' })

    // ── Stage 3: clustering ───────────────────────────────────────────────────
    const featureRows = db
      .prepare(`SELECT id, ${CLUSTER_FEATURES.join(', ')} FROM observations WHERE dataset_id = ?`)
      .all(datasetId)

    const k = config.n_clusters ?? 4
    const assignments = clusterObservations(featureRows, CLUSTER_FEATURES, k)

    if (assignments.length) {
      const setCluster = db.prepare('UPDATE observations SET cluster = ? WHERE id = ?')
      const setClusterMany = db.transaction((list) => {
        for (const a of list) setCluster.run(a.cluster, a.id)
      })
      setClusterMany(assignments)
    }

    updateDataset(datasetId, { status: 'complete', stage: 'done' })
  } catch (err) {
    console.error('[pipeline] error:', err)
    updateDataset(datasetId, { status: 'error', stage: 'error', error_message: String(err) })
  }
}
