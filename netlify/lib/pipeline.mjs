import { getStore } from '@netlify/blobs'
import { buildInatUrl, fetchInatFeatures } from './observations.mjs'
import { syncObservationsToSupabase } from './supabase.mjs'

export const STAGE_KEYS = {
  fetch: 'fetch-observations',
  enrich: 'enriched-observations',
  cluster: 'clustered-observations',
  export: 'exported-observations',
}

export function parseSpeciesList(species) {
  if (!species) return ['morchella']
  if (Array.isArray(species)) return species.filter(Boolean).map((value) => String(value).trim()).filter(Boolean)
  return String(species)
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
}

export function getRefreshAllFlag(env = process.env) {
  const raw = env.REFRESH_ALL ?? env.INAT_REFRESH_ALL ?? env.FULL_REFRESH ?? '0'
  const value = String(raw).trim().toLowerCase()
  return ['1', 'true', 'yes', 'y', 'on'].includes(value)
}

export function getRuntimeConfig(overrides = {}, request = null) {
  const params = request instanceof Request ? new URL(request.url).searchParams : new URLSearchParams()
  const species = overrides.species ?? params.get('species') ?? process.env.INAT_TAXON ?? process.env.SPECIES ?? 'morchella'
  const lat = Number(overrides.lat ?? params.get('lat') ?? process.env.INAT_LAT ?? 40.0)
  const lng = Number(overrides.lng ?? params.get('lng') ?? process.env.INAT_LNG ?? -105.0)
  const radius = Number(overrides.radius ?? params.get('radius') ?? process.env.INAT_RADIUS ?? process.env.RADIUS ?? 500)
  const perPage = Number(overrides.perPage ?? params.get('perPage') ?? process.env.INAT_PER_PAGE ?? process.env.PER_PAGE ?? 100)
  const qualityGrade = overrides.qualityGrade ?? params.get('qualityGrade') ?? process.env.INAT_QUALITY_GRADE ?? process.env.QUALITY_GRADE ?? 'research'
  const refreshAll = overrides.refreshAll ?? params.get('refreshAll') ?? getRefreshAllFlag()
  const syncToSupabase = overrides.syncToSupabase ?? params.get('syncToSupabase') ?? process.env.SUPABASE_SYNC ?? process.env.SYNC_TO_SUPABASE ?? '0'

  return {
    species: parseSpeciesList(species),
    lat,
    lng,
    radius,
    perPage,
    qualityGrade,
    refreshAll: Boolean(refreshAll),
    syncToSupabase: ['1', 'true', 'yes', 'y', 'on'].includes(String(syncToSupabase).trim().toLowerCase()),
  }
}

export async function pipelineStore() {
  return getStore('pipeline')
}

export async function writeStage(stageName, payload) {
  const store = await pipelineStore()
  await store.setJSON(STAGE_KEYS[stageName] ?? stageName, payload)
  return payload
}

export async function readStage(stageName) {
  const store = await pipelineStore()
  const key = STAGE_KEYS[stageName] ?? stageName
  try {
    return await store.get(key, { type: 'json' })
  } catch {
    return null
  }
}

export function dedupeFeatures(features) {
  const map = new Map()
  for (const feature of features || []) {
    const id = feature?.properties?.uuid ?? feature?.properties?.inat_id ?? feature?.properties?.species
    if (!id) continue
    map.set(String(id), feature)
  }
  return [...map.values()]
}

export function toFeatureCollection(features) {
  return { type: 'FeatureCollection', features: features || [] }
}

export function buildStageResult(stageName, payload) {
  const meta = payload ?? {}
  return {
    ok: true,
    stage: stageName,
    recordedAt: new Date().toISOString(),
    ...meta,
  }
}

export function filterExistingFeatures(freshFeatures, existingFeatures = []) {
  const seen = new Set(
    (existingFeatures || [])
      .map((feature) => feature?.properties?.inat_id ?? feature?.properties?.uuid)
      .filter(Boolean)
      .map(String),
  )

  return (freshFeatures || []).filter((feature) => {
    const id = feature?.properties?.inat_id ?? feature?.properties?.uuid
    if (!id) return true
    if (seen.has(String(id))) return false
    seen.add(String(id))
    return true
  })
}

export async function fetchObservationsStage(config, fetchImpl = fetch) {
  const species = config.species || ['morchella']
  const collected = []

  for (const taxonName of species) {
    const features = await fetchInatFeatures({
      taxonName,
      lat: config.lat,
      lng: config.lng,
      radius: config.radius,
      perPage: config.perPage,
      qualityGrade: config.qualityGrade,
    }, fetchImpl)
    collected.push(...features)
  }

  let unique = dedupeFeatures(collected)
  if (!config.refreshAll) {
    const existing = await readStage('fetch')
    const existingFeatures = existing?.items?.features ?? existing?.features ?? []
    unique = filterExistingFeatures(unique, existingFeatures)
  }

  const collection = toFeatureCollection(unique)
  await writeStage('fetch', { count: unique.length, species, items: collection, refreshAll: Boolean(config.refreshAll) })

  if (config.syncToSupabase) {
    const syncResult = await syncObservationsToSupabase(
      collection.features?.map((feature) => ({
        ...feature.properties,
        id: feature.properties?.inat_id ?? feature.properties?.uuid ?? null,
        uuid: feature.properties?.uuid ?? null,
        geojson: feature.geometry,
        taxon: { name: feature.properties?.species ?? species[0] ?? 'unknown' },
        place_guess: feature.properties?.location ?? null,
        observed_on: feature.properties?.date ?? null,
        quality_grade: 'research',
      })) ?? [],
      { env: process.env },
    )

    if (syncResult.ok) {
      console.log(`Supabase sync complete: ${syncResult.rowCount} rows upserted to ${syncResult.table}`)
    } else {
      console.warn(`Supabase sync skipped: ${syncResult.reason}`)
    }
  }

  return buildStageResult('fetch', {
    species,
    count: unique.length,
    source: 'inaturalist',
    refreshAll: Boolean(config.refreshAll),
    syncToSupabase: Boolean(config.syncToSupabase),
    featureCollection: collection,
  })
}

export function enrichFeature(feature) {
  const base = feature && typeof feature === 'object' ? { ...feature } : { type: 'Feature', properties: {} }
  const props = { ...(base.properties || {}) }
  const coords = Array.isArray(base.geometry?.coordinates) ? base.geometry.coordinates : [null, null]

  return {
    ...base,
    properties: {
      ...props,
      elevation: props.elevation ?? null,
      land_cover_label: props.land_cover_label ?? null,
      ndvi: props.ndvi ?? null,
      soil_moisture: props.soil_moisture ?? null,
      solar_exposure: props.solar_exposure ?? null,
      wind_exposure: props.wind_exposure ?? null,
      water_retention: props.water_retention ?? null,
      slope: props.slope ?? null,
      aspect: props.aspect ?? null,
      latitude: coords[1] ?? props.latitude ?? null,
      longitude: coords[0] ?? props.longitude ?? null,
      enriched_at: new Date().toISOString(),
      stage: 'enriched',
    },
  }
}

export async function enrichObservationsStage(config) {
  const staged = await readStage('fetch')
  const source = staged?.items?.features ?? staged?.features ?? []
  const enriched = source.map(enrichFeature)
  const collection = toFeatureCollection(enriched)
  await writeStage('enrich', { count: enriched.length, species: config.species, items: collection })

  return buildStageResult('enrich', {
    species: config.species,
    count: enriched.length,
    featureCollection: collection,
  })
}

export function clusterFeature(feature, index) {
  const props = { ...(feature.properties || {}) }
  const coords = Array.isArray(feature.geometry?.coordinates) ? feature.geometry.coordinates : [0, 0]
  const rawCluster = props.cluster ?? `${props.species ?? 'unknown'}-${Math.round((coords[1] ?? 0) * 10)}-${Math.round((coords[0] ?? 0) * 10)}`
  const clusterId = typeof rawCluster === 'number' ? rawCluster : `${rawCluster}`

  return {
    ...feature,
    properties: {
      ...props,
      cluster: index,
      cluster_key: clusterId,
      clustered_at: new Date().toISOString(),
      stage: 'clustered',
    },
  }
}

export async function clusterObservationsStage(config) {
  const staged = await readStage('enrich')
  const source = staged?.items?.features ?? staged?.features ?? []
  const clustered = source.map((feature, index) => clusterFeature(feature, index))
  const collection = toFeatureCollection(clustered)
  await writeStage('cluster', { count: clustered.length, species: config.species, items: collection })

  return buildStageResult('cluster', {
    species: config.species,
    count: clustered.length,
    featureCollection: collection,
  })
}

export async function exportObservationsStage(config) {
  const staged = await readStage('cluster')
  const source = staged?.items?.features ?? staged?.features ?? []
  const collection = toFeatureCollection(source)
  await writeStage('export', { count: source.length, species: config.species, items: collection })

  return buildStageResult('export', {
    species: config.species,
    count: source.length,
    featureCollection: collection,
  })
}

export async function runPipelineStages(config, fetchImpl = fetch) {
  const fetchResult = await fetchObservationsStage(config, fetchImpl)
  const enrichResult = await enrichObservationsStage(config)
  const clusterResult = await clusterObservationsStage(config)
  const exportResult = await exportObservationsStage(config)

  return {
    ok: true,
    stages: {
      fetch: fetchResult,
      enrich: enrichResult,
      cluster: clusterResult,
      export: exportResult,
    },
  }
}
