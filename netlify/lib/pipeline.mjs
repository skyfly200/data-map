import { getStore } from '@netlify/blobs'
import { buildInatUrl, fetchInatFeatures } from './observations.mjs'

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

export function getRuntimeConfig(overrides = {}, request = null) {
  const params = request instanceof Request ? new URL(request.url).searchParams : new URLSearchParams()
  const species = overrides.species ?? params.get('species') ?? process.env.INAT_TAXON ?? process.env.SPECIES ?? 'morchella'
  const lat = Number(overrides.lat ?? params.get('lat') ?? process.env.INAT_LAT ?? 40.0)
  const lng = Number(overrides.lng ?? params.get('lng') ?? process.env.INAT_LNG ?? -105.0)
  const radius = Number(overrides.radius ?? params.get('radius') ?? process.env.INAT_RADIUS ?? process.env.RADIUS ?? 500)
  const perPage = Number(overrides.perPage ?? params.get('perPage') ?? process.env.INAT_PER_PAGE ?? process.env.PER_PAGE ?? 100)
  const qualityGrade = overrides.qualityGrade ?? params.get('qualityGrade') ?? process.env.INAT_QUALITY_GRADE ?? process.env.QUALITY_GRADE ?? 'research'

  return {
    species: parseSpeciesList(species),
    lat,
    lng,
    radius,
    perPage,
    qualityGrade,
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

export async function fetchObservationsStage(config, fetchImpl = fetch) {
  const species = config.species || ['morchella']
  const collected = []

  for (const taxonName of species) {
    const url = buildInatUrl({
      taxonName,
      lat: config.lat,
      lng: config.lng,
      radius: config.radius,
      perPage: config.perPage,
      qualityGrade: config.qualityGrade,
    })
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

  const unique = dedupeFeatures(collected)
  const collection = toFeatureCollection(unique)
  await writeStage('fetch', { count: unique.length, species, items: collection })

  return buildStageResult('fetch', {
    species,
    count: unique.length,
    source: 'inaturalist',
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
