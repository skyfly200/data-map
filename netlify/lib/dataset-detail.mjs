// What a saved dataset is made of: the settings that produced it and what it
// has been enriched with since.
//
// A dataset row on its own says little — a title, a path, a count. The rest is
// spread over other rows: the job that produced it (its spec and result_meta),
// the dataset that job read from, and the enrichment jobs run over it since.
// This gathers those into one answer for the detail view, and decides what of
// each a viewer is shown.
//
// The settings of a job are shown to anyone who may read the dataset it made:
// they describe the dataset, not the member. What is internal to the worker
// (checkpoints, retry counts) is dropped, and a parent dataset the viewer may
// not read is reported as hidden rather than named — the same reason a
// forbidden slug and a missing one get the same 404 in dataset-access.mjs.

import { canRead } from './dataset-access.mjs'
import { STAGES } from './ee-pipeline.mjs'

export const ORIGINS = {
  job: 'Earth Engine job',
  inat: 'iNaturalist fetch',
  gbif: 'GBIF fetch',
  upload: 'File upload',
  gbif_archive: 'GBIF archive import',
  ee_asset: 'Earth Engine asset import',
}

// Each field, and how to make what was sent safe to store. Anything else in
// the input is dropped.
const str = (max) => (v) => (v == null || v === '' ? undefined : String(v).slice(0, max))
const num = (v) => (v == null || v === '' || !Number.isFinite(Number(v)) ? undefined : Number(v))
const date = (v) => (/^\d{4}-\d{2}-\d{2}/.test(String(v || '')) ? String(v).slice(0, 10) : undefined)
const PROVENANCE_FIELDS = {
  origin: (v) => (ORIGINS[v] ? v : undefined),
  taxon: str(200),
  date_from: date,
  date_to: date,
  lat: num,
  lng: num,
  radius_km: num,
  quality: str(30),
  file_name: str(200),
  format: str(20),
  records: num,
  skipped: num,
  media: num,
  dna: num,
  asset_path: str(300),
  fetched_at: str(40),
}

/** The whitelisted part of a provenance object, or null when nothing is left. */
export function cleanProvenance(input) {
  if (!input || typeof input !== 'object' || Array.isArray(input)) return null
  const out = {}
  for (const [key, clean] of Object.entries(PROVENANCE_FIELDS)) {
    const value = clean(input[key])
    if (value !== undefined) out[key] = value
  }
  return out.origin ? out : null
}

/**
 * The provenance of an iNat or GBIF fetch, from the query it actually ran:
 * the place is the effective one, server default included, since that circle
 * decided which records came back as much as anything the member chose.
 */
export function fetchProvenance({ origin, taxon, d1, d2, lat, lng, radius, quality, records, now = new Date() }) {
  return cleanProvenance({
    origin, taxon, date_from: d1, date_to: d2, lat, lng, radius_km: radius, quality, records,
    fetched_at: now.toISOString(),
  })
}

/**
 * Where a dataset came from when nothing was recorded — rows saved before
 * migration 019. Read off the path, which every writer chooses by origin.
 */
export function inferOrigin(row = {}) {
  if (row.job_id) return 'job'
  const path = String(row.path || '')
  if (path.startsWith('species/')) {
    return path.split('/').pop().startsWith('gbif-') ? 'gbif' : 'inat'
  }
  const desc = String(row.description || '')
  if (/^(DwC-A|CSV) import/.test(desc)) return 'gbif_archive'
  if (/Earth Engine asset/i.test(desc)) return 'ee_asset'
  if (path.startsWith('datasets/')) return 'upload'
  return null
}

// What the worker adds to a spec as it runs. Not a setting anybody chose.
const INTERNAL_PARAMS = new Set(['attempts', 'continuations', 'checkpoint_path', 'checkpoint_progress'])

/** A job's spec as settings: the worker's bookkeeping left out. */
export function jobSettings(params) {
  if (!params || typeof params !== 'object') return {}
  return Object.fromEntries(Object.entries(params).filter(([k]) => !INTERNAL_PARAMS.has(k)))
}

/** The stage catalogue the view labels columns with: every stage, in order. */
export function stageCatalog() {
  return Object.entries(STAGES).map(([key, s]) => ({
    key, label: s.label, asset: s.asset, scale: s.scale, perDate: s.perDate,
    bands: s.bands, description: s.description,
  }))
}

/** One job, as the detail view lists it. */
function summariseJob(job, savedAs = null) {
  if (!job) return null
  const settings = jobSettings(job.params)
  return {
    id: job.id,
    kind: job.kind,
    status: job.status,
    title: job.title || settings.title || '',
    created_at: job.created_at,
    started_at: job.started_at || null,
    finished_at: job.finished_at || null,
    cost_units: job.cost_units ?? null,
    settings,
    result: job.result_meta || null,
    saved_as: savedAs ? { slug: savedAs.slug, title: savedAs.title } : null,
  }
}

/**
 * Put the pieces together.
 *
 *   row          the dataset, already checked readable
 *   job          the job that produced it, or null
 *   parent       the dataset that job read from, or null
 *   enrichments  jobs since run over this dataset, the viewer's own
 *   derived      datasets saved from those jobs
 */
export function buildDetail({ row, job = null, parent = null, enrichments = [], derived = [], viewer }) {
  const source = job?.params?.source || null
  let from = null
  if (source?.type === 'dataset') {
    from = parent && canRead(parent, viewer)
      ? { slug: parent.slug, title: parent.title }
      : { hidden: true }
  }
  const savedFrom = new Map()
  for (const d of derived) if (d.job_id && canRead(d, viewer)) savedFrom.set(d.job_id, d)

  // The row without the storage path: it is how the server finds the file,
  // and nothing in the view needs it.
  const { path: _path, ...dataset } = row
  const origin = row.provenance?.origin || inferOrigin(row)
  return {
    dataset: { ...dataset, mine: Boolean(viewer?.userId && row.owner_id === viewer.userId) },
    origin,
    origin_label: ORIGINS[origin] || 'Unknown',
    provenance: row.provenance || null,
    inferred: !row.provenance,
    job: summariseJob(job),
    from,
    enrichments: enrichments.map((j) => summariseJob(j, savedFrom.get(j.id))),
    stages: stageCatalog(),
  }
}
