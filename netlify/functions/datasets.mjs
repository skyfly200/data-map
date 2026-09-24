// A member's own datasets: naming a job result so it can be used again.
//
//   GET  /.netlify/functions/datasets              the caller's datasets
//   GET  /.netlify/functions/datasets?available=1  everything they may read
//   GET  /.netlify/functions/datasets?slug=x       download the GeoJSON
//   POST { action: 'save',   job_id, title, description?, visibility? }
//   POST { action: 'update', id, title?, description?, visibility? }
//   POST { action: 'delete', id }
//
// This is the step that makes jobs compose. A finished job is a file at a path
// only its owner can reach; naming it gives it an identity another job can
// reference, which is the difference between running a pipeline and building
// something out of one.
//
// Private by default, and the default is the whole policy: a member's results
// are their own work, and nothing here publishes anything unless they say so.
// `public` is not theirs to choose — see MEMBER_VISIBILITIES.

import { adminClient, requireMemberFresh, requireUser } from '../lib/auth.mjs'
import {
  DEFAULT_VISIBILITY, DatasetAccessError,
  canWrite, checkVisibility, nextFreeSlug, resolveDataset, slugify, viewerFrom,
} from '../lib/dataset-access.mjs'
import { readJson, uploadJson } from '../lib/datasets-store.mjs'
import { loadEeAsset } from '../lib/ee-assets.mjs'

export const config = { timeout: 30 }

/** As many datasets as one member may keep. Rows are cheap and these point at
 *  files that already exist, so this is a guard against runaway automation
 *  rather than a meaningful limit on anybody's work. */
export const MAX_DATASETS_PER_MEMBER = 100

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

const FIELDS = 'id, owner_id, job_id, slug, title, description, path, visibility, '
  + 'feature_count, bytes, created_at, updated_at'

function fail(err) {
  if (err instanceof DatasetAccessError) {
    return json({ ok: false, error: err.message, code: err.code }, err.status)
  }
  return json({ ok: false, error: String(err?.message || err) }, 500)
}

/** The caller's own datasets, newest first. */
async function listMine(client, viewer) {
  const { data, error } = await client.from('saved_datasets')
    .select(FIELDS).eq('owner_id', viewer.userId).order('created_at', { ascending: false })
  if (error) throw new Error(error.message)
  return json({ ok: true, datasets: data || [], max: MAX_DATASETS_PER_MEMBER })
}

/**
 * Everything the caller may run a job over.
 *
 * Their own, plus what has been shared with them. Filtered in SQL by the same
 * rule canRead() applies, because listing is the one place that asks the
 * question about many rows at once and doing it in JavaScript would mean
 * fetching every dataset on the deployment first.
 */
async function listAvailable(client, viewer) {
  let query = client.from('saved_datasets').select(FIELDS).order('title')
  if (viewer.tier !== 'admin') {
    const clauses = ['visibility.eq.public']
    if (viewer.userId) clauses.push(`owner_id.eq.${viewer.userId}`)
    if (viewer.tier === 'member' || viewer.tier === 'perpetual') clauses.push('visibility.eq.members')
    query = query.or(clauses.join(','))
  }
  const { data, error } = await query
  if (error) throw new Error(error.message)
  return json({ ok: true, datasets: data || [] })
}

/**
 * Download one.
 *
 * Served through here rather than straight from storage because the bucket
 * policy only knows about path prefixes: it can tell that a file is under
 * jobs/<uid>/ and cannot tell that its owner shared it. Sharing lives in the
 * row, so anything shared is read by a caller that has looked at the row.
 */
async function download(client, viewer, slug) {
  const row = await resolveDataset({ client, slug, viewer })
  const data = await readJson(row.path)
  if (!data) {
    throw new DatasetAccessError(`"${row.slug}" is registered but its file could not be read.`,
      { status: 500, code: 'no_file' })
  }
  return json({ ok: true, dataset: row, geojson: data })
}

/** Name a finished job of the caller's, so other jobs can use it. */
async function save(client, viewer, body) {
  const jobId = String(body.job_id || '').trim()
  if (!jobId) throw new DatasetAccessError('Which job?', { status: 400, code: 'no_job' })

  const { data: job, error: jobErr } = await client
    .from('ee_jobs').select('id, user_id, status, result_path, result_meta, title')
    .eq('id', jobId).maybeSingle()
  if (jobErr) throw new Error(jobErr.message)

  // Absent and not-yours are the same answer, for the same reason as a slug:
  // job ids are opaque, but there is no reason to confirm one exists.
  if (!job || (job.user_id !== viewer.userId && viewer.tier !== 'admin')) {
    throw new DatasetAccessError('No job of yours has that id.', { status: 404, code: 'no_job' })
  }
  if (job.status !== 'succeeded' || !job.result_path) {
    throw new DatasetAccessError(
      `That job ${job.status === 'succeeded' ? 'produced no result file' : `is ${job.status}`}, `
      + 'so there is nothing to save yet.',
      { status: 409, code: 'not_finished' })
  }

  const { count, error: countErr } = await client.from('saved_datasets')
    .select('id', { count: 'exact', head: true }).eq('owner_id', viewer.userId)
  if (countErr) throw new Error(countErr.message)
  if ((count || 0) >= MAX_DATASETS_PER_MEMBER) {
    throw new DatasetAccessError(
      `You have ${count} saved datasets, which is the limit. Delete one to save another.`,
      { status: 409, code: 'too_many' })
  }

  const title = String(body.title || job.title || 'Untitled dataset').trim().slice(0, 200)
  const visibility = checkVisibility(body.visibility, viewer)

  // Slugs are unique across the table, so a title somebody else has used gets
  // the next free suffix rather than an error naming their dataset.
  const base = slugify(title)
  const { data: clashes } = await client.from('saved_datasets')
    .select('slug').like('slug', `${base}%`)
  const slug = nextFreeSlug(base, (clashes || []).map((r) => r.slug))

  const { data, error } = await client.from('saved_datasets').insert({
    owner_id: viewer.userId,
    job_id: job.id,
    slug,
    title,
    description: String(body.description || '').slice(0, 2000) || null,
    // Never from the request. The member names a job; the server decides which
    // file that is. See the note in migration 005 about why.
    path: job.result_path,
    visibility,
    feature_count: job.result_meta?.features ?? null,
  }).select(FIELDS).single()
  if (error) throw new Error(error.message)

  return json({ ok: true, dataset: data, status: 'saved' })
}

async function update(client, viewer, body) {
  const id = String(body.id || '').trim()
  if (!id) throw new DatasetAccessError('Which dataset?', { status: 400, code: 'no_id' })

  const { data: row, error: readErr } = await client
    .from('saved_datasets').select(FIELDS).eq('id', id).maybeSingle()
  if (readErr) throw new Error(readErr.message)
  if (!canWrite(row, viewer)) {
    throw new DatasetAccessError('No dataset of yours has that id.', { status: 404, code: 'no_dataset' })
  }

  const patch = {}
  if (body.title !== undefined) {
    const title = String(body.title).trim().slice(0, 200)
    if (!title) throw new DatasetAccessError('A dataset needs a name.', { status: 400, code: 'no_title' })
    patch.title = title
  }
  if (body.description !== undefined) {
    patch.description = String(body.description).slice(0, 2000) || null
  }
  if (body.visibility !== undefined) patch.visibility = checkVisibility(body.visibility, viewer)
  if (!Object.keys(patch).length) {
    throw new DatasetAccessError('Nothing to change.', { status: 400, code: 'no_change' })
  }

  // The slug is left alone on purpose: it is how other jobs name this dataset,
  // and renaming it would break a spec that already references it.
  const { data, error } = await client.from('saved_datasets')
    .update(patch).eq('id', id).select(FIELDS).single()
  if (error) throw new Error(error.message)
  return json({ ok: true, dataset: data })
}

async function remove(client, viewer, body) {
  const id = String(body.id || '').trim()
  if (!id) throw new DatasetAccessError('Which dataset?', { status: 400, code: 'no_id' })

  const { data: row, error: readErr } = await client
    .from('saved_datasets').select('id, owner_id, visibility').eq('id', id).maybeSingle()
  if (readErr) throw new Error(readErr.message)
  if (!canWrite(row, viewer)) {
    throw new DatasetAccessError('No dataset of yours has that id.', { status: 404, code: 'no_dataset' })
  }

  // The row goes; the file stays. It is still the job's result and the job row
  // still points at it, so deleting the file here would quietly empty a job
  // the member can still see in their history.
  const { error } = await client.from('saved_datasets').delete().eq('id', id)
  if (error) throw new Error(error.message)
  return json({ ok: true, deleted: id })
}

/** Import an Earth Engine asset as a dataset. */
async function importAsset(client, viewer, body) {
  const assetPath = String(body.asset_path || '').trim()
  if (!assetPath) {
    throw new DatasetAccessError('Provide an asset_path.', { status: 400, code: 'no_asset' })
  }

  const { count, error: countErr } = await client.from('saved_datasets')
    .select('id', { count: 'exact', head: true }).eq('owner_id', viewer.userId)
  if (countErr) throw new Error(countErr.message)
  if ((count || 0) >= MAX_DATASETS_PER_MEMBER) {
    throw new DatasetAccessError(
      `You have ${count} saved datasets, which is the limit. Delete one to save another.`,
      { status: 409, code: 'too_many' })
  }

  // Load the asset from Earth Engine. Server-side failures (missing credentials,
  // EE not configured) are 503; a bad path or a missing asset is the member's 400.
  let geojson
  try {
    geojson = await loadEeAsset(assetPath)
  } catch (err) {
    const serverSide = err.message?.includes('not configured')
      || err.message?.includes('SDK not installed')
    throw new DatasetAccessError(err.message, {
      status: serverSide ? 503 : 400,
      code: 'ee_error',
    })
  }

  // Validate we got features
  if (!geojson || !geojson.features || geojson.features.length === 0) {
    throw new DatasetAccessError(
      'Asset contains no features or could not be converted to GeoJSON.',
      { status: 400, code: 'no_features' })
  }

  const title = String(body.title || 'EE Asset Import').trim().slice(0, 200)
  const visibility = checkVisibility(body.visibility, viewer)

  // Generate slug
  const base = slugify(title)
  const { data: clashes } = await client.from('saved_datasets')
    .select('slug').like('slug', `${base}%`)
  const slug = nextFreeSlug(base, (clashes || []).map((r) => r.slug))

  // Store the GeoJSON file
  const timestamp = Date.now()
  const path = `datasets/${viewer.userId}/${slug}-${timestamp}.geojson`
  await uploadJson(path, geojson)

  // Calculate stats
  const featureCount = geojson.features.length
  const bytes = JSON.stringify(geojson).length

  // Create the dataset record
  const { data, error } = await client.from('saved_datasets').insert({
    owner_id: viewer.userId,
    job_id: null,  // Not from a job
    slug,
    title,
    description: String(body.description || `Imported from Earth Engine asset: ${assetPath}`).slice(0, 2000),
    path,
    visibility,
    feature_count: featureCount,
    bytes,
  }).select(FIELDS).single()
  if (error) throw new Error(error.message)

  return json({ ok: true, dataset: data, status: 'imported' })
}

export default async function handler(request) {
  const client = adminClient()
  if (!client) {
    return json({
      ok: false,
      error: 'Supabase is not configured on this deployment.',
      hint: 'Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.',
    }, 503)
  }

  try {
    if (request.method === 'GET') {
      // Reading only needs an account: a public dataset is readable by anyone
      // signed in, and the resolve step decides the rest.
      const auth = await requireUser(request)
      if (!auth.ok) return auth.response
      const viewer = viewerFrom(auth)
      const params = new URL(request.url).searchParams

      if (params.get('slug')) return await download(client, viewer, params.get('slug'))
      if (params.get('available') === '1') return await listAvailable(client, viewer)
      return await listMine(client, viewer)
    }

    if (request.method !== 'POST') return json({ ok: false, error: 'Use GET or POST.' }, 405)

    // Writing is a membership benefit, and read fresh rather than from the
    // token for the same reason submitting a job is: a lapsed membership
    // should stop working now, not at the next refresh.
    const auth = await requireMemberFresh(request)
    if (!auth.ok) return auth.response
    const viewer = viewerFrom(auth)

    let body
    try {
      body = await request.json()
    } catch {
      return json({ ok: false, error: 'Send JSON.' }, 400)
    }

    switch (body.action) {
      case 'save': return await save(client, viewer, body)
      case 'update': return await update(client, viewer, body)
      case 'delete': return await remove(client, viewer, body)
      case 'import_asset': return await importAsset(client, viewer, body)
      default:
        return json({
          ok: false,
          error: `Unknown action "${body.action ?? ''}".`,
          actions: ['save', 'update', 'delete', 'import_asset'],
          note: `Datasets are ${DEFAULT_VISIBILITY} unless you say otherwise.`,
        }, 400)
    }
  } catch (err) {
    return fail(err)
  }
}
