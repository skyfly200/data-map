// What fetch-species and gbif-fetch do with a fetch once they have it: store
// the file, register it as a dataset, and queue its enrichment.
//
// A fetch narrowed by place or date is the caller's own selection, so it gets
// its own file under species/<uid>/. Only an unnarrowed fetch writes the shared
// species/<slug>.geojson and the public manifest entry; before this split, one
// member's date-limited fetch silently replaced the shared file, and with it
// the contents of every dataset row pointing at it.

import { DatasetAccessError, checkVisibility, nextFreeSlug, slugify } from './dataset-access.mjs'
import { atLeast, effectiveTier } from './tiers.mjs'

/** Params that narrow a fetch to one caller's selection. */
export const NARROWING_PARAMS = ['lat', 'lng', 'radius', 'd1', 'd2']

export function isNarrowedFetch(searchParams) {
  return NARROWING_PARAMS.some((k) => (searchParams.get(k) || '').trim() !== '')
}

/** Storage key for a fetch. Shared only when unnarrowed or nobody is signed in. */
export function fetchedPath({ slug, userId = null, narrowed = false, now = Date.now() }) {
  if (narrowed && userId) return `species/${userId}/${slug}-${now}.geojson`
  return `species/${slug}.geojson`
}

/** Paths save_fetched will register: under species/, no traversal. */
export function isFetchedPath(path) {
  const p = String(path || '')
  return p.startsWith('species/') && !p.split('/').some((seg) => seg === '..' || seg === '.' || seg === '')
}

export const MAX_DATASETS_PER_MEMBER = 100

/**
 * Register a stored fetch in saved_datasets for the viewer, or return the row
 * they already have for that path (with its count refreshed, since a shared
 * file may have been re-fetched since).
 */
export async function registerFetched(client, viewer, body) {
  const path = String(body.path || '').trim()
  const incomingSlug = String(body.slug || '').trim()
  if (!path) throw new DatasetAccessError('path is required.', { status: 400, code: 'no_path' })
  if (!isFetchedPath(path)) {
    throw new DatasetAccessError('Only species/ paths may be registered this way.', { status: 400, code: 'bad_path' })
  }

  const FIELDS = 'id, owner_id, job_id, slug, title, description, path, visibility, '
    + 'feature_count, bytes, created_at, updated_at'
  const title = String(body.title || incomingSlug || 'Fetched observations').trim().slice(0, 200)

  const { data: existing } = await client.from('saved_datasets')
    .select(FIELDS).eq('owner_id', viewer.userId).eq('path', path).maybeSingle()
  if (existing) {
    if (body.feature_count != null && body.feature_count !== existing.feature_count) {
      const { data: updated } = await client.from('saved_datasets')
        .update({ feature_count: body.feature_count, title }).eq('id', existing.id).select(FIELDS).single()
      if (updated) return { dataset: updated, status: 'existing' }
    }
    return { dataset: existing, status: 'existing' }
  }

  const { count, error: countErr } = await client.from('saved_datasets')
    .select('id', { count: 'exact', head: true }).eq('owner_id', viewer.userId)
  if (countErr) throw new Error(countErr.message)
  if ((count || 0) >= MAX_DATASETS_PER_MEMBER) {
    throw new DatasetAccessError(
      `You have ${count} saved datasets, which is the limit. Delete one to save another.`,
      { status: 409, code: 'too_many' })
  }

  const visibility = checkVisibility(body.visibility, viewer)
  const base = slugify(incomingSlug || title)
  const { data: clashes } = await client.from('saved_datasets')
    .select('slug').like('slug', `${base}%`)
  const slug = nextFreeSlug(base, (clashes || []).map((r) => r.slug))

  const { data, error } = await client.from('saved_datasets').insert({
    owner_id: viewer.userId,
    slug,
    title,
    description: String(body.description || '').slice(0, 2000) || null,
    path,
    visibility,
    feature_count: body.feature_count ?? null,
  }).select(FIELDS).single()
  if (error) throw new Error(error.message)
  return { dataset: data, status: 'saved' }
}

/**
 * Register the fetch for a member and queue its enrichment against that row.
 *
 * The job names the registered dataset's slug. It used to name the species
 * slug, which is a file name and not a dataset: the lookup failed (and the
 * failure was swallowed), or found an unrelated dataset that happened to share
 * the name. Free accounts get neither; registering is a membership benefit.
 */
export async function registerAndEnrich({
  auth, profile, client, storagePath, slug, title, count,
  submitJob, measureSource, viewerFrom, poke,
}) {
  if (!auth?.user || !client) return null
  if (!atLeast(effectiveTier(profile), 'member')) return null
  const viewer = { ...viewerFrom(auth), tier: effectiveTier(profile) }
  const { dataset } = await registerFetched(client, viewer, {
    path: storagePath, slug, title, feature_count: count,
  })
  const { job } = await submitJob({
    user: auth.user,
    profile,
    spec: { kind: 'enrich', source: { type: 'dataset', slug: dataset.slug }, title: `Enrich ${title}` },
    counter: (spec) => measureSource(spec, { client, viewer }),
  })
  if (job?.id && poke) poke()
  return { dataset, job }
}
