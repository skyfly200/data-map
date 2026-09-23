// Who may read a saved dataset, and who may change one.
//
// This is the one copy of that rule. It is applied in three places that would
// otherwise each invent their own: the download route, the job pipeline
// resolving a `dataset` source, and the worker re-resolving it when the job
// actually runs. Row-level security in migration 005 states the same rule a
// fourth time, in SQL, for callers that reach the table directly from a
// browser — those two have to agree, and the policy quotes this file.
//
// The rule itself is small. What is worth being careful about is the failure:
// a dataset the viewer may not read and a dataset that does not exist must be
// indistinguishable from outside. Slugs are short and guessable, and an API
// that says "you may not read that" for one and "no such dataset" for the
// other lets anybody enumerate what other members have saved, by name, without
// ever reading a row.

import { atLeast } from './tiers.mjs'

export class DatasetAccessError extends Error {
  constructor(message, { status = 403, code = 'forbidden' } = {}) {
    super(message)
    this.status = status
    this.code = code
  }
}

/** Every visibility the column allows. */
export const VISIBILITIES = ['private', 'members', 'public']

/**
 * The visibilities a member may choose for a dataset of their own.
 *
 * `public` is missing deliberately. It means the open web, with no account and
 * no membership, and publishing under the FRMS name is an FRMS decision rather
 * than one a member makes for themselves from a form. An admin can promote a
 * dataset from the admin screen.
 */
export const MEMBER_VISIBILITIES = ['private', 'members']

/** The visibility a dataset gets when nobody says otherwise. */
export const DEFAULT_VISIBILITY = 'private'

/**
 * A viewer, as the rules below need them.
 *
 * `tier` is the effective tier — lapsed already collapsed to 'free' — because
 * a membership that ran out should not still be opening members-only datasets.
 */
export function viewerFrom(auth = {}) {
  return { userId: auth.user?.id || null, tier: auth.tier || 'free' }
}

/** May this viewer read this row? Mirrors the select policy in migration 002. */
export function canRead(row, viewer = {}) {
  if (!row) return false
  if (viewer.tier === 'admin') return true
  if (row.owner_id && viewer.userId && row.owner_id === viewer.userId) return true
  if (row.visibility === 'public') return true
  if (row.visibility === 'members') return atLeast(viewer.tier || 'free', 'member')
  return false
}

/**
 * May this viewer change or delete this row?
 *
 * Narrower than reading: shared with you is not yours. A dataset visible to
 * every member would otherwise be deletable by every member.
 */
export function canWrite(row, viewer = {}) {
  if (!row) return false
  if (viewer.tier === 'admin') return true
  return Boolean(row.owner_id && viewer.userId && row.owner_id === viewer.userId)
}

/** The visibility this viewer is allowed to set. Throws if they are not. */
export function checkVisibility(visibility, viewer = {}) {
  if (visibility === undefined || visibility === null || visibility === '') {
    return DEFAULT_VISIBILITY
  }
  const want = String(visibility)
  if (!VISIBILITIES.includes(want)) {
    throw new DatasetAccessError(`Visibility must be one of ${VISIBILITIES.join(', ')}.`,
      { status: 400, code: 'bad_visibility' })
  }
  if (want === 'public' && viewer.tier !== 'admin') {
    throw new DatasetAccessError(
      'Only an administrator can publish a dataset to the open web. '
      + 'Share it with members instead, or ask an admin to publish it.',
      { status: 403, code: 'not_publishable' })
  }
  return want
}

/**
 * Turn a slug into a row the viewer is allowed to read.
 *
 * Reads with whatever client it is given. Callers pass the service client,
 * because the rule is applied here rather than by row-level security — the
 * worker has no session to be scoped by, and the job pipeline has to answer
 * for the member who submitted the job rather than for itself.
 */
export async function resolveDataset({ client, slug, viewer }) {
  // Same error for absent and forbidden. See the note at the top of the file:
  // telling them apart is a way to enumerate other members' dataset names.
  const refuse = () => {
    throw new DatasetAccessError(`No dataset named "${slug}" is available to you.`,
      { status: 404, code: 'no_dataset' })
  }

  if (!client) {
    throw new DatasetAccessError('Datasets need Supabase to be configured.',
      { status: 503, code: 'unconfigured' })
  }
  const clean = String(slug || '').trim()
  if (!clean) refuse()

  const { data, error } = await client
    .from('saved_datasets').select('*').eq('slug', clean).maybeSingle()
  if (error) {
    throw new DatasetAccessError(`Could not look that dataset up: ${error.message}`,
      { status: 500, code: 'lookup_failed' })
  }
  if (!canRead(data, viewer)) refuse()
  return data
}

/**
 * A URL-safe slug for a title, and the next free one if it is taken.
 *
 * Slugs are unique across the whole table, so two members naming a dataset
 * "Autumn 2026" collide. Refusing the second would be both unhelpful and a
 * disclosure — it says somebody else has one by that name. Suffixing instead
 * is silent and gives them something that works.
 */
export function slugify(title) {
  return String(title || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 60)
    || 'dataset'
}

export function nextFreeSlug(base, taken = []) {
  const used = new Set(taken)
  if (!used.has(base)) return base
  for (let n = 2; n < 1000; n += 1) {
    const candidate = `${base}-${n}`
    if (!used.has(candidate)) return candidate
  }
  // A thousand datasets called the same thing is not a case worth handling
  // gracefully, but it should not loop forever either.
  return `${base}-${Date.now().toString(36)}`
}
