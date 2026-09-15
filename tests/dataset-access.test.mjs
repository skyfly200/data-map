/* Who may read a saved dataset.
 *
 * These matter more than most: a dataset is a member's own work, saved private
 * by default, and the failures are all quiet ones. Nothing throws when a rule
 * is too loose — somebody else's data simply comes back.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  DEFAULT_VISIBILITY, DatasetAccessError, MEMBER_VISIBILITIES, VISIBILITIES,
  canRead, canWrite, checkVisibility, nextFreeSlug, resolveDataset, slugify, viewerFrom,
} from '../netlify/lib/dataset-access.mjs'

const OWNER = 'user-1'
const OTHER = 'user-2'

const mine = (visibility = 'private') => ({ owner_id: OWNER, visibility, slug: 's', path: 'p' })

const asFree = { userId: OTHER, tier: 'free' }
const asMember = { userId: OTHER, tier: 'member' }
const asPerpetual = { userId: OTHER, tier: 'perpetual' }
const asAdmin = { userId: OTHER, tier: 'admin' }
const asOwner = { userId: OWNER, tier: 'member' }
const anonymous = { userId: null, tier: 'free' }

// ── Reading ──────────────────────────────────────────────────────────────────

test('a private dataset is readable only by its owner and an admin', () => {
  assert.ok(canRead(mine(), asOwner))
  assert.ok(canRead(mine(), asAdmin))
  assert.ok(!canRead(mine(), asMember))
  assert.ok(!canRead(mine(), asFree))
  assert.ok(!canRead(mine(), anonymous))
})

test('private is the default a dataset gets when nobody chooses', () => {
  assert.equal(DEFAULT_VISIBILITY, 'private')
  assert.equal(checkVisibility(undefined, asOwner), 'private')
  assert.equal(checkVisibility('', asOwner), 'private')
  assert.equal(checkVisibility(null, asOwner), 'private')
})

test('a members dataset is readable by anyone whose membership is current', () => {
  assert.ok(canRead(mine('members'), asMember))
  assert.ok(canRead(mine('members'), asPerpetual))
  assert.ok(canRead(mine('members'), asAdmin))
  // The tier passed in is the effective one, so a lapsed member arrives as
  // 'free' and is refused here rather than needing its own date arithmetic.
  assert.ok(!canRead(mine('members'), asFree))
  assert.ok(!canRead(mine('members'), anonymous))
})

test('a public dataset is readable by anyone at all', () => {
  assert.ok(canRead(mine('public'), anonymous))
  assert.ok(canRead(mine('public'), asFree))
})

test('an ownerless row is still governed by its visibility', () => {
  // Datasets outlive their owner's account: owner_id is "on delete set null".
  const orphan = { owner_id: null, visibility: 'private' }
  assert.ok(!canRead(orphan, asOwner), 'a null owner must not match a null userId')
  assert.ok(canRead(orphan, asAdmin))
  assert.ok(canRead({ owner_id: null, visibility: 'public' }, anonymous))
})

test('a viewer with no id never matches an owner', () => {
  // The trap: null === null would make every signed-out viewer the owner of
  // every orphaned row.
  assert.ok(!canRead({ owner_id: null, visibility: 'private' }, anonymous))
  assert.ok(!canWrite({ owner_id: null, visibility: 'private' }, anonymous))
})

test('nothing is readable from a missing row', () => {
  assert.ok(!canRead(null, asAdmin))
  assert.ok(!canRead(undefined, asAdmin))
})

// ── Writing ──────────────────────────────────────────────────────────────────

test('shared with you is not yours', () => {
  // A members-visible dataset is readable by every member and deletable by
  // none of them but its owner.
  assert.ok(canRead(mine('members'), asMember))
  assert.ok(!canWrite(mine('members'), asMember))
  assert.ok(canWrite(mine('members'), asOwner))
  assert.ok(canWrite(mine('members'), asAdmin))
})

test('a public dataset is not writable by the public', () => {
  assert.ok(canRead(mine('public'), anonymous))
  assert.ok(!canWrite(mine('public'), anonymous))
  assert.ok(!canWrite(mine('public'), asMember))
})

// ── Choosing a visibility ────────────────────────────────────────────────────

test('a member may keep a dataset private or share it with members', () => {
  for (const v of MEMBER_VISIBILITIES) {
    assert.equal(checkVisibility(v, asOwner), v)
  }
})

test('a member may not publish to the open web, an admin may', () => {
  assert.throws(() => checkVisibility('public', asOwner), DatasetAccessError)
  assert.throws(() => checkVisibility('public', asMember), DatasetAccessError)
  assert.equal(checkVisibility('public', asAdmin), 'public')
})

test('an unknown visibility is refused rather than defaulted', () => {
  // Defaulting would be the dangerous direction: a typo'd "publik" silently
  // becoming private is fine, silently becoming public is not — but a caller
  // that meant something and got something else is worth an error either way.
  assert.throws(() => checkVisibility('everyone', asAdmin), DatasetAccessError)
  assert.throws(() => checkVisibility('publik', asAdmin), DatasetAccessError)
  assert.deepEqual(VISIBILITIES, ['private', 'members', 'public'])
})

// ── Resolving a slug ─────────────────────────────────────────────────────────

const clientReturning = (row) => ({
  from: () => ({
    select: () => ({
      eq: () => ({ maybeSingle: async () => ({ data: row, error: null }) }),
    }),
  }),
})

test('a slug resolves to the row when the viewer may read it', async () => {
  const row = { ...mine(), slug: 'autumn-2026' }
  const got = await resolveDataset({ client: clientReturning(row), slug: 'autumn-2026', viewer: asOwner })
  assert.equal(got.slug, 'autumn-2026')
})

test('a dataset you may not read is reported exactly like one that does not exist', async () => {
  // The whole point. Two different messages here would let anyone enumerate
  // other members' dataset names by guessing slugs.
  const hidden = await resolveDataset({
    client: clientReturning({ ...mine(), slug: 'secret' }), slug: 'secret', viewer: asMember,
  }).catch((e) => e)
  const absent = await resolveDataset({
    client: clientReturning(null), slug: 'secret', viewer: asMember,
  }).catch((e) => e)

  assert.ok(hidden instanceof DatasetAccessError)
  assert.ok(absent instanceof DatasetAccessError)
  assert.equal(hidden.message, absent.message)
  assert.equal(hidden.status, absent.status)
  assert.equal(hidden.code, absent.code)
})

test('an empty slug is refused without reaching the database', async () => {
  let touched = false
  const client = { from: () => { touched = true; return clientReturning(null).from() } }
  await assert.rejects(() => resolveDataset({ client, slug: '  ', viewer: asAdmin }),
    DatasetAccessError)
  assert.equal(touched, false)
})

test('no Supabase is a 503, not a refusal', async () => {
  // "Not configured" and "not allowed" are different problems with different
  // fixes, and an automation retrying on one should not retry on the other.
  const err = await resolveDataset({ client: null, slug: 'x', viewer: asAdmin }).catch((e) => e)
  assert.equal(err.status, 503)
})

// ── Slugs ────────────────────────────────────────────────────────────────────

test('a title becomes a slug', () => {
  assert.equal(slugify('Autumn 2026, Front Range'), 'autumn-2026-front-range')
  assert.equal(slugify('  Morels!  '), 'morels')
  assert.equal(slugify('—'), 'dataset')
  assert.equal(slugify(''), 'dataset')
  assert.equal(slugify('x'.repeat(200)).length, 60)
})

test('a taken slug gets the next free suffix rather than an error', () => {
  assert.equal(nextFreeSlug('autumn', []), 'autumn')
  assert.equal(nextFreeSlug('autumn', ['autumn']), 'autumn-2')
  assert.equal(nextFreeSlug('autumn', ['autumn', 'autumn-2', 'autumn-3']), 'autumn-4')
})

// ── Viewers ──────────────────────────────────────────────────────────────────

test('a viewer without a session is free and has no id', () => {
  assert.deepEqual(viewerFrom({}), { userId: null, tier: 'free' })
  assert.deepEqual(viewerFrom({ user: { id: 'u' }, tier: 'member' }), { userId: 'u', tier: 'member' })
})
