// Membership tiers, read from the access token.
//
// The tier is stamped into the JWT by the custom access token hook in
// supabase_migrations/002_membership_jobs_and_admin.sql, so authorizing a
// request costs no database round trip: the token has to be verified anyway,
// and the answer is already inside it.
//
// Everything here is pure, and deliberately so. The rules about who may do what
// are the part worth testing, and they should not need a live Supabase project
// or a signed token to exercise.
//
// One caveat that shapes how these are used: a claim is fixed for the life of a
// token. Promote someone to member and they feel it at their next refresh, an
// hour by default; demote someone and they keep the old tier just as long. That
// is acceptable for showing UI and gating reads, and not acceptable for
// anything that spends money, which is why submitting a job re-reads the
// profile row. See requireMember/requireAdmin in auth.mjs.

/** Ascending: each tier can do everything the ones before it can. */
export const TIERS = ['free', 'member', 'admin']

export const TIER_LABELS = {
  free: 'Free account',
  member: 'Society member',
  admin: 'Administrator',
}

/**
 * The payload of a JWT, without verifying it.
 *
 * Decoding is not validation and this must never be the only thing standing
 * between a request and a member-only resource. It is called on tokens that
 * verifyToken has already checked with Supabase Auth, which is what makes the
 * claims inside trustworthy.
 */
export function decodeJwtPayload(token) {
  if (typeof token !== 'string') return null
  const parts = token.split('.')
  if (parts.length !== 3) return null
  try {
    // JWTs use base64url, which differs from base64 in two characters and in
    // dropping the padding.
    const b64 = parts[1].replace(/-/g, '+').replace(/_/g, '/')
    const pad = b64.length % 4 ? '='.repeat(4 - (b64.length % 4)) : ''
    // atob and TextDecoder rather than Buffer: the same rules have to hold in
    // the browser, where the header decides what to show, and in the function,
    // where it decides what to allow. Two copies would eventually disagree.
    // atob gives one byte per character, so the bytes are recovered before
    // being read as UTF-8 — otherwise any non-ASCII claim decodes to mojibake.
    const bin = atob(b64 + pad)
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0))
    const payload = JSON.parse(new TextDecoder().decode(bytes))
    return payload && typeof payload === 'object' ? payload : null
  } catch {
    return null
  }
}

/**
 * The tier a set of claims carries.
 *
 * Anything unrecognised is 'free'. A tier arriving as something not on the list
 * is either a hook that has drifted from this file or someone trying it on;
 * both should land on the least privilege, not the most.
 */
export function tierFromClaims(claims) {
  const raw = claims?.app_metadata?.tier ?? claims?.tier
  return TIERS.includes(raw) ? raw : 'free'
}

/** The tier carried by a token. See decodeJwtPayload on why this is not a check. */
export function tierFromToken(token) {
  return tierFromClaims(decodeJwtPayload(token))
}

/** Does `tier` reach `required`? Admin reaches everything. */
export function atLeast(tier, required) {
  const have = TIERS.indexOf(TIERS.includes(tier) ? tier : 'free')
  const need = TIERS.indexOf(required)
  if (need < 0) return false
  return have >= need
}

/**
 * The tier a profile row entitles someone to, honouring expiry.
 *
 * The token hook applies the same rule when minting a claim; this is what the
 * spending paths use, where the row is read fresh and a lapsed membership must
 * be felt now rather than at the next refresh.
 */
export function effectiveTier(profile, now = new Date()) {
  const tier = TIERS.includes(profile?.tier) ? profile.tier : 'free'
  if (tier === 'free') return 'free'
  const until = profile?.member_until ? new Date(profile.member_until) : null
  if (until && Number.isFinite(until.getTime()) && until < now) return 'free'
  return tier
}
