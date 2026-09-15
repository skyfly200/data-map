// Granting, extending and revoking membership from outside the app.
//
// The caller this exists for is an automation: a PayPal payment on the FRMS
// website fires a webhook, which calls this to turn a payment into a member.
// Nobody is sitting at a browser, so there is no session to authorise against
// and no one to read an error message — which sets three requirements that
// shape everything here.
//
//   Idempotent. Payment processors retry. A webhook delivered twice must not
//   sell somebody two years for one payment, so a grant carries the
//   processor's own reference and a repeat returns the first outcome instead
//   of applying a second.
//
//   Survives arriving early. Payment happens before signup at least as often
//   as after it: somebody pays, then makes an account, possibly days later. A
//   grant for an email with no account is recorded rather than refused, and
//   applied when that email signs up.
//
//   Additive. Renewing before you lapse has to extend the term you already
//   have, not restart it from today — otherwise renewing early costs the
//   member the time they had left.
//
// The arithmetic lives here, pure, because that is the part that silently
// takes money for time it does not grant.

export class MembershipError extends Error {
  constructor(message, code = 'invalid') {
    super(message)
    this.code = code
  }
}

export const TIERS = ['free', 'member', 'admin']

/** The longest term one call may grant. A typo in an automation is otherwise
 *  a lifetime membership nobody meant to sell. */
export const MAX_MONTHS = 120

/**
 * `date` plus `months`, clamped to the end of the target month.
 *
 * JavaScript rolls an overflowing day into the next month, so 31 January plus
 * one month is 3 March. For a membership term that is a day or two of free
 * time each renewal and a date that drifts — and it lands on exactly the
 * people who signed up at the end of a long month.
 */
export function addMonths(date, months) {
  const d = new Date(date)
  const day = d.getUTCDate()
  const target = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth() + months, 1,
    d.getUTCHours(), d.getUTCMinutes(), d.getUTCSeconds(), d.getUTCMilliseconds()))
  // Day 0 of the following month is the last day of this one.
  const lastDay = new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0)).getUTCDate()
  target.setUTCDate(Math.min(day, lastDay))
  return target
}

/**
 * When a membership should end after this grant.
 *
 * Extends from whichever is later: now, or the expiry they already hold. A
 * member renewing with two months left gets those two months plus the new
 * term; a lapsed member starts from today rather than from the date they
 * lapsed, which would sell them time that has already passed.
 */
export function extendUntil({ current = null, months = 0, now = new Date() } = {}) {
  const held = current ? new Date(current) : null
  const from = held && Number.isFinite(held.getTime()) && held > now ? held : now
  return addMonths(from, months)
}

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/

/** Lower-cased and trimmed, which is how it has to be compared. */
export function normaliseEmail(raw) {
  const email = String(raw ?? '').trim().toLowerCase()
  if (!email) throw new MembershipError('An email address is required.', 'no_email')
  if (email.length > 320 || !EMAIL.test(email)) {
    throw new MembershipError(`“${raw}” is not an email address.`, 'bad_email')
  }
  return email
}

function isoOrNull(raw, field) {
  if (raw === undefined || raw === null || raw === '') return null
  const d = new Date(raw)
  if (!Number.isFinite(d.getTime())) throw new MembershipError(`${field} is not a date.`, 'bad_date')
  return d.toISOString()
}

/**
 * Validate what an automation sent.
 *
 * `months` and `until` are alternatives: months is the normal case ("they
 * bought a year"), `until` is for migrating a term that was decided elsewhere.
 * Sending both is a contradiction rather than a preference, so it is refused
 * instead of one silently winning.
 */
export function normaliseGrant(input = {}) {
  const email = normaliseEmail(input.email)

  const tier = input.tier === undefined || input.tier === '' ? 'member' : String(input.tier)
  if (!TIERS.includes(tier)) throw new MembershipError(`Unknown tier “${tier}”.`, 'bad_tier')
  if (tier === 'free') {
    throw new MembershipError('Use revoke to remove a membership, not a grant of “free”.', 'bad_tier')
  }

  const hasMonths = input.months !== undefined && input.months !== null && input.months !== ''
  const until = isoOrNull(input.until, 'until')
  if (hasMonths && until) {
    throw new MembershipError('Send months or until, not both.', 'ambiguous_term')
  }

  let months = 0
  if (hasMonths) {
    months = Math.floor(Number(input.months))
    if (!Number.isFinite(months) || months < 1) {
      throw new MembershipError('months must be a whole number of at least 1.', 'bad_months')
    }
    if (months > MAX_MONTHS) {
      throw new MembershipError(`months may not exceed ${MAX_MONTHS}.`, 'bad_months')
    }
  } else if (!until) {
    // Neither: default to a year, which is what a society subscription is.
    months = 12
  }

  // The payment processor's own id. Optional, but without one a retried
  // webhook is indistinguishable from a second payment.
  const ref = input.ref === undefined || input.ref === null || input.ref === ''
    ? null
    : String(input.ref).trim().slice(0, 200)

  return {
    email,
    tier,
    months: until ? null : months,
    until,
    ref,
    source: String(input.source ?? 'api').trim().slice(0, 60) || 'api',
    note: input.note === undefined || input.note === null ? null : String(input.note).slice(0, 2000),
    display_name: input.name === undefined || input.name === null
      ? null
      : String(input.name).trim().slice(0, 200) || null,
  }
}

/**
 * The profile patch a grant implies.
 *
 * Deliberately does NOT demote an administrator to member. An admin who
 * renews their own dues through the same PayPal button as everybody else
 * should not lose the admin screen for doing it.
 */
export function applyGrant({ grant, profile = null, now = new Date() } = {}) {
  const memberUntil = grant.until
    ? new Date(grant.until)
    : extendUntil({ current: profile?.member_until, months: grant.months, now })

  const tier = profile?.tier === 'admin' ? 'admin' : grant.tier

  const patch = { tier, member_until: memberUntil.toISOString() }
  if (grant.display_name && !profile?.display_name) patch.display_name = grant.display_name
  return patch
}

/** What a member's standing is right now, for a lookup. */
export function describeMembership(profile, now = new Date()) {
  if (!profile) return { known: false, tier: 'free', active: false, member_until: null }
  const until = profile.member_until ? new Date(profile.member_until) : null
  const lapsed = !!until && Number.isFinite(until.getTime()) && until <= now
  const tier = lapsed && profile.tier !== 'admin' ? 'free' : profile.tier
  return {
    known: true,
    tier,
    stored_tier: profile.tier,
    active: tier === 'member' || tier === 'admin',
    lapsed,
    member_until: profile.member_until ?? null,
    days_left: until && !lapsed
      ? Math.ceil((until.getTime() - now.getTime()) / 86400000)
      : 0,
  }
}
