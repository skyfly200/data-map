/* Turning a payment into a membership.
 *
 * The caller is an automation with nobody watching it, so the failures that
 * matter are the quiet ones: a webhook retried and a year sold twice, a
 * renewal that throws away time already paid for, a month-end date that drifts
 * a little further every year.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  MAX_MONTHS, MembershipError,
  addMonths, applyGrant, describeMembership, extendUntil, normaliseEmail, normaliseGrant,
} from '../netlify/lib/membership.mjs'

const iso = (d) => new Date(d).toISOString()

// ── Month arithmetic ─────────────────────────────────────────────────────────

test('adding months lands on the same day of the month', () => {
  assert.equal(iso(addMonths(new Date('2026-03-15T12:00:00Z'), 1)), iso('2026-04-15T12:00:00Z'))
  assert.equal(iso(addMonths(new Date('2026-03-15T12:00:00Z'), 12)), iso('2027-03-15T12:00:00Z'))
})

test('a day that does not exist in the target month clamps to its end', () => {
  // JavaScript rolls 31 January + 1 month into 3 March. For a membership that
  // is a couple of days of free time per renewal, and a date that drifts — and
  // it lands on exactly the people who joined at the end of a long month.
  assert.equal(iso(addMonths(new Date('2026-01-31T00:00:00Z'), 1)), iso('2026-02-28T00:00:00Z'))
  assert.equal(iso(addMonths(new Date('2026-08-31T00:00:00Z'), 1)), iso('2026-09-30T00:00:00Z'))
  assert.equal(iso(addMonths(new Date('2026-05-31T00:00:00Z'), 1)), iso('2026-06-30T00:00:00Z'))
})

test('February in a leap year gets its 29th', () => {
  assert.equal(iso(addMonths(new Date('2028-01-31T00:00:00Z'), 1)), iso('2028-02-29T00:00:00Z'))
})

test('adding months crosses a year boundary', () => {
  assert.equal(iso(addMonths(new Date('2026-11-15T00:00:00Z'), 3)), iso('2027-02-15T00:00:00Z'))
})

test('the time of day is preserved, so a term does not creep', () => {
  const at = new Date('2026-03-15T08:30:45.123Z')
  assert.equal(iso(addMonths(at, 6)), iso('2026-09-15T08:30:45.123Z'))
})

// ── Extending a term ─────────────────────────────────────────────────────────

const NOW = new Date('2026-06-15T00:00:00Z')

test('renewing early adds to the time already held', () => {
  // The one that costs a member money if it is wrong: renewing with two months
  // left must not throw those two months away.
  const current = '2026-08-15T00:00:00Z'
  assert.equal(iso(extendUntil({ current, months: 12, now: NOW })), iso('2027-08-15T00:00:00Z'))
})

test('renewing after lapsing starts from today, not from when it lapsed', () => {
  // Extending from a date in the past would sell them time that has already
  // gone by — a year bought in June expiring the following March.
  const current = '2026-01-01T00:00:00Z'
  assert.equal(iso(extendUntil({ current, months: 12, now: NOW })), iso('2027-06-15T00:00:00Z'))
})

test('a first membership runs from today', () => {
  assert.equal(iso(extendUntil({ current: null, months: 12, now: NOW })), iso('2027-06-15T00:00:00Z'))
})

test('an unreadable stored date is treated as no membership rather than throwing', () => {
  assert.equal(iso(extendUntil({ current: 'not a date', months: 1, now: NOW })), iso('2026-07-15T00:00:00Z'))
})

test('renewals stack without drift', () => {
  let until = null
  for (let i = 0; i < 3; i += 1) {
    until = extendUntil({ current: until, months: 12, now: NOW }).toISOString()
  }
  assert.equal(until, iso('2029-06-15T00:00:00Z'))
})

// ── What an automation may send ──────────────────────────────────────────────

test('an email is required and has to look like one', () => {
  assert.throws(() => normaliseGrant({}), MembershipError)
  assert.throws(() => normaliseGrant({ email: 'not-an-email' }), MembershipError)
  assert.throws(() => normaliseGrant({ email: 'a@b' }), MembershipError)
  assert.equal(normaliseGrant({ email: 'A.Person@Example.ORG ' }).email, 'a.person@example.org')
})

test('emails are compared lower-cased, because the processor will not agree on case', () => {
  assert.equal(normaliseEmail('  Foo@Bar.com '), 'foo@bar.com')
})

test('a grant defaults to a year of membership', () => {
  const g = normaliseGrant({ email: 'a@b.org' })
  assert.equal(g.tier, 'member')
  assert.equal(g.months, 12)
  assert.equal(g.until, null)
})

test('months and until are alternatives, not a preference', () => {
  // Accepting both and picking one silently would let an automation think it
  // granted a date while the app granted a duration.
  assert.throws(() => normaliseGrant({ email: 'a@b.org', months: 12, until: '2027-01-01' }),
    MembershipError)
})

test('an absurd term is refused rather than sold', () => {
  assert.throws(() => normaliseGrant({ email: 'a@b.org', months: 0 }), MembershipError)
  assert.throws(() => normaliseGrant({ email: 'a@b.org', months: -1 }), MembershipError)
  assert.throws(() => normaliseGrant({ email: 'a@b.org', months: MAX_MONTHS + 1 }), MembershipError)
  assert.throws(() => normaliseGrant({ email: 'a@b.org', months: 'twelve' }), MembershipError)
  assert.equal(normaliseGrant({ email: 'a@b.org', months: MAX_MONTHS }).months, MAX_MONTHS)
})

test('granting “free” is refused, since that is a revocation wearing a grant', () => {
  assert.throws(() => normaliseGrant({ email: 'a@b.org', tier: 'free' }), MembershipError)
  assert.throws(() => normaliseGrant({ email: 'a@b.org', tier: 'superuser' }), MembershipError)
  assert.equal(normaliseGrant({ email: 'a@b.org', tier: 'admin' }).tier, 'admin')
})

test('a bad date is refused rather than becoming Invalid Date', () => {
  assert.throws(() => normaliseGrant({ email: 'a@b.org', until: 'next year' }), MembershipError)
  assert.equal(normaliseGrant({ email: 'a@b.org', until: '2027-01-01' }).until,
    iso('2027-01-01T00:00:00Z'))
})

test('the processor reference is carried through, trimmed and capped', () => {
  assert.equal(normaliseGrant({ email: 'a@b.org', ref: '  TXN-123 ' }).ref, 'TXN-123')
  assert.equal(normaliseGrant({ email: 'a@b.org' }).ref, null)
  assert.equal(normaliseGrant({ email: 'a@b.org', ref: 'x'.repeat(400) }).ref.length, 200)
})

test('the source defaults to api and is recorded', () => {
  assert.equal(normaliseGrant({ email: 'a@b.org' }).source, 'api')
  assert.equal(normaliseGrant({ email: 'a@b.org', source: 'paypal' }).source, 'paypal')
})

// ── Applying one ─────────────────────────────────────────────────────────────

test('a grant to a new member sets the tier and the expiry', () => {
  const grant = normaliseGrant({ email: 'a@b.org', months: 12 })
  const patch = applyGrant({ grant, profile: null, now: NOW })
  assert.equal(patch.tier, 'member')
  assert.equal(patch.member_until, iso('2027-06-15T00:00:00Z'))
})

test('a grant never demotes an administrator', () => {
  // An admin renewing their dues through the same PayPal button as everybody
  // else must not lose the admin screen for doing it.
  const grant = normaliseGrant({ email: 'a@b.org', months: 12 })
  const patch = applyGrant({ grant, profile: { tier: 'admin', member_until: null }, now: NOW })
  assert.equal(patch.tier, 'admin')
  assert.equal(patch.member_until, iso('2027-06-15T00:00:00Z'))
})

test('a grant extends an existing member rather than restarting them', () => {
  const grant = normaliseGrant({ email: 'a@b.org', months: 12 })
  const patch = applyGrant({
    grant, profile: { tier: 'member', member_until: '2026-09-01T00:00:00Z' }, now: NOW,
  })
  assert.equal(patch.member_until, iso('2027-09-01T00:00:00Z'))
})

test('an explicit until is taken as given', () => {
  const grant = normaliseGrant({ email: 'a@b.org', until: '2030-01-01' })
  const patch = applyGrant({ grant, profile: { member_until: '2026-09-01T00:00:00Z' }, now: NOW })
  assert.equal(patch.member_until, iso('2030-01-01T00:00:00Z'))
})

test('a name is filled in only when the profile has none', () => {
  const grant = normaliseGrant({ email: 'a@b.org', name: 'A Person' })
  assert.equal(applyGrant({ grant, profile: null, now: NOW }).display_name, 'A Person')
  // Overwriting would let a payment processor rename somebody who had already
  // chosen what to be called.
  assert.equal(
    applyGrant({ grant, profile: { display_name: 'Their Choice' }, now: NOW }).display_name,
    undefined,
  )
})

// ── Reading a membership back ────────────────────────────────────────────────

test('an unknown email is reported as unknown, not as lapsed', () => {
  const d = describeMembership(null, NOW)
  assert.equal(d.known, false)
  assert.equal(d.active, false)
  assert.equal(d.tier, 'free')
})

test('a current member reads as active with days remaining', () => {
  const d = describeMembership({ tier: 'member', member_until: '2026-07-15T00:00:00Z' }, NOW)
  assert.equal(d.active, true)
  assert.equal(d.lapsed, false)
  assert.equal(d.days_left, 30)
})

test('a lapsed member reads as free however the row is stored', () => {
  // The token hook treats a past date as free, so the API has to agree — an
  // automation asking "are they a member" must get the same answer the app
  // would give.
  const d = describeMembership({ tier: 'member', member_until: '2026-01-01T00:00:00Z' }, NOW)
  assert.equal(d.tier, 'free')
  assert.equal(d.stored_tier, 'member')
  assert.equal(d.active, false)
  assert.equal(d.lapsed, true)
  assert.equal(d.days_left, 0)
})

test('an admin stays an admin past the expiry date', () => {
  const d = describeMembership({ tier: 'admin', member_until: '2026-01-01T00:00:00Z' }, NOW)
  assert.equal(d.tier, 'admin')
  assert.equal(d.active, true)
})

test('no expiry means it does not lapse', () => {
  const d = describeMembership({ tier: 'member', member_until: null }, NOW)
  assert.equal(d.active, true)
  assert.equal(d.lapsed, false)
})
