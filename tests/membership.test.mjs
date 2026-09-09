/* The rules about who may spend Earth Engine quota.
 *
 * These are worth testing precisely because the failure is quiet in both
 * directions: too strict and a paying member is locked out of the thing they
 * paid for, too loose and one runaway job spends the society's whole month.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  TIERS, atLeast, decodeJwtPayload, effectiveTier, tierFromClaims, tierFromToken,
} from '../netlify/lib/tiers.mjs'
import {
  CHUNK_SIZE, DEFAULT_LIMITS, checkQuota, estimateUnits, summariseUsage,
} from '../netlify/lib/quotas.mjs'

/** A JWT-shaped string. Unsigned: these functions decode, they do not verify. */
function token(payload) {
  const b64 = (o) => Buffer.from(JSON.stringify(o)).toString('base64')
    .replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
  return `${b64({ alg: 'HS256' })}.${b64(payload)}.signature`
}

// ── Reading the tier out of a token ──────────────────────────────────────────

test('the tier comes out of app_metadata', () => {
  assert.equal(tierFromToken(token({ sub: 'u1', app_metadata: { tier: 'member' } })), 'member')
  assert.equal(tierFromToken(token({ sub: 'u1', app_metadata: { tier: 'admin' } })), 'admin')
})

test('a token with no claim is a free account', () => {
  // This is the state before the access token hook is enabled in the Supabase
  // dashboard, so it has to degrade to the least privilege rather than throw.
  assert.equal(tierFromToken(token({ sub: 'u1' })), 'free')
  assert.equal(tierFromToken(token({ sub: 'u1', app_metadata: {} })), 'free')
})

test('a tier that is not one of ours is a free account', () => {
  for (const bad of ['superuser', 'ADMIN', '', null, 0, { tier: 'admin' }, ['admin']]) {
    assert.equal(tierFromClaims({ app_metadata: { tier: bad } }), 'free', `for ${JSON.stringify(bad)}`)
  }
})

test('a malformed token is a free account, not a crash', () => {
  for (const bad of ['', 'not.a.jwt.at.all', 'onlyonepart', 'a.b', null, undefined, 42, {}]) {
    assert.equal(tierFromToken(bad), 'free')
  }
  // Two dots but garbage in the middle: the shape passes, the decode must not.
  assert.equal(tierFromToken('aaa.!!!not-base64!!!.ccc'), 'free')
  assert.equal(decodeJwtPayload('aaa.eyJub3QifQ.ccc'), null)
})

test('base64url decodes where plain base64 would not', () => {
  // A payload whose base64 contains - and _ and needs padding. Getting this
  // wrong fails only for some tokens, which is the worst kind of wrong.
  const payload = { sub: 'u1', app_metadata: { tier: 'member' }, note: '???>>>~~~ ûñí' }
  assert.deepEqual(decodeJwtPayload(token(payload)), payload)
})

// ── Tier ordering ────────────────────────────────────────────────────────────

test('tiers are ordered and admin reaches everything', () => {
  assert.deepEqual(TIERS, ['free', 'member', 'admin'])
  assert.ok(atLeast('admin', 'member'))
  assert.ok(atLeast('admin', 'admin'))
  assert.ok(atLeast('member', 'member'))
  assert.ok(atLeast('member', 'free'))
  assert.ok(!atLeast('member', 'admin'))
  assert.ok(!atLeast('free', 'member'))
  assert.ok(!atLeast('nonsense', 'member'))
  assert.ok(!atLeast('admin', 'nonsense'))
})

// ── Expiry ───────────────────────────────────────────────────────────────────

test('a lapsed membership is a free account', () => {
  const now = new Date('2026-09-09T00:00:00Z')
  assert.equal(effectiveTier({ tier: 'member', member_until: '2026-09-08T00:00:00Z' }, now), 'free')
  assert.equal(effectiveTier({ tier: 'member', member_until: '2026-09-10T00:00:00Z' }, now), 'member')
  // No expiry set means it does not expire: a life member, or dues tracked
  // somewhere other than this column.
  assert.equal(effectiveTier({ tier: 'member', member_until: null }, now), 'member')
  // An admin whose membership lapsed loses the admin tier too. Administration
  // of the society is not separable from being in it.
  assert.equal(effectiveTier({ tier: 'admin', member_until: '2026-01-01T00:00:00Z' }, now), 'free')
})

test('a missing or unparseable profile is a free account', () => {
  assert.equal(effectiveTier(null), 'free')
  assert.equal(effectiveTier({}), 'free')
  assert.equal(effectiveTier({ tier: 'member', member_until: 'not a date' }), 'member')
})

// ── Cost estimation ──────────────────────────────────────────────────────────

const CATALOGUE = {
  elevation: { passes: 1 },
  rainfall7d: { passes: 7 },
  ndvi: { passes: 1 },
}

test('cost scales with points and with passes over the data', () => {
  // One chunk of points, one pass: the cheapest job there is.
  assert.equal(estimateUnits({ points: 10, stages: ['elevation'] }, CATALOGUE), 1)
  // A week of rainfall is seven images, so seven times the requests.
  assert.equal(estimateUnits({ points: 10, stages: ['rainfall7d'] }, CATALOGUE), 7)
  // Points round up into whole chunks: 501 points is two requests, not 1.002.
  assert.equal(estimateUnits({ points: CHUNK_SIZE + 1, stages: ['elevation'] }, CATALOGUE), 2)
  assert.equal(estimateUnits({ points: 1200, stages: ['elevation', 'ndvi'] }, CATALOGUE), 6)
})

test('an unknown stage costs nothing rather than throwing', () => {
  // The catalogue is the allowlist; a stage not on it never runs, so it must
  // not be billed either.
  assert.equal(estimateUnits({ points: 500, stages: ['made-up'] }, CATALOGUE), 0)
  assert.equal(estimateUnits({}, CATALOGUE), 0)
})

// ── The gate ─────────────────────────────────────────────────────────────────

const member = { tier: 'member', ...DEFAULT_LIMITS }

test('a free account cannot run jobs', () => {
  const got = checkQuota({ profile: { tier: 'free' }, estimate: 1 })
  assert.equal(got.ok, false)
  assert.equal(got.code, 'not_a_member')
})

test('a lapsed member is told it lapsed, not that they never joined', () => {
  const got = checkQuota({
    profile: { tier: 'member', member_until: '2020-01-01T00:00:00Z' },
    estimate: 1,
  })
  assert.equal(got.code, 'not_a_member')
  assert.match(got.message, /lapsed/)
})

test('a member within every limit may run', () => {
  const got = checkQuota({
    profile: member,
    usage: { unitsThisMonth: 10, jobsToday: 1 },
    running: 0,
    estimate: 20,
    points: 100,
  })
  assert.equal(got.ok, true)
  assert.equal(got.remaining, DEFAULT_LIMITS.ee_quota_monthly - 10 - 20)
})

test('the monthly quota counts the job being asked for, not just what is spent', () => {
  // 490 spent of 500, asking for 20: refusing this is the whole point. Checking
  // only what is already spent would let every member overshoot by one job.
  const got = checkQuota({
    profile: member,
    usage: { unitsThisMonth: 490 },
    estimate: 20,
  })
  assert.equal(got.ok, false)
  assert.equal(got.code, 'over_quota')
  assert.match(got.message, /10 left/)
})

test('landing exactly on the quota is allowed', () => {
  const got = checkQuota({ profile: member, usage: { unitsThisMonth: 480 }, estimate: 20 })
  assert.equal(got.ok, true)
  assert.equal(got.remaining, 0)
})

test('the other three limits each refuse with their own reason', () => {
  const over = (patch) => checkQuota({ profile: member, estimate: 1, ...patch })

  assert.equal(over({ points: DEFAULT_LIMITS.ee_max_points + 1 }).code, 'too_many_points')
  assert.equal(over({ running: 1 }).code, 'already_running')
  assert.equal(over({ usage: { jobsToday: DEFAULT_LIMITS.ee_jobs_per_day } }).code, 'daily_limit')
})

test('an admin is metered but not held to the per-member ceilings', () => {
  const admin = { tier: 'admin', ...DEFAULT_LIMITS }
  // Ceilings that divide a shared pool between members do not apply.
  assert.equal(checkQuota({ profile: admin, estimate: 1, running: 5 }).ok, true)
  assert.equal(checkQuota({ profile: admin, estimate: 1, points: 999999 }).ok, true)
  // The pool itself still does.
  assert.equal(checkQuota({ profile: admin, estimate: 999999 }).code, 'over_quota')
})

test('a raised quota takes effect immediately', () => {
  const usage = { unitsThisMonth: 490 }
  assert.equal(checkQuota({ profile: member, usage, estimate: 20 }).ok, false)
  const raised = { ...member, ee_quota_monthly: 2000 }
  assert.equal(checkQuota({ profile: raised, usage, estimate: 20 }).ok, true)
})

// ── Rolling usage up ─────────────────────────────────────────────────────────

test('usage counts this month and today, and only jobs that spent something', () => {
  const now = new Date('2026-09-09T12:00:00Z')
  const jobs = [
    { status: 'succeeded', created_at: '2026-09-09T09:00:00Z', cost_units: 5 },   // today
    { status: 'succeeded', created_at: '2026-09-02T09:00:00Z', cost_units: 7 },   // this month
    { status: 'succeeded', created_at: '2026-08-30T09:00:00Z', cost_units: 100 }, // last month
    { status: 'failed', created_at: '2026-09-09T10:00:00Z', cost_units: 3 },      // not counted
    { status: 'cancelled', created_at: '2026-09-09T10:00:00Z', cost_units: 3 },   // not counted
    { status: 'running', created_at: '2026-09-09T11:00:00Z', estimated_units: 4 },
    { status: 'queued', created_at: '2026-09-09T11:30:00Z', estimated_units: 9 },
  ]
  const got = summariseUsage(jobs, now)
  assert.equal(got.unitsThisMonth, 5 + 7 + 4)
  assert.equal(got.jobsToday, 2)          // the succeeded one and the running one
  assert.equal(got.running, 2)            // running + queued both occupy a slot
})

test('a job that broke does not cost the member their day', () => {
  const now = new Date('2026-09-09T12:00:00Z')
  const jobs = Array.from({ length: 30 }, () => (
    { status: 'failed', created_at: '2026-09-09T09:00:00Z', cost_units: 1 }
  ))
  assert.equal(summariseUsage(jobs, now).jobsToday, 0)
  assert.equal(checkQuota({ profile: member, usage: summariseUsage(jobs, now), estimate: 1 }).ok, true)
})

test('a job with an unreadable timestamp is skipped, not counted as now', () => {
  const got = summariseUsage([{ status: 'succeeded', created_at: 'nonsense', cost_units: 500 }])
  assert.equal(got.unitsThisMonth, 0)
})
