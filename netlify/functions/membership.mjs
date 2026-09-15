// The membership API, for automations rather than for people.
//
//   GET  ?email=…                      what standing does this address have
//   POST { action: 'grant',  email, months?|until?, tier?, ref?, source?, name?, note? }
//   POST { action: 'revoke', email, reason? }
//   POST { action: 'lookup', email }
//
// Authenticated with a shared key, not a session: the caller is a webhook from
// the FRMS website's PayPal automation, and there is no member signed in to
// authorise as. Send it as either header:
//
//   Authorization: Bearer <MEMBERSHIP_API_KEY>
//   X-API-Key: <MEMBERSHIP_API_KEY>
//
// Fails closed. With no key configured the endpoint refuses everything rather
// than falling open, because the one thing worse than an automation that
// cannot grant membership is an open endpoint that grants it to anyone.

import { createHash, timingSafeEqual } from 'node:crypto'

import { adminClient } from '../lib/auth.mjs'
import {
  MembershipError, applyGrant, describeMembership, normaliseEmail, normaliseGrant,
} from '../lib/membership.mjs'

export const config = { timeout: 30 }

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

/**
 * Constant-time key comparison.
 *
 * Digest both sides first so the comparison is over fixed-length buffers:
 * timingSafeEqual throws on a length mismatch, and catching that would leak
 * the key's length through which branch ran.
 */
function keyMatches(presented, expected) {
  if (!presented || !expected) return false
  const a = createHash('sha256').update(String(presented)).digest()
  const b = createHash('sha256').update(String(expected)).digest()
  return timingSafeEqual(a, b)
}

function authorise(request) {
  const expected = process.env.MEMBERSHIP_API_KEY
  if (!expected) {
    return json({
      ok: false,
      error: 'The membership API is not configured on this deployment.',
      hint: 'Set MEMBERSHIP_API_KEY in the environment to enable it.',
    }, 503)
  }
  const header = request.headers.get('authorization') || ''
  const bearer = header.toLowerCase().startsWith('bearer ') ? header.slice(7).trim() : ''
  const presented = bearer || request.headers.get('x-api-key') || ''
  if (!keyMatches(presented, expected)) {
    return json({ ok: false, error: 'Not authorised.' }, 401)
  }
  return null
}

/** The profile for an email, or null when nobody has signed up with it yet. */
async function profileFor(client, email) {
  const { data: userId, error } = await client.rpc('user_id_for_email', { p_email: email })
  if (error) throw new Error(`Could not look up that address: ${error.message}`)
  if (!userId) return { userId: null, profile: null }
  const { data, error: pErr } = await client
    .from('profiles').select('*').eq('user_id', userId).maybeSingle()
  if (pErr) throw new Error(pErr.message)
  return { userId, profile: data || null }
}

/**
 * Grant or extend a membership.
 *
 * Two outcomes, and the caller is told which: `applied` when the address has an
 * account and the profile was updated, `pending` when it does not and the grant
 * is waiting for them to sign up. A webhook that gets `pending` has not failed —
 * that is the normal shape of paying before making an account.
 */
async function grant(client, body) {
  const spec = normaliseGrant(body)

  // Idempotency first, before anything is written. A payment processor retries
  // until it gets a 2xx, and a delivery that timed out after succeeding looks
  // exactly like one that failed.
  if (spec.ref) {
    const { data: seen } = await client
      .from('membership_grants').select('*').eq('ref', spec.ref).maybeSingle()
    if (seen) {
      return json({
        ok: true,
        status: seen.applied_at ? 'applied' : 'pending',
        duplicate: true,
        email: seen.email,
        member_until: seen.member_until,
        ref: seen.ref,
        note: 'This reference was already processed; nothing was changed.',
      })
    }
  }

  const { userId, profile } = await profileFor(client, spec.email)

  const row = {
    email: spec.email,
    tier: spec.tier,
    months: spec.months,
    until: spec.until,
    ref: spec.ref,
    source: spec.source,
    note: spec.note,
  }

  if (!userId) {
    // Recorded, not refused. Applied by the signup trigger in migration 004
    // when this address makes an account.
    const { data, error } = await client.from('membership_grants').insert(row).select().single()
    if (error) {
      // A concurrent retry can lose the race to the unique index rather than to
      // the check above, which is the point of having the constraint at all.
      if (/duplicate key|unique/i.test(error.message)) {
        return json({ ok: true, status: 'pending', duplicate: true, email: spec.email, ref: spec.ref })
      }
      throw new Error(error.message)
    }
    return json({
      ok: true,
      status: 'pending',
      email: spec.email,
      grant_id: data.id,
      note: 'No account uses that address yet. The membership will be applied when they sign up.',
    })
  }

  const patch = applyGrant({ grant: spec, profile })
  const { error: upErr } = await client.from('profiles').update(patch).eq('user_id', userId)
  if (upErr) throw new Error(upErr.message)

  const { error: gErr } = await client.from('membership_grants').insert({
    ...row, user_id: userId, applied_at: new Date().toISOString(), member_until: patch.member_until,
  })
  // The membership is already granted; failing the call now would invite a
  // retry that grants it a second time. Reported instead of thrown.
  const recorded = !gErr

  return json({
    ok: true,
    status: 'applied',
    email: spec.email,
    tier: patch.tier,
    member_until: patch.member_until,
    recorded,
    ...(recorded ? {} : { warning: 'Membership applied, but the grant could not be recorded.' }),
  })
}

async function revoke(client, body) {
  const email = normaliseEmail(body.email)
  const { userId, profile } = await profileFor(client, email)
  if (!userId) {
    // Drop any pending grants too: revoking a membership somebody paid for and
    // never claimed should not leave it waiting to apply itself later.
    const { count } = await client.from('membership_grants')
      .delete({ count: 'exact' }).is('applied_at', null).eq('email', email)
    return json({ ok: true, status: 'unknown', email, pending_cleared: count || 0 })
  }
  if (profile?.tier === 'admin') {
    return json({
      ok: false,
      error: 'That address belongs to an administrator. Change the tier in the admin screen instead.',
    }, 409)
  }
  const { error } = await client.from('profiles')
    .update({ tier: 'free', member_until: null }).eq('user_id', userId)
  if (error) throw new Error(error.message)
  return json({ ok: true, status: 'revoked', email })
}

async function lookup(client, email) {
  const clean = normaliseEmail(email)
  const { userId, profile } = await profileFor(client, clean)
  const described = describeMembership(profile)
  const { data: pending } = await client.from('membership_grants')
    .select('id, months, until, source, created_at').eq('email', clean).is('applied_at', null)
  return json({
    ok: true,
    email: clean,
    has_account: !!userId,
    ...described,
    pending_grants: pending || [],
  })
}

export default async function handler(request) {
  const refused = authorise(request)
  if (refused) return refused

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
      return await lookup(client, new URL(request.url).searchParams.get('email'))
    }
    if (request.method !== 'POST') {
      return json({ ok: false, error: 'Use GET or POST.' }, 405)
    }

    let body
    try {
      body = await request.json()
    } catch {
      return json({ ok: false, error: 'Send JSON.' }, 400)
    }

    switch (body.action) {
      case 'grant': return await grant(client, body)
      case 'revoke': return await revoke(client, body)
      case 'lookup': return await lookup(client, body.email)
      default:
        return json({
          ok: false,
          error: `Unknown action “${body.action ?? ''}”.`,
          actions: ['grant', 'revoke', 'lookup'],
        }, 400)
    }
  } catch (err) {
    // A validation fault is the caller's to fix and its message says how; the
    // status separates the two so an automation can tell "my request was wrong"
    // from "try again later".
    if (err instanceof MembershipError) {
      return json({ ok: false, error: err.message, code: err.code }, 400)
    }
    return json({ ok: false, error: String(err?.message || err) }, 500)
  }
}
