// Server-side Supabase Auth gate for the API-calling Netlify functions
// (fetch-species, run-data-pipeline). These make outbound iNaturalist / Earth
// Engine / pipeline calls on demand, so they must not be callable anonymously.
//
// A request proves identity with a Supabase access token (JWT) in the
// Authorization header: `Authorization: Bearer <access_token>`. We validate it
// by asking Supabase Auth who the token belongs to (auth.getUser), which
// rejects expired, tampered, or foreign tokens.
//
// Enforcement is on whenever Supabase is configured (SUPABASE_URL present),
// which is the case for any real deployment. Local/unconfigured dev runs open
// so the app still works without credentials; set AUTH_DISABLED=true to force
// it open even when configured (e.g. a private preview), or AUTH_REQUIRED=true
// to fail closed and refuse traffic until Supabase is configured.

import { createClient } from '@supabase/supabase-js'
import { atLeast, effectiveTier, tierFromToken } from './tiers.mjs'

export function authEnforced() {
  if (String(process.env.AUTH_DISABLED).toLowerCase() === 'true') return false
  if (String(process.env.AUTH_REQUIRED).toLowerCase() === 'true') return true
  return Boolean(process.env.SUPABASE_URL
    && (process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY))
}

export function bearer(request) {
  const h = request.headers.get('authorization') || request.headers.get('Authorization') || ''
  const m = /^Bearer\s+(.+)$/i.exec(h.trim())
  return m ? m[1].trim() : null
}

// Validate the request's bearer token against Supabase Auth.
// Returns the user object on success, or null on any failure.
export async function verifyToken(token) {
  if (!token) return null
  const url = process.env.SUPABASE_URL
  const key = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!url || !key) return null
  try {
    const client = createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } })
    const { data, error } = await client.auth.getUser(token)
    if (error || !data?.user) return null
    return data.user
  } catch {
    return null
  }
}

// Guard for a function handler. Resolves to:
//   { ok: true, user }            authenticated (or auth not enforced → user null)
//   { ok: false, response }       a 401 Response to return immediately
export async function requireUser(request) {
  // Unenforced means no Supabase at all: a local dev run, where the whole app
  // is already open and there is nobody to be an administrator over. 'admin'
  // here lets that run reach the admin screens; it is not a tier anyone can
  // reach on a deployment, because authEnforced() is true wherever Supabase is
  // configured.
  if (!authEnforced()) return { ok: true, user: null, tier: 'admin', token: null }
  const token = bearer(request)
  const user = await verifyToken(token)
  if (!user) {
    return {
      ok: false,
      response: new Response(
        JSON.stringify({ ok: false, error: 'Sign in required. Include a Supabase access token as a Bearer token.' }),
        { status: 401, headers: { 'content-type': 'application/json', 'www-authenticate': 'Bearer' } },
      ),
    }
  }
  // The token has just been validated by Supabase Auth, so the claims inside it
  // — including the tier stamped on by the custom access token hook — can be
  // trusted without a second round trip.
  return { ok: true, user, token, tier: tierFromToken(token) }
}

function deny(status, error, extra = {}) {
  return new Response(JSON.stringify({ ok: false, error, ...extra }),
    { status, headers: { 'content-type': 'application/json' } })
}

/**
 * A gate on tier, answered from the token.
 *
 * Fast, and stale by up to one token refresh. Right for reads and for showing
 * work someone has already paid for; wrong on its own for anything that spends
 * Earth Engine quota, which must call loadProfile and judge the row instead.
 * See requireMemberFresh.
 */
export async function requireTier(request, required = 'member', { message = '' } = {}) {
  const auth = await requireUser(request)
  if (!auth.ok) return auth
  if (!authEnforced()) return auth
  if (!atLeast(auth.tier, required)) {
    // Callers pass their own message where the default would be wrong: a map
    // layer refused because of tier is not "running pipeline jobs", and being
    // told about a feature you were not using is how a refusal reads as a bug.
    const why = message || (required === 'admin'
      ? 'That is an administrator action.'
      : 'Running pipeline jobs is a membership benefit.')
    return { ok: false, response: deny(403, why, { tier: auth.tier, required }) }
  }
  return auth
}

export const requireMember = (request) => requireTier(request, 'member')
export const requireAdmin = (request) => requireTier(request, 'admin')

/** Service-role client, for reading rows the caller's own key must not reach. */
export function adminClient() {
  const url = process.env.SUPABASE_URL
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!url || !key) return null
  return createClient(url, key, { auth: { persistSession: false, autoRefreshToken: false } })
}

/** A member's profile row, read past RLS. Null when Supabase is unconfigured. */
export async function loadProfile(userId) {
  const client = adminClient()
  if (!client || !userId) return null
  const { data } = await client.from('profiles').select('*').eq('user_id', userId).maybeSingle()
  return data || null
}

/**
 * Membership checked against the database rather than the token.
 *
 * Used by the paths that spend quota. They have to read the profile for the
 * limits anyway, so insisting on a fresh answer costs nothing — and it closes
 * the hour-long window in which a cancelled membership still carries a valid
 * token saying otherwise.
 */
export async function requireMemberFresh(request) {
  const auth = await requireTier(request, 'member')
  if (!auth.ok) return auth
  if (!authEnforced()) return { ...auth, profile: null }

  const profile = await loadProfile(auth.user.id)
  // No profile row and Supabase configured means the trigger in migration 002
  // has not run. Failing open here would hand out unmetered Earth Engine.
  const tier = effectiveTier(profile)
  if (!atLeast(tier, 'member')) {
    return {
      ok: false,
      response: deny(403,
        profile?.member_until
          ? 'Your membership has lapsed. Renew it to run pipeline jobs.'
          : 'Running pipeline jobs is a membership benefit.',
        { tier }),
    }
  }
  return { ...auth, tier, profile }
}
