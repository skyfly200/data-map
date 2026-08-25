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

export function authEnforced() {
  if (String(process.env.AUTH_DISABLED).toLowerCase() === 'true') return false
  if (String(process.env.AUTH_REQUIRED).toLowerCase() === 'true') return true
  return Boolean(process.env.SUPABASE_URL
    && (process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY))
}

function bearer(request) {
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
  if (!authEnforced()) return { ok: true, user: null }
  const user = await verifyToken(bearer(request))
  if (!user) {
    return {
      ok: false,
      response: new Response(
        JSON.stringify({ ok: false, error: 'Sign in required. Include a Supabase access token as a Bearer token.' }),
        { status: 401, headers: { 'content-type': 'application/json', 'www-authenticate': 'Bearer' } },
      ),
    }
  }
  return { ok: true, user }
}
