// A member's own Earth Engine credential.
//
//   GET    status only: { configured, client_email, project_id, ... }
//   POST   { key, project } validate against Earth Engine, then store encrypted
//   DELETE revoke
//
// The key is write-only: no response ever contains it, and it is never logged.

import { adminClient, requireMemberFresh } from '../lib/auth.mjs'
import {
  CredentialError, credentialEncryptionConfigured, encryptSecret, parseServiceAccountKey,
  publicSummary, validateProjectId,
} from '../lib/ee-credentials.mjs'
import { verifyCredential } from '../lib/ee-runner.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
})

const COLUMNS = 'client_email, project_id, validated_at, created_at'

export default async function handler(request) {
  const auth = await requireMemberFresh(request)
  if (!auth.ok) return auth.response
  const userId = auth.user?.id
  const client = adminClient()
  if (!client || !userId) return json({ ok: false, error: 'Supabase is not configured.' }, 503)

  try {
    if (request.method === 'GET') {
      const { data } = await client.from('member_ee_credentials').select(COLUMNS).eq('user_id', userId).maybeSingle()
      return json({ ok: true, ...publicSummary(data), storage_available: credentialEncryptionConfigured() })
    }

    if (request.method === 'DELETE') {
      const { error } = await client.from('member_ee_credentials').delete().eq('user_id', userId)
      if (error) throw new CredentialError('Could not remove the credential.', { status: 500 })
      return json({ ok: true, configured: false })
    }

    if (request.method !== 'POST') return json({ ok: false, error: 'Use GET, POST or DELETE.' }, 405)

    let body
    try { body = await request.json() } catch { return json({ ok: false, error: 'Send JSON.' }, 400) }

    const key = parseServiceAccountKey(body?.key)
    const project = validateProjectId(body?.project || key.project_id)
    // Round-trip against Earth Engine before storing anything.
    await verifyCredential({ key, project })
    const ciphertext = encryptSecret(JSON.stringify(key), userId)

    const { data, error } = await client.from('member_ee_credentials').upsert({
      user_id: userId, ciphertext, client_email: key.client_email, project_id: project,
      validated_at: new Date().toISOString(),
    }).select(COLUMNS).single()
    if (error) throw new CredentialError('Could not store the credential.', { status: 500 })
    return json({ ok: true, ...publicSummary(data) })
  } catch (err) {
    if (err instanceof CredentialError) return json({ ok: false, error: err.message }, err.status)
    // Generic: an unexpected error may embed request material.
    console.error('[ee-credentials] failed:', err?.name || 'Error')
    return json({ ok: false, error: 'Could not process the credential.' }, 500)
  }
}
