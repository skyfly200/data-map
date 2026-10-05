// Member-supplied Earth Engine credentials (WANT-3).
//
// A member may store their own Google service-account key so their jobs run
// under their own Earth Engine project and budget. The key is a long-lived
// secret, so:
//   * it is encrypted at rest with AES-256-GCM under EE_CREDENTIAL_KEY, which
//     lives in the environment, never in the database next to the ciphertext;
//   * the owner's user id is bound in as additional authenticated data, so a
//     ciphertext copied onto another member's row fails to decrypt;
//   * the table has RLS on with no policies: only the service role reads it;
//   * nothing here returns the key. `publicSummary` is the only shape that
//     leaves the server, and it carries the service-account email and project.

import { createCipheriv, createDecipheriv, randomBytes } from 'node:crypto'

const ALGO = 'aes-256-gcm'
const VERSION = 'v1'

export class CredentialError extends Error {
  constructor(message, { status = 400 } = {}) {
    super(message)
    this.name = 'CredentialError'
    this.status = status
  }
}

export function credentialEncryptionConfigured() {
  try { masterKey(); return true } catch { return false }
}

function masterKey() {
  const raw = process.env.EE_CREDENTIAL_KEY || ''
  const key = Buffer.from(raw, 'base64')
  if (key.length !== 32) {
    throw new CredentialError(
      'Credential storage is not configured on this deployment (EE_CREDENTIAL_KEY must be 32 bytes, base64).',
      { status: 503 },
    )
  }
  return key
}

/** Encrypt a string for one owner. Output: `v1.<iv>.<tag>.<ciphertext>` (base64). */
export function encryptSecret(plaintext, userId) {
  if (!userId) throw new CredentialError('An owner is required.')
  const iv = randomBytes(12)
  const cipher = createCipheriv(ALGO, masterKey(), iv)
  cipher.setAAD(Buffer.from(String(userId)))
  const body = Buffer.concat([cipher.update(String(plaintext), 'utf8'), cipher.final()])
  return [VERSION, iv.toString('base64'), cipher.getAuthTag().toString('base64'), body.toString('base64')].join('.')
}

export function decryptSecret(blob, userId) {
  const [version, iv, tag, body] = String(blob || '').split('.')
  if (version !== VERSION || !iv || !tag || !body) throw new CredentialError('Stored credential is unreadable.', { status: 500 })
  try {
    const decipher = createDecipheriv(ALGO, masterKey(), Buffer.from(iv, 'base64'))
    decipher.setAAD(Buffer.from(String(userId)))
    decipher.setAuthTag(Buffer.from(tag, 'base64'))
    return Buffer.concat([decipher.update(Buffer.from(body, 'base64')), decipher.final()]).toString('utf8')
  } catch (err) {
    if (err instanceof CredentialError) throw err
    // Deliberately vague: do not echo anything derived from the ciphertext.
    throw new CredentialError('Stored credential could not be decrypted.', { status: 500 })
  }
}

/**
 * Check an uploaded key's shape and return it parsed. Error messages never
 * quote the input, since it may contain a private key.
 */
export function parseServiceAccountKey(input) {
  let key = input
  if (typeof input === 'string') {
    const text = input.trim().startsWith('{') ? input : Buffer.from(input, 'base64').toString('utf8')
    try { key = JSON.parse(text) } catch { throw new CredentialError('That is not valid service-account JSON.') }
  }
  if (!key || typeof key !== 'object' || key.type !== 'service_account'
    || typeof key.client_email !== 'string' || typeof key.private_key !== 'string'
    || !key.private_key.includes('PRIVATE KEY')) {
    throw new CredentialError('That does not look like a Google service-account key file.')
  }
  return key
}

export function validateProjectId(project) {
  const p = String(project || '').trim()
  // Cloud project ids: 6-30 chars, lowercase letters, digits, hyphens.
  if (!/^[a-z][a-z0-9-]{4,28}[a-z0-9]$/.test(p)) throw new CredentialError('That is not a valid Cloud project id.')
  return p
}

/** What the browser may see about a stored credential. Never the key. */
export function publicSummary(row) {
  if (!row) return { configured: false }
  return {
    configured: true,
    client_email: row.client_email,
    project_id: row.project_id,
    validated_at: row.validated_at,
    created_at: row.created_at,
  }
}

/** Does this member have a credential? Reads no secret material. */
export async function hasCredential(client, userId) {
  if (!client || !userId) return false
  const { data } = await client.from('member_ee_credentials').select('user_id').eq('user_id', userId).maybeSingle()
  return Boolean(data)
}

/** Decrypted `{ key, project }` for a job's owner, or null. Server-side only. */
export async function loadCredential(client, userId) {
  if (!client || !userId) return null
  const { data } = await client
    .from('member_ee_credentials').select('ciphertext, project_id').eq('user_id', userId).maybeSingle()
  if (!data) return null
  const key = parseServiceAccountKey(decryptSecret(data.ciphertext, userId))
  return { key, project: data.project_id, id: userId }
}
