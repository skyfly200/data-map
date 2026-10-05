import test from 'node:test'
import assert from 'node:assert/strict'
import { randomBytes } from 'node:crypto'

import {
  CredentialError, decryptSecret, encryptSecret, parseServiceAccountKey, publicSummary, validateProjectId,
} from '../netlify/lib/ee-credentials.mjs'
import { checkQuota } from '../netlify/lib/quotas.mjs'

process.env.EE_CREDENTIAL_KEY = randomBytes(32).toString('base64')

const KEY = {
  type: 'service_account', client_email: 'sa@proj-123456.iam.gserviceaccount.com',
  private_key: '-----BEGIN PRIVATE KEY-----\nSECRETSECRET\n-----END PRIVATE KEY-----\n', project_id: 'proj-123456',
}

test('encrypt/decrypt round-trips and hides the plaintext', () => {
  const blob = encryptSecret(JSON.stringify(KEY), 'user-a')
  assert.ok(!blob.includes('SECRETSECRET'))
  assert.deepEqual(JSON.parse(decryptSecret(blob, 'user-a')), KEY)
})

test('ciphertext is bound to its owner', () => {
  const blob = encryptSecret('x', 'user-a')
  assert.throws(() => decryptSecret(blob, 'user-b'), CredentialError)
})

test('tampered ciphertext is rejected', () => {
  const parts = encryptSecret('hello', 'u').split('.')
  parts[3] = Buffer.from('tampered').toString('base64')
  assert.throws(() => decryptSecret(parts.join('.'), 'u'), CredentialError)
})

test('missing master key refuses to encrypt', () => {
  const saved = process.env.EE_CREDENTIAL_KEY
  delete process.env.EE_CREDENTIAL_KEY
  assert.throws(() => encryptSecret('x', 'u'), CredentialError)
  process.env.EE_CREDENTIAL_KEY = saved
})

test('key parsing accepts JSON, base64 and rejects junk without quoting it', () => {
  assert.equal(parseServiceAccountKey(KEY).client_email, KEY.client_email)
  assert.equal(parseServiceAccountKey(JSON.stringify(KEY)).client_email, KEY.client_email)
  assert.equal(parseServiceAccountKey(Buffer.from(JSON.stringify(KEY)).toString('base64')).client_email, KEY.client_email)
  assert.throws(() => parseServiceAccountKey({ ...KEY, type: 'authorized_user' }), (e) => !e.message.includes('SECRET'))
  assert.throws(() => parseServiceAccountKey('not json'), CredentialError)
})

test('project id validation', () => {
  assert.equal(validateProjectId(' my-project-1 '), 'my-project-1')
  assert.throws(() => validateProjectId('Bad Project'), CredentialError)
})

test('public summary never carries key material', () => {
  const s = publicSummary({ ciphertext: 'c', private_key: 'p', client_email: 'e', project_id: 'p1' })
  assert.deepEqual(Object.keys(s).sort(), ['client_email', 'configured', 'created_at', 'project_id', 'validated_at'])
  assert.deepEqual(publicSummary(null), { configured: false })
})

test('own-project jobs skip the shared monthly cap but keep point and concurrency caps', () => {
  const profile = { tier: 'member', member_until: new Date(Date.now() + 86400000).toISOString() }
  const usage = { unitsThisMonth: 1e9 }
  assert.equal(checkQuota({ profile, usage, estimate: 10 }).code, 'over_quota')
  assert.equal(checkQuota({ profile, usage, estimate: 10, ownProject: true }).ok, true)
  assert.equal(checkQuota({ profile, usage, estimate: 10, points: 1e9, ownProject: true }).code, 'too_many_points')
  assert.equal(checkQuota({ profile, usage, estimate: 10, running: 99, ownProject: true }).code, 'already_running')
})
