import test from 'node:test'
import assert from 'node:assert/strict'

import { explainSyncError } from '../composables/useCloudSync.js'

// "Sync failed" on its own is useless: the usual cause is a setup step that has
// not been run, and PostgREST reports that with a schema-cache message that
// means nothing unless you already know what it implies.

test('a missing table is named as a setup step, with the file to run', () => {
  // What Supabase actually returns when the migration has not been applied.
  const err = {
    code: 'PGRST205',
    message: "Could not find the table 'public.user_settings' in the schema cache",
  }
  const info = explainSyncError(err)
  assert.match(info.message, /not set up for sync/i)
  assert.match(info.hint, /001_user_settings_and_charts\.sql/)
  assert.equal(info.raw, err.message)
})

test('the Postgres undefined-table code is recognised too', () => {
  const info = explainSyncError({ code: '42P01', message: 'relation "public.saved_charts" does not exist' })
  assert.match(info.message, /not set up for sync/i)
  assert.match(info.hint, /migration/i)
})

test('a missing table is recognised from the message alone', () => {
  // Some clients surface no code at all.
  const info = explainSyncError(new Error('relation "public.user_settings" does not exist'))
  assert.match(info.message, /not set up for sync/i)
})

test('a row-level security refusal is reported as a policy problem', () => {
  const info = explainSyncError({ code: '42501', message: 'new row violates row-level security policy for table "user_settings"' })
  assert.match(info.message, /not allowed/i)
  assert.match(info.hint, /row-level security/i)
  // Crucially NOT reported as a missing table, which would send someone to the
  // wrong fix.
  assert.doesNotMatch(info.message, /missing/i)
})

test('a network failure is not blamed on the schema', () => {
  const info = explainSyncError(new TypeError('Failed to fetch'))
  assert.match(info.message, /reach your account/i)
  assert.doesNotMatch(info.hint, /migration/i)
})

test('an unrecognised error still yields something printable', () => {
  const info = explainSyncError(new Error('something odd'))
  assert.ok(info.message.length > 0)
  assert.equal(info.raw, 'something odd')
})

test('a thrown non-Error does not produce "undefined"', () => {
  for (const bad of [null, undefined, '', 0]) {
    const info = explainSyncError(bad)
    assert.ok(info.message.length > 0)
    assert.ok(!/undefined/.test(info.raw), `raw was "${info.raw}"`)
  }
})
