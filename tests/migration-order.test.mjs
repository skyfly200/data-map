/* Ordering invariants in the SQL migrations.
 *
 * These exist because of a real failure. Migration 002 granted to
 * supabase_auth_admin before it defined is_member(). On a deployment without
 * that role — a plain Postgres, a self-hosted stack, an older project — the
 * grant failed, psql abandoned the rest of the script, and the tables existed
 * while the functions did not. The symptom appeared much later and somewhere
 * else entirely: migration 003 failing with "function public.is_member() does
 * not exist", which points at the wrong file.
 *
 * A migration cannot be unit tested without a database, but these two
 * properties are visible in the text, and both are what actually went wrong:
 *
 *   1. Nothing that depends on the deployment's roles may precede the
 *      definitions everything else needs.
 *   2. A statement naming a role that may not exist must be guarded, so a
 *      missing role costs that statement rather than the migration.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import { readFileSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'supabase_migrations')

/** Roles hosted Supabase provides and a plain Postgres does not. */
const HOSTED_ROLES = ['supabase_auth_admin', 'authenticated', 'anon', 'service_role']

const read = (file) => readFileSync(join(DIR, file), 'utf8')

/** The file with its comments blanked, so a role named in prose is not a hit.
 *  Line numbers are preserved, which is what makes a failure locatable. */
function withoutComments(sql) {
  return sql.split('\n').map((line) => (line.trim().startsWith('--') ? '' : line)).join('\n')
}

const migrations = readdirSync(DIR).filter((f) => f.endsWith('.sql')).sort()

test('there are migrations to check', () => {
  assert.ok(migrations.length >= 3, 'expected the migration set to still be here')
})

test('002 defines the tier helpers before it touches a hosted-only role', () => {
  const sql = withoutComments(read('002_membership_jobs_and_admin.sql'))

  for (const fn of ['current_tier', 'is_admin', 'is_member']) {
    const defined = sql.indexOf(`function public.${fn}()`)
    assert.notEqual(defined, -1, `002 should define ${fn}()`)

    for (const role of HOSTED_ROLES) {
      const named = sql.indexOf(role)
      if (named === -1) continue
      assert.ok(
        defined < named,
        `002 names ${role} at character ${named}, before defining ${fn}() at ${defined}. `
        + 'If that role is absent the statement fails and the function is never created, '
        + 'which surfaces from a later migration as a missing-function error.',
      )
    }
  }
})

test('no migration grants or revokes against a hosted-only role unguarded', () => {
  // A guarded statement lives inside a do-block and so is indented. One at
  // column zero is top-level, and a top-level failure ends the script.
  for (const file of migrations) {
    const lines = withoutComments(read(file)).split('\n')
    lines.forEach((line, i) => {
      if (!/^(grant|revoke)\s/i.test(line)) return
      for (const role of HOSTED_ROLES) {
        assert.ok(
          !new RegExp(`\\b${role}\\b`).test(line),
          `${file}:${i + 1} names ${role} in a top-level statement:\n    ${line.trim()}\n`
          + '  Wrap it in a do-block that checks pg_roles first, so a deployment '
          + 'without that role loses the statement and not the migration.',
        )
      }
    })
  }
})

test('003 checks for what it needs before it uses it', () => {
  const sql = withoutComments(read('003_custom_ee_layers.sql'))

  const guard = sql.indexOf('to_regprocedure')
  assert.notEqual(guard, -1,
    '003 should check the functions it depends on exist, so a half-applied 002 '
    + 'reports itself rather than raising a bare 42883 against this file.')

  for (const fn of ['is_member', 'is_admin']) {
    const used = sql.indexOf(`public.${fn}()`)
    if (used === -1) continue
    assert.ok(guard < used, `003 uses ${fn}() at ${used} before its guard at ${guard}`)
  }

  // The message has to name the migration to re-run; "missing function" alone
  // is what sent us to the wrong file in the first place.
  assert.match(sql, /Migration 002/, 'the guard should name 002 in its message')
})
