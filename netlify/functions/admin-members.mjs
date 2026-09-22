// The admin surface: members, their quotas, and the saved datasets.
//
//   GET  ?what=members    every profile, with this month's usage alongside
//   GET  ?what=datasets   every saved dataset
//   POST { action: 'set-member', user_id, tier?, member_until?, limits… }
//   POST { action: 'set-dataset', id, visibility?, title?, description? }
//   POST { action: 'delete-dataset', id }
//
// Guarded by the admin tier read from the token. Unlike the job path this does
// not need a fresh read: the worst a stale claim allows is an ex-admin editing
// quotas for up to an hour, which is a governance problem rather than a way to
// spend the FRMS Earth Engine budget, and it is bounded by the same
// refresh either way.

import { adminClient, requireAdmin } from '../lib/auth.mjs'
import { DEFAULT_LIMITS, summariseUsage } from '../lib/quotas.mjs'
import { TIERS } from '../lib/tiers.mjs'
import { CustomLayerError, normaliseCustomLayer } from '../lib/ee-custom-layers.mjs'
import { CRON_JOBS } from '../lib/cron-jobs.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json' },
})

/** The fields an admin may change, and what counts as a valid value for each. */
const LIMIT_FIELDS = Object.keys(DEFAULT_LIMITS)

function cleanLimits(input) {
  const out = {}
  for (const field of LIMIT_FIELDS) {
    if (input[field] === undefined) continue
    const value = Math.floor(Number(input[field]))
    if (!Number.isFinite(value) || value < 0) throw new Error(`${field} must be a whole number, zero or more.`)
    // An upper bound so a typo cannot hand out an unmetered account: five
    // zeroes too many is easier to type than to notice.
    out[field] = Math.min(value, 1_000_000)
  }
  return out
}

/**
 * Every account's email and metadata name, keyed by user id.
 *
 * Emails live in Supabase auth, not in the profiles table, so an admin looking
 * at "3dbe7a05" cannot tell who it is. Read through the auth admin API, which
 * the service-role client is allowed and paged through because it returns a
 * bounded page at a time. Best-effort: if it is unavailable the member list
 * still renders, just without the emails.
 */
async function authIdentities(client) {
  const out = new Map()
  if (!client.auth?.admin?.listUsers) return out
  const perPage = 200
  for (let page = 1; page <= 50; page += 1) {
    const { data, error } = await client.auth.admin.listUsers({ page, perPage })
    const users = data?.users || []
    if (error || !users.length) break
    for (const u of users) {
      // A name the member set at sign-up, as a fallback below display_name.
      const metaName = u.user_metadata?.full_name || u.user_metadata?.name || ''
      out.set(u.id, { email: u.email || '', metaName })
    }
    if (users.length < perPage) break
  }
  return out
}

async function listMembers(client) {
  const { data: profiles, error } = await client
    .from('profiles')
    .select('user_id, tier, display_name, member_until, notes, created_at, '
      + LIMIT_FIELDS.join(', '))
    .order('tier', { ascending: false })
  if (error) throw new Error(error.message)

  // Emails and sign-up names, best-effort, so a nameless profile still shows who
  // it is rather than eight characters of its UUID.
  let identities = new Map()
  try {
    identities = await authIdentities(client)
  } catch { /* the list still renders without emails */ }

  // Usage alongside each member: a quota with no reading against it is not
  // something an admin can act on.
  const { data: jobs } = await client
    .from('ee_jobs')
    .select('user_id, status, created_at, cost_units, estimated_units')
    .gte('created_at', new Date(Date.now() - 32 * 86400000).toISOString())

  const byUser = new Map()
  for (const job of jobs || []) {
    if (!byUser.has(job.user_id)) byUser.set(job.user_id, [])
    byUser.get(job.user_id).push(job)
  }

  return (profiles || []).map((p) => {
    const who = identities.get(p.user_id) || {}
    return {
      ...p,
      email: who.email || '',
      // The name to show: what the member set on their profile, then their
      // sign-up name, and only then the UUID stub the page fell back to before.
      display_name: p.display_name || who.metaName || '',
      usage: summariseUsage(byUser.get(p.user_id) || []),
    }
  })
}

export default async function handler(request) {
  const auth = await requireAdmin(request)
  if (!auth.ok) return auth.response

  const client = adminClient()
  if (!client) return json({ ok: false, error: 'Supabase is not configured.' }, 503)

  if (request.method === 'GET') {
    const what = new URL(request.url).searchParams.get('what') || 'members'
    try {
      if (what === 'datasets') {
        const { data, error } = await client.from('saved_datasets')
          .select('*').order('created_at', { ascending: false })
        if (error) throw new Error(error.message)
        return json({ ok: true, datasets: data || [] })
      }
      if (what === 'layers') {
        const { data, error } = await client.from('ee_custom_layers')
          .select('*').order('name')
        if (error) throw new Error(error.message)
        return json({ ok: true, layers: data || [] })
      }
      if (what === 'crons') {
        // Last 50 log rows per job, newest first. Rows older than 90 days are
        // dropped here so the table stays bounded without a scheduled prune job.
        const cutoff = new Date(Date.now() - 90 * 86400000).toISOString()
        const { data: logs, error: logErr } = await client
          .from('cron_logs')
          .select('*')
          .gte('fired_at', cutoff)
          .order('fired_at', { ascending: false })
          .limit(200)
        if (logErr) throw new Error(logErr.message)

        // Group logs by job_id, capped at 50 per job.
        const byJob = {}
        for (const row of logs || []) {
          if (!byJob[row.job_id]) byJob[row.job_id] = []
          if (byJob[row.job_id].length < 50) byJob[row.job_id].push(row)
        }
        return json({ ok: true, jobs: CRON_JOBS, logs: byJob })
      }
      return json({ ok: true, members: await listMembers(client), limitFields: LIMIT_FIELDS })
    } catch (err) {
      return json({ ok: false, error: err.message }, 500)
    }
  }

  if (request.method !== 'POST') return json({ ok: false, error: 'Use GET or POST.' }, 405)

  let body
  try {
    body = await request.json()
  } catch {
    return json({ ok: false, error: 'Send JSON.' }, 400)
  }

  try {
    if (body.action === 'set-member') {
      if (!body.user_id) throw new Error('Which member?')
      const patch = cleanLimits(body)

      if (body.tier !== undefined) {
        if (!TIERS.includes(body.tier)) throw new Error(`Unknown tier “${body.tier}”.`)
        // An admin removing their own admin tier locks FRMS out of this
        // screen entirely, and the only way back is the SQL editor.
        if (body.user_id === auth.user?.id && body.tier !== 'admin') {
          throw new Error('You cannot remove your own administrator tier from here.')
        }
        patch.tier = body.tier
      }
      if (body.member_until !== undefined) {
        if (body.member_until === null || body.member_until === '') {
          patch.member_until = null
        } else {
          const when = new Date(body.member_until)
          if (!Number.isFinite(when.getTime())) throw new Error('That is not a date.')
          patch.member_until = when.toISOString()
        }
      }
      if (body.notes !== undefined) patch.notes = String(body.notes).slice(0, 2000)

      if (!Object.keys(patch).length) throw new Error('Nothing to change.')

      const { data, error } = await client.from('profiles')
        .update(patch).eq('user_id', body.user_id).select().single()
      if (error) throw new Error(error.message)
      return json({
        ok: true,
        member: data,
        // Said plainly because it looks like a bug otherwise: the member keeps
        // the old tier until their token refreshes.
        note: patch.tier ? 'The new tier reaches that member when their session next refreshes, within about an hour.' : '',
      })
    }

    if (body.action === 'set-dataset') {
      if (!body.id) throw new Error('Which dataset?')
      const patch = {}
      if (body.visibility !== undefined) {
        if (!['private', 'members', 'public'].includes(body.visibility)) {
          throw new Error('Visibility must be private, members or public.')
        }
        patch.visibility = body.visibility
      }
      if (body.title !== undefined) patch.title = String(body.title).slice(0, 200)
      if (body.description !== undefined) patch.description = String(body.description).slice(0, 2000)
      if (!Object.keys(patch).length) throw new Error('Nothing to change.')

      const { data, error } = await client.from('saved_datasets')
        .update(patch).eq('id', body.id).select().single()
      if (error) throw new Error(error.message)
      return json({ ok: true, dataset: data })
    }

    if (body.action === 'delete-dataset') {
      if (!body.id) throw new Error('Which dataset?')
      // The row goes; the stored file is left. Deleting both from one click is
      // how a shared dataset gets lost by accident, and storage is cheap.
      const { error } = await client.from('saved_datasets').delete().eq('id', body.id)
      if (error) throw new Error(error.message)
      return json({ ok: true, deleted: body.id })
    }

    // ── Custom Earth Engine layers ──────────────────────────────────────────
    if (body.action === 'save-layer') {
      // Validated before it is stored, not before it is used: a bad asset ID
      // saved now is a broken layer found later, by somebody else.
      const layer = normaliseCustomLayer(body)
      const row = { ...layer, created_by: auth.user?.id || null }

      const { data, error } = body.id
        ? await client.from('ee_custom_layers').update(layer).eq('id', body.id).select().single()
        : await client.from('ee_custom_layers').insert(row).select().single()
      if (error) {
        // A duplicate slug is the common mistake and its raw message is
        // Postgres talking about a unique constraint.
        throw new Error(/duplicate key|unique/i.test(error.message)
          ? `A layer with the short name “${layer.slug}” already exists.`
          : error.message)
      }
      return json({ ok: true, layer: data })
    }

    if (body.action === 'delete-layer') {
      if (!body.id) throw new Error('Which layer?')
      const { error } = await client.from('ee_custom_layers').delete().eq('id', body.id)
      if (error) throw new Error(error.message)
      return json({ ok: true, deleted: body.id })
    }

    throw new Error(`Unknown action “${body.action}”.`)
  } catch (err) {
    // A CustomLayerError is the administrator's form being wrong and its
    // message is written for them.
    if (err instanceof CustomLayerError) return json({ ok: false, error: err.message }, 400)
    return json({ ok: false, error: err.message }, 400)
  }
}
