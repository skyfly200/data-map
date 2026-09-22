// Notification preferences and the browser's push key.
//
//   GET   /.netlify/functions/job-notifications
//         → { ok, email, push, pushConfigured, emailConfigured, vapidPublicKey }
//   PATCH /.netlify/functions/job-notifications   body { email?, push? }
//         → updates the caller's opt-out flags
//
// Preferences live on the profiles row, which members may read but not update
// under RLS (only admins and the token hook write it), so the flip goes through
// the service role here. The VAPID public key is safe to hand the browser — it
// is the public half of the signing pair and is required to subscribe.

import { adminClient, requireUser } from '../lib/auth.mjs'
import { notifyConfigured, pushConfigured } from '../lib/notify.mjs'

const json = (body, status = 200) => new Response(JSON.stringify(body), {
  status, headers: { 'content-type': 'application/json' },
})

export default async function handler(request) {
  const auth = await requireUser(request)
  if (!auth.ok) return auth.response

  const client = adminClient()
  const base = {
    ok: true,
    emailConfigured: notifyConfigured(),
    pushConfigured: pushConfigured(),
    vapidPublicKey: process.env.VAPID_PUBLIC_KEY || '',
  }

  // Unconfigured Supabase (local dev): no profile to read, everything on.
  if (!client || !auth.user?.id) {
    return json({ ...base, email: true, push: true })
  }

  if (request.method === 'GET') {
    const { data } = await client.from('profiles')
      .select('notify_job_email, notify_job_push').eq('user_id', auth.user.id).maybeSingle()
    return json({
      ...base,
      email: data?.notify_job_email !== false,
      push: data?.notify_job_push !== false,
    })
  }

  if (request.method === 'PATCH' || request.method === 'POST') {
    let body
    try {
      body = await request.json()
    } catch {
      return json({ ok: false, error: 'Send the preferences as JSON.' }, 400)
    }
    const patch = {}
    if (typeof body.email === 'boolean') patch.notify_job_email = body.email
    if (typeof body.push === 'boolean') patch.notify_job_push = body.push
    if (!Object.keys(patch).length) {
      return json({ ok: false, error: 'Nothing to update.' }, 400)
    }
    const { error } = await client.from('profiles').update(patch).eq('user_id', auth.user.id)
    if (error) return json({ ok: false, error: error.message }, 500)

    const { data } = await client.from('profiles')
      .select('notify_job_email, notify_job_push').eq('user_id', auth.user.id).maybeSingle()
    return json({
      ...base,
      email: data?.notify_job_email !== false,
      push: data?.notify_job_push !== false,
    })
  }

  return json({ ok: false, error: 'Use GET or PATCH.' }, 405)
}
