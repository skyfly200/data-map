// Is this request the platform's scheduler rather than a person?
//
// Netlify: a scheduled function cannot be invoked by URL in production, and
// the scheduler's request body is `{ "next_run": "<ISO time>" }`. The docs name
// no header, so the body is what identifies it; `x-netlify-event: schedule` is
// still accepted for older runtimes. The body check is skipped on Vercel, where
// the same route is a public URL and anyone could send that body.
//
// Vercel: a cron request carries `Authorization: Bearer <CRON_SECRET>` when the
// project has CRON_SECRET set. Without the secret a Vercel cron cannot be told
// apart from a visitor, so it is not treated as scheduled.

import { bearer } from './auth.mjs'

export async function isScheduledInvocation(request, env = process.env) {
  const secret = env.CRON_SECRET
  if (secret && bearer(request) === secret) return true
  if (env.VERCEL) return false
  if (request.headers.get('x-netlify-event') === 'schedule') return true
  if (request.method !== 'POST') return false
  try {
    const body = await request.clone().json()
    return typeof body?.next_run === 'string' && body.next_run.length > 0
  } catch {
    return false
  }
}
