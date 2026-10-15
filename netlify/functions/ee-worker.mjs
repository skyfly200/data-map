// The worker that runs queued Earth Engine jobs: the scheduled entry point.
//
// On Netlify a scheduled function is cut off at 30 seconds and cannot be
// called by URL, so here it only hands off to ee-worker-background, which runs
// the job with a 15-minute ceiling. That hand-off needs WORKER_POKE_SECRET;
// without it the job runs inline and anything over 30 seconds is cut off.
// On Vercel there is no background variant and this runs the job itself.
//
// A manual call has to be an admin, or the submit endpoint's poke carrying
// WORKER_POKE_SECRET, so the queue cannot be driven by anyone who finds the URL.

import { requireAdmin } from '../lib/auth.mjs'
import { isScheduledInvocation } from '../lib/cron-auth.mjs'
import { runOneJob } from '../lib/worker-run.mjs'

// Every minute, so a job that misses the submit-time poke still starts within a
// minute. Netlify ignores the timeout for a scheduled function (30 s cap); it
// still applies when this runs on Vercel.
export const config = { timeout: 300, schedule: '* * * * *' }

/** Hand the run to the background function and wait only for its 202. */
async function handOff(request, secret) {
  const url = new URL('/.netlify/functions/ee-worker-background', request.url)
  const res = await fetch(url, { method: 'POST', headers: { 'x-worker-secret': secret } })
  return res.status
}

export default async function handler(request) {
  const pokeSecret = process.env.WORKER_POKE_SECRET
  const poked = Boolean(pokeSecret) && request.headers.get('x-worker-secret') === pokeSecret
  // The scheduler is recognised by what the platform sends (see cron-auth), not
  // by anything a visitor can put in the URL.
  const scheduled = await isScheduledInvocation(request)
  if (!scheduled && !poked) {
    const auth = await requireAdmin(request)
    if (!auth.ok) return auth.response
  }

  if (scheduled && pokeSecret && !process.env.VERCEL) {
    try {
      const status = await handOff(request, pokeSecret)
      if (status === 202) {
        return new Response(JSON.stringify({ ok: true, handedOff: true }),
          { headers: { 'content-type': 'application/json' } })
      }
      console.warn(`[ee-worker] background hand-off returned ${status}; running inline`)
    } catch (e) {
      console.warn('[ee-worker] background hand-off failed; running inline:', String(e))
    }
  }

  // Under the platform's limit with room for the request in flight: Vercel's
  // 300 s, or Netlify's 30 s when there is no background function to hand to.
  return runOneJob({ budgetMs: process.env.VERCEL ? 200_000 : 18_000 })
}
