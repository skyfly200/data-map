// Netlify background function: runs one queued job with a 15-minute ceiling.
// The `-background` suffix makes Netlify answer the caller 202 at once and keep
// running. Reached by ee-worker's scheduled hand-off and the submit-time poke
// (both carry WORKER_POKE_SECRET), or by an admin.

import { requireAdmin } from '../lib/auth.mjs'
import { runOneJob } from '../lib/worker-run.mjs'

export default async function handler(request) {
  const pokeSecret = process.env.WORKER_POKE_SECRET
  const poked = Boolean(pokeSecret) && request.headers.get('x-worker-secret') === pokeSecret
  if (!poked) {
    const auth = await requireAdmin(request)
    if (!auth.ok) return auth.response
  }
  return runOneJob()
}
