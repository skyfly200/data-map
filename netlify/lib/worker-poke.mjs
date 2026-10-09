/**
 * Start the worker now rather than at its next scheduled tick.
 *
 * Fire-and-forget: the request is started and then abandoned after a moment, so
 * the caller returns immediately while the worker keeps running server-side (an
 * aborted client fetch does not stop the invocation it already triggered). It
 * needs WORKER_POKE_SECRET set; without it the worker only takes the cron and an
 * admin, so this is a no-op and the job still starts on the next tick.
 */
export function pokeWorker(request) {
  const secret = process.env.WORKER_POKE_SECRET
  if (!secret) return
  try {
    const url = new URL('/.netlify/functions/ee-worker', request.url)
    const controller = new AbortController()
    // Long enough to hand the request off, short enough not to hold up the caller.
    const timer = setTimeout(() => controller.abort(), 500)
    fetch(url, { method: 'POST', headers: { 'x-worker-secret': secret }, signal: controller.signal })
      .catch(() => { /* the cron backstop covers a dropped poke */ })
      .finally(() => clearTimeout(timer))
  } catch { /* a poke that cannot even be built is not worth failing over */ }
}
