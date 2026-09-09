// Turning a failed function call into something worth showing someone.
//
// The functions answer a refusal with `{ ok: false, error: 'a sentence' }`, and
// those sentences are written for the member — a quota that ran out, an area
// with nothing in it — so they are shown as they arrive.
//
// What this exists for is everything that is NOT one of those. When a function
// is missing (running the built app locally, where Netlify functions are not
// served), Nitro answers with its own JSON error payload, and that payload has
// an `error` field of its own holding `true`. Reading it as a message put the
// word "true" on screen where an explanation belonged.

/** The message in an error payload, or a plain description of the status. */
export function messageFrom(data, status) {
  const said = data?.error
  // Only a non-empty string is a message. A boolean is Nitro's own flag, and an
  // object is a shape we did not send.
  if (typeof said === 'string' && said.trim()) return said.trim()

  if (status === 404) {
    return 'That server function is not available here. It only runs on the deployed site.'
  }
  if (status === 401) return 'Sign in again, then retry.'
  if (status === 403) return 'Your account is not allowed to do that.'
  if (status === 503) return 'That service is not configured on this deployment.'
  return `The server returned ${status}.`
}
