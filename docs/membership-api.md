# The membership API

For automations, not for people. The caller this exists for is the PayPal
webhook on the FRMS website: somebody pays, the automation calls this, and they
are a member.

    POST https://<your-site>/.netlify/functions/membership

## Authentication

A shared key, not a session — there is no member signed in to authorise as.
Send it as either header:

```
Authorization: Bearer <MEMBERSHIP_API_KEY>
X-API-Key: <MEMBERSHIP_API_KEY>
```

Set `MEMBERSHIP_API_KEY` in the Netlify environment. Generate one with
`openssl rand -base64 32`. **With no key set the endpoint refuses everything**
rather than falling open.

The key grants membership to any address, so treat it like the service role
key: Netlify environment variables only, never in the browser, never in a
client-side automation step.

## Grant or renew

```http
POST /.netlify/functions/membership
Authorization: Bearer <key>
Content-Type: application/json

{
  "action": "grant",
  "email": "member@example.org",
  "months": 12,
  "ref": "PAYPAL-TXN-8H2094",
  "source": "paypal",
  "name": "A Person"
}
```

| Field | | |
| --- | --- | --- |
| `email` | required | The payer's address. Matched case-insensitively. |
| `months` | default 12 | Months to add. 1–120. |
| `until` | | An explicit end date instead of a duration. Not with `months`. |
| `tier` | default `member` | `member`, `perpetual` or `admin`. `free` is refused — use revoke. |
| `ref` | strongly advised | The processor's transaction id. See idempotency. |
| `source` | default `api` | Where it came from, for the record. |
| `name` | | Fills the display name if they have not set one. |
| `note` | | Free text, kept with the grant. |

### Two successful outcomes

```json
{ "ok": true, "status": "applied", "email": "…", "tier": "member",
  "member_until": "2027-06-15T00:00:00.000Z" }
```

```json
{ "ok": true, "status": "pending", "email": "…",
  "note": "No account uses that address yet. The membership will be applied when they sign up." }
```

**`pending` is not a failure.** Paying before making an account is at least as
common as the other way round. The grant is recorded and applied automatically
by a database trigger the moment that address signs up — nobody has to notice.
An automation should treat both as success.

### Idempotency

Payment processors retry until they get a 2xx, and a delivery that timed out
*after* succeeding looks exactly like one that failed. Always send `ref` — the
processor's own transaction id. A second call with a `ref` already seen returns
the first outcome with `"duplicate": true` and changes nothing.

Without a `ref`, a retried webhook grants a second term. The uniqueness is
enforced by a database constraint, not by an application check, so two
deliveries racing each other cannot both get through.

### Renewing early does not cost time

A term extends from whichever is later: today, or the expiry already held. A
member renewing with two months left gets those two months plus the new term.
A lapsed member starts from today rather than from the date they lapsed.

### Tiers that do not expire

Two tiers ignore `member_until` entirely:

| | |
| --- | --- |
| `perpetual` | A member whose standing does not run out — honorary and life members, founders, anyone FRMS does not want to invoice. Same powers as `member`. |
| `admin` | Also exempt, and for a structural reason: the admin screen is the only place a membership date can be corrected, so an admin demoted by their own dues date would lock FRMS out of fixing it. |

A grant never converts either one into a membership that expires. Somebody with
a perpetual account who pays anyway — out of habit, or because nobody told the
website — keeps the perpetual account, and an admin renewing through the same
PayPal button as everybody else keeps the admin screen. The date is still
recorded in both cases; it just stops deciding anything.

A `lookup` on such an account reports `"expires": false`, `"lapsed": false` and
`"days_left": 0`. **An automation chasing renewals should filter on `expires`,
not on `days_left`** — a perpetual member reads as zero days left because there
is no countdown, not because they are about to run out.

## Revoke

```json
{ "action": "revoke", "email": "member@example.org" }
```

Sets the tier to `free` and clears the expiry. For an address with no account,
any pending grants are dropped. Refuses to revoke an administrator — change
that in the admin screen, deliberately.

## Look up

```json
{ "action": "lookup", "email": "member@example.org" }
```

or `GET /.netlify/functions/membership?email=member@example.org`

```json
{
  "ok": true, "email": "…", "has_account": true, "known": true,
  "tier": "member", "active": true, "lapsed": false, "expires": true,
  "member_until": "2027-06-15T00:00:00.000Z", "days_left": 365,
  "pending_grants": []
}
```

`tier` is the **effective** tier: a member whose expiry has passed reads as
`free` here, the same as the app treats them. `stored_tier` carries what the
row literally says if you need the difference.

## Status codes

| | |
| --- | --- |
| 200 | Done, or already done. Check `status`. |
| 400 | The request was wrong, and `error` says how. Do not retry unchanged. |
| 401 | The key is missing or wrong. |
| 409 | Refused on purpose — revoking an administrator. |
| 503 | Not configured: no API key, or no Supabase. Retry later. |
| 500 | Something else. Safe to retry; `ref` makes that harmless. |

## Wiring it to PayPal

FRMS memberships renew annually, so the payment side is a PayPal
**subscription** rather than a one-off order. The webhook does not call this
directly — PayPal sends its own payload shape. Something has to sit between
them: on the FRMS site that is the serverless function already handling the
webhook and sending the welcome email. It should:

1. Verify the webhook signature before reading the body. Anyone can POST to a
   public URL, and this one mints memberships.
2. Filter to the subscription's **payment completed** event, for the membership
   plan only, and ignore every other event type.
3. Map the payer's address to `email` and the payment's own transaction id to
   `ref`.
4. POST the JSON above with `months: 12`.
5. Treat `applied` and `pending` alike as success.

Send the transaction id as `ref` and retries stop being something you have to
think about.

### Grant first, then send the email

Order matters, and it is the one thing in this integration that is easy to get
backwards.

The grant is idempotent — the same `ref` twice changes nothing. The welcome
email is not; there is no way to un-send one. So the handler should do the
un-repeatable thing last:

```
verify signature → grant membership → if that failed retryably, return 500
                 → send welcome email → return 200
```

Returning 500 has PayPal redeliver the whole webhook. With the grant first, a
redelivery that happens because the grant failed has not yet sent an email, so
nobody gets two. A redelivery that happens because the *email* failed re-runs
the grant, which recognises the `ref` and does nothing. Either way the member
ends up with one membership and one email.

Put the email first and a transient failure anywhere after it emails them
again on every retry.

### A function to call it

Drop this beside the existing handler. It never throws: a membership fault
should not take down the welcome email, and an exception inside a webhook
handler usually becomes a 502 that tells PayPal nothing useful.

```js
const NEXSTRATA_URL = process.env.NEXSTRATA_URL
const MEMBERSHIP_API_KEY = process.env.MEMBERSHIP_API_KEY

export async function grantMembership({ email, ref, name = null, months = 12 }) {
  if (!NEXSTRATA_URL || !MEMBERSHIP_API_KEY) {
    // Misconfiguration, not a transient fault. Retrying will not fix it, and a
    // webhook retried forever is worse than one that reports the problem.
    return { ok: false, retryable: false, error: 'NEXSTRATA_URL or MEMBERSHIP_API_KEY is unset' }
  }

  let res
  let data = {}
  try {
    res = await fetch(`${NEXSTRATA_URL}/.netlify/functions/membership`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${MEMBERSHIP_API_KEY}`,
      },
      body: JSON.stringify({ action: 'grant', email, months, ref, source: 'paypal', name }),
      signal: AbortSignal.timeout(10_000),
    })
    data = await res.json().catch(() => ({}))
  } catch (err) {
    // Network fault or timeout. The grant may or may not have landed, which is
    // exactly the case `ref` exists to make harmless.
    return { ok: false, retryable: true, error: String(err?.message || err) }
  }

  if (res.ok && data.ok) {
    return {
      ok: true,
      status: data.status,          // 'applied' or 'pending' — both are success
      duplicate: !!data.duplicate,
      memberUntil: data.member_until,
      retryable: false,
    }
  }

  // 400, 401 and 409 will fail identically forever: a malformed address, a
  // wrong key, a deliberate refusal. Anything else is worth another delivery.
  return {
    ok: false,
    retryable: ![400, 401, 409].includes(res.status),
    error: data.error || `membership api returned ${res.status}`,
  }
}
```

Plain `fetch` and `AbortSignal.timeout`, so it runs unchanged on Netlify
Functions, Vercel, Cloudflare Workers and Node 18+ Lambda.

In the handler:

```js
const membership = await grantMembership({
  email: payerEmail,
  ref: transactionId,
  name: payerName,
})

if (!membership.ok) {
  console.error('membership grant failed', membership.error)
  // Let PayPal redeliver. No email has gone out yet, so the retry is clean.
  if (membership.retryable) return { statusCode: 500, body: 'retry' }
  // Not retryable: log loudly and carry on, or this address never gets its
  // welcome email either.
}

await sendWelcomeEmail({
  email: payerEmail,
  name: payerName,
  // 'pending' means they have not made a Nexstrata account yet, which changes
  // what the email should ask them to do.
  needsAccount: membership.status === 'pending',
})

return { statusCode: 200, body: 'ok' }
```

### The email can say different things

`applied` and `pending` are both success, but they describe people in different
situations, and the welcome email is the one chance to tell them what to do:

| | |
| --- | --- |
| `applied` | They already have a Nexstrata account and it is now a membership. Point them at signing in. Their tier arrives at the next token refresh, within about an hour, or immediately if they sign out and back in. |
| `pending` | They have paid but have no account. Tell them to sign up **with this same address** and their membership applies itself the moment they do. |

Getting this wrong is not fatal — a `pending` member who is told to sign in
will work it out — but it is the difference between a welcome email that reads
as written for them and one that does not.

### Environment, on the FRMS site

| | |
| --- | --- |
| `NEXSTRATA_URL` | `https://<the Nexstrata site>`, no trailing slash. |
| `MEMBERSHIP_API_KEY` | The same value set on Nexstrata. Server-side environment only. |

The key grants membership to any address, so it goes in the function's
environment configuration and nowhere else — never in browser code, never in a
client-side step, never committed.

### Act on one event per year, not two

A subscription emits an activation event *and* a payment event when it starts,
and each annual renewal emits another payment event. Those carry genuinely
different ids, so `ref` cannot tell that activation and the first payment are
the same purchase — two grants, both idempotent, two years sold for one year
paid.

Pick the event that fires once per payment, on the first year and on every
renewal alike, and act on that one alone. Payment completed satisfies this;
activation does not, since it never fires again.

Cancellation is a separate decision. A cancelled subscription has usually paid
through the end of its term, so `revoke` on cancellation cuts short time the
member already bought. Let it lapse on its own instead — the expiry is already
set — unless the cancellation is a refund.

## What it cannot do

It cannot create accounts. Membership attaches to an address; the person still
signs up themselves, at which point a pending grant applies itself. That is
deliberate — creating accounts from a payment webhook means an automation can
mint logins.
