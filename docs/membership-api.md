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
| `tier` | default `member` | `member` or `admin`. `free` is refused — use revoke. |
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

An administrator is never demoted to member by a grant, so an admin renewing
their dues through the same PayPal button as everybody else keeps the admin
screen.

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
  "tier": "member", "active": true, "lapsed": false,
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
directly — PayPal sends its own payload shape. Put an automation between them
(Zapier, Make, n8n, or a small function) that:

1. Verifies the webhook signature before reading the body. Anyone can POST to a
   public URL, and this one mints memberships.
2. Filters to the subscription's **payment completed** event, for the
   membership plan only, and ignores every other event type.
3. Maps the payer's address to `email` and the payment's own transaction id to
   `ref`.
4. POSTs the JSON above with `months: 12`.
5. Treats `applied` and `pending` alike as success.

Send the transaction id as `ref` and retries stop being something you have to
think about.

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
