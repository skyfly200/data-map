// Email notifications for jobs that finish while the member is away.
//
// A pipeline or model job outlives the request that started it, so a member
// almost always closes the tab and comes back later. This is the "come back"
// signal: when the worker settles a job it sends one email — succeeded or
// failed — to the address on the account, unless the member has opted out.
//
// Kept deliberately thin and dependency-free. Email goes out over Resend's HTTP
// API with `fetch`, so there is nothing to install and nothing to hold open.
// Every part is optional: with no RESEND_API_KEY (or no Supabase to look the
// address up) this is a silent no-op, exactly as the queue behaved before it
// existed. Sending must never be able to fail a job, so callers wrap it and it
// also guards itself — a bounced email is not a failed model.

import { adminClient } from './auth.mjs'

/** Configured only when we have somewhere to send from and a key to send with. */
export function notifyConfigured() {
  return Boolean(process.env.RESEND_API_KEY && fromAddress())
}

/** Web Push is configured when a VAPID key pair is present. */
export function pushConfigured() {
  return Boolean(process.env.VAPID_PUBLIC_KEY && process.env.VAPID_PRIVATE_KEY)
}

function fromAddress() {
  // A verified sender on the Resend account. No sensible default exists — an
  // unverified From is rejected — so absence just means "notifications off".
  return process.env.NOTIFY_FROM_EMAIL || process.env.RESEND_FROM_EMAIL || ''
}

/**
 * Where the member reads their jobs. Used to build the link in the email.
 * Falls back to the deployment's own URL (Netlify sets URL/DEPLOY_PRIME_URL),
 * then to a bare path if even that is missing.
 */
function jobsUrl(kind) {
  const base = (process.env.NOTIFY_APP_URL || process.env.URL
    || process.env.DEPLOY_PRIME_URL || '').replace(/\/+$/, '')
  const path = kind === 'model' ? '/modeling/maxent' : '/jobs'
  return base ? `${base}${path}` : path
}

/**
 * The address to notify, and whether they want it.
 *
 * The email lives in auth.users, reachable only with the service role, and the
 * opt-out lives in profiles. Both are read here so a caller need not. Returns
 * null when there is nobody to email, no address, or the member has said no —
 * every one of which means "send nothing", so the caller treats them alike.
 */
async function recipientFor(userId) {
  const client = adminClient()
  if (!client || !userId) return null

  const { data: profile } = await client
    .from('profiles')
    .select('notify_job_email, notify_job_push, display_name')
    .eq('user_id', userId).maybeSingle()

  // Each column defaults to true; an explicit false is the only opt-out. A
  // missing profile row (Supabase configured but the trigger has not run) is
  // treated as opted-in, matching the column defaults.
  const wantsEmail = !(profile && profile.notify_job_email === false)
  const wantsPush = !(profile && profile.notify_job_push === false)

  const { data, error } = await client.auth.admin.getUserById(userId)
  const email = (!error && data?.user?.email) || ''

  return { email, name: profile?.display_name || '', wantsEmail, wantsPush }
}

const escapeHtml = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]
))

/** Subject and body for a settled job, in both plain text and HTML. */
function compose(job, recipient) {
  const title = job.title || (job.kind === 'model' ? 'MaxEnt model' : 'Pipeline job')
  const link = jobsUrl(job.kind)
  const hello = recipient.name ? `Hi ${recipient.name},` : 'Hi,'

  if (job.status === 'succeeded') {
    const meta = job.result_meta || {}
    const detail = job.kind === 'model'
      ? (meta.presences ? `Fitted on ${meta.presences} presence points.` : '')
      : (meta.features ? `${Number(meta.features).toLocaleString()} observations enriched.` : '')
    return {
      subject: `✓ Your job finished: ${title}`,
      text: `${hello}\n\nYour job "${title}" has finished successfully.`
        + (detail ? `\n${detail}` : '')
        + `\n\nView it here: ${link}\n`,
      html: `<p>${escapeHtml(hello)}</p>`
        + `<p>Your job <strong>${escapeHtml(title)}</strong> has finished successfully.</p>`
        + (detail ? `<p>${escapeHtml(detail)}</p>` : '')
        + `<p><a href="${escapeHtml(link)}">View it here</a></p>`,
    }
  }

  // Failed (or any non-success settle we choose to report).
  const why = job.error ? String(job.error) : 'No further detail was recorded.'
  return {
    subject: `✕ Your job failed: ${title}`,
    text: `${hello}\n\nYour job "${title}" did not finish.\n\nReason: ${why}`
      + `\n\nYou can review it and try again here: ${link}\n`,
    html: `<p>${escapeHtml(hello)}</p>`
      + `<p>Your job <strong>${escapeHtml(title)}</strong> did not finish.</p>`
      + `<p><em>Reason:</em> ${escapeHtml(why)}</p>`
      + `<p><a href="${escapeHtml(link)}">Review it and try again</a></p>`,
  }
}

/** Send the email. Never throws — returns a small result object. */
async function sendEmail(job, recipient) {
  try {
    if (!notifyConfigured() || !recipient.wantsEmail || !recipient.email) {
      return { ok: false, skipped: true }
    }
    const { subject, text, html } = compose(job, recipient)
    const res = await fetch('https://api.resend.com/emails', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${process.env.RESEND_API_KEY}`,
      },
      body: JSON.stringify({ from: fromAddress(), to: recipient.email, subject, text, html }),
    })
    if (!res.ok) {
      const detail = await res.text().catch(() => '')
      return { ok: false, error: `Resend ${res.status}: ${detail.slice(0, 200)}` }
    }
    return { ok: true }
  } catch (err) {
    return { ok: false, error: String(err?.message || err) }
  }
}

/**
 * Send a Web Push notification to every browser this member has subscribed.
 *
 * `web-push` is imported lazily so the worker only pays for it when push is
 * actually configured, and a subscription the push service has forgotten (404
 * or 410 — the browser was uninstalled, permission revoked) is pruned so it is
 * not tried again.
 */
async function sendPush(job, recipient) {
  try {
    if (!pushConfigured() || !recipient.wantsPush) return { ok: false, skipped: true }
    const client = adminClient()
    if (!client) return { ok: false, skipped: true }

    const { data: subs } = await client
      .from('push_subscriptions')
      .select('id, endpoint, p256dh, auth')
      .eq('user_id', job.user_id)
    if (!subs?.length) return { ok: false, skipped: 'no-subscriptions' }

    const webpush = (await import('web-push')).default
    webpush.setVapidDetails(
      process.env.VAPID_SUBJECT || 'mailto:notifications@example.org',
      process.env.VAPID_PUBLIC_KEY,
      process.env.VAPID_PRIVATE_KEY,
    )

    const { subject, body } = pushPayload(job)
    const payload = JSON.stringify({ title: subject, body, url: jobsUrl(job.kind) })
    const dead = []
    let sent = 0

    await Promise.all(subs.map(async (s) => {
      const subscription = { endpoint: s.endpoint, keys: { p256dh: s.p256dh, auth: s.auth } }
      try {
        await webpush.sendNotification(subscription, payload)
        sent += 1
      } catch (err) {
        if (err?.statusCode === 404 || err?.statusCode === 410) dead.push(s.id)
      }
    }))

    if (dead.length) await client.from('push_subscriptions').delete().in('id', dead)
    return { ok: sent > 0, sent, pruned: dead.length }
  } catch (err) {
    return { ok: false, error: String(err?.message || err) }
  }
}

/** A short title + body for the push balloon. */
function pushPayload(job) {
  const title = job.title || (job.kind === 'model' ? 'MaxEnt model' : 'Pipeline job')
  return job.status === 'succeeded'
    ? { subject: '✓ Job finished', body: `"${title}" is ready to view.` }
    : { subject: '✕ Job failed', body: `"${title}" did not finish.` }
}

/**
 * Notify the owner that a job settled, by every channel they've kept on.
 * Resolves to a small result object rather than throwing: the worker calls this
 * after the job is already recorded, so a failure here must be swallowed.
 *
 * `job` needs at least { user_id, status, title, kind } and, for a nicer body,
 * { error, result_meta }.
 */
export async function notifyJobSettled(job) {
  try {
    if (!job || (job.status !== 'succeeded' && job.status !== 'failed')) {
      return { ok: false, skipped: 'status' }
    }
    // Nothing to do at all if neither channel is configured — saves the lookups.
    if (!notifyConfigured() && !pushConfigured()) return { ok: false, skipped: 'unconfigured' }

    const recipient = await recipientFor(job.user_id)
    if (!recipient) return { ok: false, skipped: 'no-recipient' }

    const [email, push] = await Promise.all([sendEmail(job, recipient), sendPush(job, recipient)])
    return { ok: email.ok || push.ok, email, push }
  } catch (err) {
    // Never let a notification take down the worker.
    return { ok: false, error: String(err?.message || err) }
  }
}
