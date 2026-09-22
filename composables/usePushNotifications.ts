// Job notification preferences: email opt-out, and Web Push subscribe/unsubscribe.
//
// A job runs on the server after the tab is closed (see useEeJobs), so the way
// a member hears it finished is a channel that reaches them when they are not
// looking: an email, or a push notification. This composable owns the browser
// side of both — reading and flipping the preferences, and the subscribe dance
// for push (permission → PushManager → a row the worker can send to).
//
// Everything degrades quietly: no VAPID key from the server, or a browser
// without the Push API, just leaves push unavailable and the email toggle alone.

const PREFS_URL = '/.netlify/functions/job-notifications'

/** Base64url → Uint8Array, the form PushManager wants the VAPID key in. */
function urlBase64ToUint8Array(base64: string): Uint8Array {
  const padding = '='.repeat((4 - (base64.length % 4)) % 4)
  const b64 = (base64 + padding).replace(/-/g, '+').replace(/_/g, '/')
  const raw = atob(b64)
  const out = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i += 1) out[i] = raw.charCodeAt(i)
  return out
}

export function usePushNotifications() {
  const { $supabase } = useNuxtApp()
  const { accessToken } = useAuth()

  const emailOn = useState('notify-email', () => true)
  const pushOn = useState('notify-push', () => false)
  const emailAvailable = useState('notify-email-available', () => false)
  const pushAvailable = useState('notify-push-available', () => false)
  const vapidKey = useState('notify-vapid', () => '')
  const loaded = useState('notify-loaded', () => false)
  const busy = useState('notify-busy', () => false)
  const error = useState('notify-error', () => '')

  const supported = computed(() => import.meta.client
    && 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window)

  async function authHeaders() {
    const token = await accessToken()
    return {
      'content-type': 'application/json',
      ...(token ? { authorization: `Bearer ${token}` } : {}),
    }
  }

  /** Load configured-ness, the VAPID key, and the current preferences. */
  async function load() {
    if (!import.meta.client) return
    error.value = ''
    try {
      const res = await fetch(PREFS_URL, { headers: await authHeaders() })
      const data = await res.json().catch(() => ({}))
      if (!res.ok || !data.ok) return
      emailAvailable.value = !!data.emailConfigured
      pushAvailable.value = !!data.pushConfigured && !!data.vapidPublicKey && supported.value
      vapidKey.value = data.vapidPublicKey || ''
      emailOn.value = data.email !== false
      // Push is only "on" if they've opted in AND this browser holds a live
      // subscription — a preference alone does not deliver anything here.
      pushOn.value = (data.push !== false) && await hasSubscription()
    } finally {
      loaded.value = true
    }
  }

  async function hasSubscription(): Promise<boolean> {
    if (!supported.value) return false
    try {
      const reg = await navigator.serviceWorker.ready
      return !!(await reg.pushManager.getSubscription())
    } catch {
      return false
    }
  }

  async function savePref(patch: { email?: boolean; push?: boolean }) {
    const res = await fetch(PREFS_URL, {
      method: 'PATCH', headers: await authHeaders(), body: JSON.stringify(patch),
    })
    const data = await res.json().catch(() => ({}))
    if (!res.ok || !data.ok) throw new Error(data.error || 'Could not save that preference.')
    return data
  }

  async function setEmail(on: boolean) {
    busy.value = true
    error.value = ''
    try {
      await savePref({ email: on })
      emailOn.value = on
    } catch (e: any) {
      error.value = e.message
    } finally {
      busy.value = false
    }
  }

  /** Turn push on: ask permission, subscribe, store the subscription, flip the flag. */
  async function enablePush() {
    if (!supported.value || !vapidKey.value) return
    busy.value = true
    error.value = ''
    try {
      const permission = await Notification.requestPermission()
      if (permission !== 'granted') {
        error.value = 'Notifications are blocked for this site in your browser settings.'
        return
      }
      // The offline composable registers /sw.js; ensure it exists here too.
      await navigator.serviceWorker.register('/sw.js', { scope: '/' })
      const reg = await navigator.serviceWorker.ready
      const sub = await reg.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(vapidKey.value),
      })
      await storeSubscription(sub)
      await savePref({ push: true })
      pushOn.value = true
    } catch (e: any) {
      error.value = e?.message || 'Could not enable push notifications.'
    } finally {
      busy.value = false
    }
  }

  /** Turn push off: unsubscribe this browser, drop its row, flip the flag. */
  async function disablePush() {
    busy.value = true
    error.value = ''
    try {
      if (supported.value) {
        const reg = await navigator.serviceWorker.ready
        const sub = await reg.pushManager.getSubscription()
        if (sub) {
          await removeSubscription(sub.endpoint)
          await sub.unsubscribe()
        }
      }
      await savePref({ push: false })
      pushOn.value = false
    } catch (e: any) {
      error.value = e?.message || 'Could not disable push notifications.'
    } finally {
      busy.value = false
    }
  }

  async function storeSubscription(sub: PushSubscription) {
    if (!$supabase) return
    const { data: session } = await $supabase.auth.getSession()
    const uid = session.session?.user?.id
    if (!uid) return
    const json: any = sub.toJSON()
    // Upsert on endpoint so re-subscribing the same browser refreshes rather
    // than duplicates. RLS scopes this to the caller's own rows.
    const { error: err } = await $supabase.from('push_subscriptions').upsert({
      user_id: uid,
      endpoint: sub.endpoint,
      p256dh: json.keys?.p256dh,
      auth: json.keys?.auth,
      user_agent: navigator.userAgent.slice(0, 300),
    }, { onConflict: 'endpoint' })
    if (err) throw new Error(err.message)
  }

  async function removeSubscription(endpoint: string) {
    if (!$supabase) return
    await $supabase.from('push_subscriptions').delete().eq('endpoint', endpoint)
  }

  return {
    emailOn, pushOn, emailAvailable, pushAvailable, supported, loaded, busy, error,
    load, setEmail, enablePush, disablePush,
  }
}
