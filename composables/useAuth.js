// Reactive auth state on top of the browser Supabase client ($supabase).
// Shared across every view via useState, so the header and the Data tab see
// the same session. All sign-in methods funnel through here.
//
// `configured` is false when Supabase public env isn't set — callers should
// then treat the app as open (the functions also run unauthenticated only when
// the server is likewise unconfigured).

export function useAuth() {
  const user = useState('auth-user', () => null)
  const ready = useState('auth-ready', () => false)
  const { $supabase } = useNuxtApp()
  const configured = Boolean($supabase)

  // Wire up the session listener once on the client.
  const initialized = useState('auth-initialized', () => false)
  if (import.meta.client && configured && !initialized.value) {
    initialized.value = true
    $supabase.auth.getSession().then(({ data }) => {
      user.value = data.session?.user ?? null
      ready.value = true
    })
    $supabase.auth.onAuthStateChange((_event, session) => {
      user.value = session?.user ?? null
      ready.value = true
    })
  } else if (import.meta.client && !configured) {
    ready.value = true
  }

  // Where magic-link / OAuth flows return to.
  const redirectTo = () => (import.meta.client ? `${window.location.origin}/login` : undefined)

  async function signInWithOtp(email) {
    if (!configured) throw new Error('Auth is not configured.')
    const { error } = await $supabase.auth.signInWithOtp({ email, options: { emailRedirectTo: redirectTo() } })
    if (error) throw error
  }
  async function signInWithPassword(email, password) {
    if (!configured) throw new Error('Auth is not configured.')
    const { error } = await $supabase.auth.signInWithPassword({ email, password })
    if (error) throw error
  }
  async function signUp(email, password) {
    if (!configured) throw new Error('Auth is not configured.')
    const { error } = await $supabase.auth.signUp({ email, password, options: { emailRedirectTo: redirectTo() } })
    if (error) throw error
  }
  async function signInWithOAuth(provider) {
    if (!configured) throw new Error('Auth is not configured.')
    const { error } = await $supabase.auth.signInWithOAuth({ provider, options: { redirectTo: redirectTo() } })
    if (error) throw error
  }
  async function signOut() {
    if (!configured) return
    await $supabase.auth.signOut()
    user.value = null
  }

  // The current access token (JWT) to send as a Bearer to protected functions.
  async function accessToken() {
    if (!configured) return null
    const { data } = await $supabase.auth.getSession()
    return data.session?.access_token ?? null
  }

  const isAuthed = computed(() => Boolean(user.value))

  return {
    user, ready, configured, isAuthed,
    signInWithOtp, signInWithPassword, signUp, signInWithOAuth, signOut, accessToken,
  }
}
