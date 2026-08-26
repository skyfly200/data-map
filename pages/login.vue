<template>
  <div class="login">
    <div class="card">
      <h2>Sign in</h2>
      <p class="sub">A quick sign-in keeps the live data-fetching endpoints from being abused. Browsing the map, table, and charts stays open to everyone.</p>

      <ClientOnly>
        <template #fallback>
          <p class="loading">Loading…</p>
        </template>

      <p v-if="urlError" class="notice err-notice">{{ urlError }}</p>

      <p v-if="!configured" class="notice">
        Auth isn’t configured for this deployment. Set <code>NUXT_PUBLIC_SUPABASE_URL</code> and
        <code>NUXT_PUBLIC_SUPABASE_ANON_KEY</code> to enable sign-in. Fetching runs unauthenticated until then.
      </p>

      <template v-else-if="isAuthed">
        <p class="signed">Signed in as <strong>{{ user.email || user.id }}</strong>.</p>
        <div class="row">
          <NuxtLink to="/data" class="primary as-link">Go to Data</NuxtLink>
          <button class="ghost" @click="doSignOut">Sign out</button>
        </div>
        <button class="oauth-btn passkey-add" :disabled="busy" @click="addPasskey">
          <IconPasskey class="ico" /> <span>Add a passkey to this account</span>
        </button>
        <p v-if="msg" :class="['msg', ok ? 'ok' : 'err']">{{ msg }}</p>
      </template>

      <template v-else>
        <!-- Passkey + OAuth -->
        <div class="oauth">
          <button class="oauth-btn" :disabled="busy" @click="passkey">
            <IconPasskey class="ico" /> <span>Sign in with a passkey</span>
          </button>
          <button class="oauth-btn" @click="oauth('github')">
            <IconGithub class="ico" /> <span>Continue with GitHub</span>
          </button>
          <button class="oauth-btn" @click="oauth('google')">
            <IconGoogle class="ico" /> <span>Continue with Google</span>
          </button>
        </div>

        <div class="divider"><span>or with email</span></div>

        <!-- Email + password / magic link -->
        <form @submit.prevent="submit">
          <label>Email
            <input v-model="email" type="email" autocomplete="email" required placeholder="you@example.com" />
          </label>
          <label v-if="mode !== 'magic'">Password
            <input v-model="password" type="password" :autocomplete="mode === 'signup' ? 'new-password' : 'current-password'" required minlength="6" placeholder="••••••••" />
          </label>

          <button class="primary" type="submit" :disabled="busy">
            {{ busy ? 'Working…' : submitLabel }}
          </button>
        </form>

        <div class="modes">
          <button :class="{ on: mode === 'signin' }" @click="mode = 'signin'">Password sign in</button>
          <button :class="{ on: mode === 'signup' }" @click="mode = 'signup'">Create account</button>
          <button :class="{ on: mode === 'magic' }" @click="mode = 'magic'">Magic link</button>
        </div>

        <p v-if="msg" :class="['msg', ok ? 'ok' : 'err']">{{ msg }}</p>
      </template>
      </ClientOnly>

      <p class="policy-link">
        By signing in you agree to our <NuxtLink to="/terms">Terms</NuxtLink>
        and <NuxtLink to="/privacy">Privacy Policy</NuxtLink>.
      </p>
    </div>
  </div>
</template>

<script setup>
const { user, isAuthed, configured, signInWithOtp, signInWithPassword, signUp, signInWithOAuth, signInWithPasskey, registerPasskey, signOut } = useAuth()

const route = useRoute()
const mode = ref(route.query.mode === 'signup' ? 'signup' : 'signin') // 'signin' | 'signup' | 'magic'
const email = ref('')
const password = ref('')
const busy = ref(false)
const msg = ref('')
const ok = ref(false)

// OAuth/magic-link failures come back in the return URL (query for PKCE, hash
// for implicit). Surface them persistently — otherwise the redirect lands on a
// fresh page with no context and the reason for a failed GitHub/Google sign-in
// is lost. Then strip them from the URL so a refresh doesn't keep showing them.
const urlError = ref('')
onMounted(() => {
  const loc = window.location
  const q = new URLSearchParams(loc.search)
  const h = new URLSearchParams(loc.hash.replace(/^#/, ''))
  const desc = q.get('error_description') || h.get('error_description')
  const code = q.get('error') || h.get('error')
  if (desc || code) {
    urlError.value = decodeURIComponent(desc || code).replace(/\+/g, ' ')
    history.replaceState({}, '', loc.pathname)
  }
})

const submitLabel = computed(() => ({
  signin: 'Sign in',
  signup: 'Create account',
  magic: 'Send magic link',
}[mode.value]))

async function submit() {
  busy.value = true
  msg.value = ''
  try {
    if (mode.value === 'magic') {
      await signInWithOtp(email.value.trim())
      ok.value = true
      msg.value = 'Check your email for the sign-in link.'
    } else if (mode.value === 'signup') {
      await signUp(email.value.trim(), password.value)
      ok.value = true
      msg.value = 'Account created. Check your email to confirm, then sign in.'
    } else {
      await signInWithPassword(email.value.trim(), password.value)
      ok.value = true
      msg.value = 'Signed in.'
    }
  } catch (e) {
    ok.value = false
    msg.value = e.message || String(e)
  } finally {
    busy.value = false
  }
}

async function oauth(provider) {
  msg.value = ''
  try {
    await signInWithOAuth(provider)
    // Redirects away; nothing else to do.
  } catch (e) {
    ok.value = false
    msg.value = e.message || String(e)
  }
}

async function passkey() {
  busy.value = true
  msg.value = ''
  try {
    await signInWithPasskey()
    ok.value = true
    msg.value = 'Signed in with passkey.'
  } catch (e) {
    ok.value = false
    msg.value = e.message || String(e)
  } finally {
    busy.value = false
  }
}

async function addPasskey() {
  busy.value = true
  msg.value = ''
  try {
    await registerPasskey()
    ok.value = true
    msg.value = 'Passkey added — use “Sign in with a passkey” next time.'
  } catch (e) {
    ok.value = false
    msg.value = e.message || String(e)
  } finally {
    busy.value = false
  }
}

async function doSignOut() { await signOut() }
</script>

<style scoped>
.login { display: flex; justify-content: center; padding: 40px 16px; }
.card { width: 100%; max-width: 380px; border: 1px solid var(--border); border-radius: 12px; padding: 24px; background: var(--surface); }
h2 { margin: 0 0 4px; font-size: 1.2rem; }
.sub { margin: 0 0 16px; color: var(--muted); font-size: 0.82rem; }
.notice { background: #fff7ed; border: 1px solid #fed7aa; color: #9a3412; border-radius: 8px; padding: 10px 12px; font-size: 0.8rem; margin: 0 0 14px; }
.notice code { background: #ffedd5; padding: 1px 4px; border-radius: 4px; }
.err-notice { background: #fef2f2; border-color: #fecaca; color: #b00020; }
.loading { color: var(--muted); font-size: 0.85rem; }
.signed { font-size: 0.9rem; }

.oauth { display: flex; flex-direction: column; gap: 8px; }
.oauth-btn {
  display: flex; align-items: center; gap: 10px; border: 1px solid #d5dbe1; background: var(--surface);
  border-radius: 8px; padding: 10px 14px; font-size: 0.9rem; font-weight: 500; color: var(--text);
  cursor: pointer; transition: background 0.12s, border-color 0.12s, box-shadow 0.12s;
}
.oauth-btn:hover:not(:disabled) { background: var(--surface-2); border-color: #c3cbd3; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06); }
.oauth-btn:active:not(:disabled) { background: #eef1f4; }
.oauth-btn:disabled { opacity: 0.55; cursor: default; }
.oauth-btn .ico { flex: 0 0 18px; display: inline-flex; }
.oauth-btn > span { flex: 1; text-align: center; margin-right: 18px; }
.passkey-add { margin-top: 12px; }

.divider { display: flex; align-items: center; gap: 10px; margin: 16px 0; color: var(--muted); font-size: 0.75rem; }
.divider::before, .divider::after { content: ''; flex: 1; height: 1px; background: var(--border); }

form { display: flex; flex-direction: column; gap: 10px; }
label { display: flex; flex-direction: column; gap: 4px; font-size: 0.8rem; font-weight: 600; color: var(--text); }
input { border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px; font-size: 0.9rem; font-weight: 400; }

.primary { border: 1px solid #2b7a3d; background: #2b7a3d; color: #fff; border-radius: 8px; padding: 9px 12px; font-size: 0.9rem; font-weight: 600; cursor: pointer; }
.primary:disabled { opacity: 0.6; cursor: default; }
.primary.as-link { text-decoration: none; text-align: center; }
.ghost { border: 1px solid var(--border); background: var(--surface); border-radius: 8px; padding: 9px 12px; font-size: 0.9rem; cursor: pointer; }
.row { display: flex; gap: 10px; margin-top: 8px; }

.modes { display: flex; gap: 6px; margin-top: 14px; flex-wrap: wrap; }
.modes button { flex: 1; border: 1px solid var(--border); background: var(--surface-2); color: var(--text); border-radius: 6px; padding: 6px 8px; font-size: 0.76rem; cursor: pointer; }
.modes button.on { border-color: #2b7a3d; color: #2b7a3d; background: #f0fdf4; font-weight: 600; }

.msg { margin: 12px 0 0; font-size: 0.82rem; }
.msg.ok { color: #2b7a3d; }
.msg.err { color: #b00020; }

.policy-link { margin: 16px 0 0; font-size: 0.76rem; color: var(--muted); text-align: center; }
.policy-link a { color: var(--muted); }
</style>
