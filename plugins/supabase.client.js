// Browser-side Supabase client, created once and injected as `$supabase`.
// Only runs on the client (`.client.js`) so the session lives in the browser.
// When Supabase isn't configured (no public URL/key), we inject null and the
// auth UI degrades to a "not configured" notice instead of throwing.
import { createClient } from '@supabase/supabase-js'

export default defineNuxtPlugin(() => {
  const cfg = useRuntimeConfig().public
  if (!cfg.supabaseUrl || !cfg.supabaseAnonKey) {
    return { provide: { supabase: null } }
  }
  const client = createClient(cfg.supabaseUrl, cfg.supabaseAnonKey, {
    auth: {
      persistSession: true,
      autoRefreshToken: true,
      // Complete magic-link / OAuth redirects that land back with tokens in the URL.
      detectSessionInUrl: true,
      flowType: 'pkce',
    },
  })
  return { provide: { supabase: client } }
})
