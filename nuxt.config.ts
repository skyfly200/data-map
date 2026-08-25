// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },
  runtimeConfig: {
    public: {
      // Where the frontend loads the dataset manifest from. Defaults to the
      // committed file; set NUXT_PUBLIC_DATASETS_MANIFEST_URL to a Supabase
      // Storage public URL to serve datasets from Supabase instead.
      datasetsManifestUrl: '/data/datasets.json',
      // Supabase Auth (browser). Safe to expose — the anon key is public by
      // design; the service role key stays server-only. Set via
      // NUXT_PUBLIC_SUPABASE_URL / NUXT_PUBLIC_SUPABASE_ANON_KEY. When empty,
      // the login UI shows a "not configured" notice and API fetches run
      // unauthenticated (which the functions allow only when unconfigured).
      supabaseUrl: '',
      supabaseAnonKey: '',
    },
  },
})
