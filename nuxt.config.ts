// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: true },
  experimental: {
    appManifest: false,
  },
  vite: {
    build: {
      rollupOptions: {
        onwarn(warning, warn) {
          if (warning.code === 'UNUSED_EXTERNAL_IMPORT') return
          warn(warning)
        },
      },
    },
  },
  routeRules: {
    '/analysis': { redirect: '/charts?tab=analysis' },
  },
  nitro: {
    preset: 'netlify',
    // The observation GeoJSON is ~48 MB of highly repetitive JSON that gzips to
    // under 7 MB. Without this it ships uncompressed: Nitro serves public/
    // assets byte-for-byte, so every visitor downloaded the full 48 MB.
    compressPublicAssets: { gzip: true, brotli: true },
  },
  runtimeConfig: {
    public: {
      // Where the frontend loads the dataset manifest from. Defaults to the
      // committed file; set NUXT_PUBLIC_DATASETS_MANIFEST_URL to a Supabase
      // Storage public URL to serve datasets from Supabase instead.
      datasetsManifestUrl: '/data/datasets.json',
      // Only used to make og:image absolute for social-card scrapers, which do
      // not resolve relative URLs. Everything else in the app is root-relative,
      // so leaving this empty is fine and keeps the build domain-agnostic. Set
      // NUXT_PUBLIC_SITE_URL to the public origin (no trailing slash) if link
      // previews matter.
      siteUrl: '',
      // Supabase Auth (browser). Safe to expose — the anon key is public by
      // design; the service role key stays server-only. Set via
      // NUXT_PUBLIC_SUPABASE_URL / NUXT_PUBLIC_SUPABASE_ANON_KEY. When empty,
      // the login UI shows a "not configured" notice and API fetches run
      // unauthenticated (which the functions allow only when unconfigured).
      supabaseUrl: '',
      supabaseAnonKey: '',
      // The storage bucket holding job results and saved datasets. One source
      // of truth: the browser downloads its own results from here, the server
      // writes them through SUPABASE_DATASETS_BUCKET, and the row-level
      // security policy in migration 005 names it as a literal. Renaming it
      // means setting NUXT_PUBLIC_DATASETS_BUCKET and SUPABASE_DATASETS_BUCKET
      // together AND editing that policy — which is why the default is the
      // recommended arrangement.
      datasetsBucket: 'datasets',
    },
  },
})
