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
    },
  },
})
