// Supabase Storage config for the dataset pipeline (GeoJSON files in a bucket).
// Separate from supabase.mjs, which handles the observations *table* upserts.
// Everything is optional: with no SUPABASE_* env set, callers fall back to the
// committed public/data files and Netlify Blobs, so the app works unconfigured.
//
// Env:
//   SUPABASE_URL                 project URL, e.g. https://<ref>.supabase.co
//   SUPABASE_SERVICE_ROLE_KEY    write access (functions/scripts only — never ship to the browser)
//   SUPABASE_ANON_KEY            read access (optional; service key used if absent)
//   SUPABASE_DATASETS_BUCKET     storage bucket name (default: datasets)

import { createClient } from '@supabase/supabase-js'

export const DATASETS_BUCKET = process.env.SUPABASE_DATASETS_BUCKET || 'datasets'

export function supabaseConfigured() {
  return Boolean(process.env.SUPABASE_URL
    && (process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY))
}

// Write-capable client. Only call from trusted server contexts.
export function serviceClient() {
  const url = process.env.SUPABASE_URL
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!url || !key) return null
  return createClient(url, key, { auth: { persistSession: false } })
}

// Read client (anon key preferred, service key as fallback).
export function readClient() {
  const url = process.env.SUPABASE_URL
  const key = process.env.SUPABASE_ANON_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY
  if (!url || !key) return null
  return createClient(url, key, { auth: { persistSession: false } })
}

// Public object URL (works when the bucket is public).
export function publicUrl(path) {
  const url = process.env.SUPABASE_URL
  if (!url) return null
  return `${url}/storage/v1/object/public/${DATASETS_BUCKET}/${String(path).replace(/^\/+/, '')}`
}
