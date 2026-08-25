// Read/write GeoJSON datasets in Supabase Storage. Thin helpers over the
// storage bucket; callers decide whether to use these (Supabase configured) or
// fall back to committed files / Netlify Blobs.

import { serviceClient, readClient, DATASETS_BUCKET } from './supabase-storage.mjs'

export async function uploadJson(path, obj, contentType = 'application/geo+json') {
  const client = serviceClient()
  if (!client) throw new Error('Supabase service client is not configured (SUPABASE_SERVICE_ROLE_KEY).')
  const body = JSON.stringify(obj)
  const { error } = await client.storage.from(DATASETS_BUCKET).upload(path, body, {
    contentType, upsert: true,
  })
  if (error) throw error
  return path
}

export async function readJson(path) {
  const client = readClient()
  if (!client) return null
  try {
    const { data, error } = await client.storage.from(DATASETS_BUCKET).download(path)
    if (error || !data) return null
    return JSON.parse(await data.text())
  } catch {
    return null
  }
}
