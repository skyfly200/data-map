// Key/value storage abstraction used by the functions (model cache, tile cache,
// new-observations overlay). Replaces direct @netlify/blobs usage.
//
//   getStore(name) -> { get(key, {type}), set(key, value), setJSON(key, obj),
//                       delete(key), list({prefix}) }
//
// get: type 'json' | 'text' (default) | 'arrayBuffer' | 'blob'; missing -> null.
// set: string | ArrayBuffer | Uint8Array | Blob. TTLs live in the payload
// (callers store `expires`), so no metadata/TTL/consistency options exist.
// list: resolves { blobs: [{ key }] } (all pages), like @netlify/blobs.
//
// Backend (env STORAGE_BACKEND=supabase|netlify): default is netlify when
// running on Netlify (NETLIFY env set), supabase elsewhere.
// Supabase: one private bucket (SUPABASE_STORAGE_BUCKET, default = the datasets
// bucket), objects at `<store>/<key>`, via the service-role client.

import { serviceClient, DATASETS_BUCKET } from './supabase-storage.mjs'

const PAGE = 100

export function backendName(env = process.env) {
  const explicit = String(env.STORAGE_BACKEND || '').toLowerCase()
  if (explicit === 'supabase' || explicit === 'netlify') return explicit
  return env.NETLIFY ? 'netlify' : 'supabase'
}

function bucketName() {
  return process.env.SUPABASE_STORAGE_BUCKET || DATASETS_BUCKET
}

function toBody(value) {
  if (typeof value === 'string') return { body: value, contentType: 'text/plain;charset=UTF-8' }
  if (typeof Blob !== 'undefined' && value instanceof Blob) {
    return { body: value, contentType: value.type || 'application/octet-stream' }
  }
  return { body: value, contentType: 'application/octet-stream' }
}

export function supabaseStore(name, client = serviceClient()) {
  if (!client) throw new Error('Supabase storage is not configured (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY).')
  const bucket = () => client.storage.from(bucketName())
  const path = (key) => `${name}/${key}`

  async function upload(key, body, contentType) {
    const { error } = await bucket().upload(path(key), body, { contentType, upsert: true })
    if (error) throw error
  }

  return {
    async get(key, { type = 'text' } = {}) {
      const { data, error } = await bucket().download(path(key))
      if (error || !data) {
        const status = error?.status ?? error?.statusCode ?? error?.originalError?.status
        const missing = !error || status === 400 || status === 404 || /not.?found/i.test(String(error?.message))
        if (missing) return null
        throw error
      }
      if (type === 'json') return JSON.parse(await data.text())
      if (type === 'arrayBuffer') return data.arrayBuffer()
      if (type === 'blob') return data
      return data.text()
    },
    async set(key, value) {
      const { body, contentType } = toBody(value)
      await upload(key, body, contentType)
    },
    async setJSON(key, obj) {
      await upload(key, JSON.stringify(obj), 'application/json')
    },
    async delete(key) {
      const { error } = await bucket().remove([path(key)])
      if (error) throw error
    },
    async list({ prefix = '' } = {}) {
      // Supabase lists one folder level; split the prefix into folder + name filter.
      const slash = prefix.lastIndexOf('/')
      const dir = slash >= 0 ? prefix.slice(0, slash + 1) : ''
      const search = prefix.slice(slash + 1)
      const folder = `${name}/${dir}`.replace(/\/$/, '')
      const blobs = []
      for (let offset = 0; ; offset += PAGE) {
        const { data, error } = await bucket().list(folder, {
          limit: PAGE, offset, ...(search ? { search } : {}),
        })
        if (error) throw error
        for (const o of data || []) {
          if (o.id === null || o.id === undefined) continue // sub-folder placeholder
          if (search && !o.name.startsWith(search)) continue
          blobs.push({ key: dir + o.name })
        }
        if (!data || data.length < PAGE) break
      }
      return { blobs }
    },
  }
}

function netlifyStore(name) {
  // Lazy: the Vercel build/runtime never needs @netlify/blobs.
  let store
  const load = async () => {
    if (!store) store = (await import('@netlify/blobs')).getStore(name)
    return store
  }
  return {
    get: async (key, opts) => (await load()).get(key, opts),
    set: async (key, value) => (await load()).set(key, value),
    setJSON: async (key, obj) => (await load()).setJSON(key, obj),
    delete: async (key) => (await load()).delete(key),
    list: async (opts) => (await load()).list(opts),
  }
}

export function getStore(name) {
  return backendName() === 'netlify' ? netlifyStore(name) : supabaseStore(name)
}
