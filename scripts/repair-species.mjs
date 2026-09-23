#!/usr/bin/env node
// One-off script: re-fetch the correct species name for every observation in
// the Supabase Storage observations.geojson by querying iNaturalist in batches.
//
// Usage:
//   node scripts/repair-species.mjs
//
// Reads SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) from
// .env in the project root. Writes the fixed file back to Supabase Storage and
// saves a local backup at scripts/observations-repaired.geojson.
//
// Throttle: 1 request/second to stay well inside iNat's 100 req/min limit.
// ETA for 48k records: ~4 minutes (240 batches × 1s).

import { readFileSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createClient } from '@supabase/supabase-js'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')

// ── Load .env ────────────────────────────────────────────────────────────────
function loadEnv() {
  try {
    const lines = readFileSync(join(root, '.env'), 'utf8').split('\n')
    for (const line of lines) {
      const m = line.match(/^\s*([A-Z_][A-Z0-9_]*)=(.*)$/)
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim().replace(/^['"]|['"]$/g, '')
    }
  } catch {
    console.error('No .env found — relying on existing environment variables.')
  }
}
loadEnv()

const SUPABASE_URL = process.env.SUPABASE_URL
const SUPABASE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY
const BUCKET = process.env.SUPABASE_DATASETS_BUCKET || 'datasets'
const STORAGE_PATH = 'observations.geojson'
const INAT_API = 'https://api.inaturalist.org/v1/observations'
const BATCH_SIZE = 200
const DELAY_MS = 1100  // ~1 req/sec

if (!SUPABASE_URL || !SUPABASE_KEY) {
  console.error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_ANON_KEY) are required.')
  process.exit(1)
}

const sb = createClient(SUPABASE_URL, SUPABASE_KEY, { auth: { persistSession: false } })

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)) }

// ── Download from Supabase Storage ───────────────────────────────────────────
async function download() {
  console.log(`Downloading ${STORAGE_PATH} from Supabase Storage bucket "${BUCKET}"…`)
  const { data, error } = await sb.storage.from(BUCKET).download(STORAGE_PATH)
  if (error) throw new Error(`Download failed: ${error.message}`)
  const text = await data.text()
  const geojson = JSON.parse(text)
  console.log(`  ${geojson.features?.length ?? 0} features loaded.`)
  return geojson
}

// ── Batch fetch species from iNat ────────────────────────────────────────────
async function fetchSpeciesBatch(ids) {
  const url = `${INAT_API}?id=${ids.join(',')}&per_page=${ids.length}&only_id=false`
  const res = await fetch(url, {
    headers: { 'User-Agent': 'data-map-repair/1.0 (+https://github.com/skyfly200/data-map)' },
  })
  if (res.status === 429) {
    console.warn('  429 rate-limited — waiting 60s…')
    await sleep(60000)
    return fetchSpeciesBatch(ids)
  }
  if (!res.ok) throw new Error(`iNat HTTP ${res.status} for batch starting ${ids[0]}`)
  const data = await res.json()
  const map = new Map()
  for (const obs of data.results || []) {
    if (obs.id && obs.taxon?.name) map.set(obs.id, obs.taxon.name)
  }
  return map
}

// ── Upload fixed file back to Supabase Storage ────────────────────────────────
async function upload(geojson) {
  console.log(`Uploading fixed file back to Supabase Storage…`)
  const body = JSON.stringify(geojson)
  const { error } = await sb.storage.from(BUCKET).upload(STORAGE_PATH, body, {
    contentType: 'application/geo+json',
    upsert: true,
  })
  if (error) throw new Error(`Upload failed: ${error.message}`)
  console.log('  Upload complete.')
}

// ── Main ──────────────────────────────────────────────────────────────────────
async function main() {
  const geojson = await download()
  const features = geojson.features || []

  // Collect inat_ids, skipping features without one
  const withId = features.filter((f) => f.properties?.inat_id != null)
  const withoutId = features.filter((f) => f.properties?.inat_id == null)
  console.log(`  ${withId.length} features have inat_id, ${withoutId.length} do not (will be left as-is).`)

  // Build id → feature index map
  const byId = new Map()
  for (const f of withId) byId.set(Number(f.properties.inat_id), f)

  const ids = [...byId.keys()]
  const batches = []
  for (let i = 0; i < ids.length; i += BATCH_SIZE) batches.push(ids.slice(i, i + BATCH_SIZE))

  console.log(`\nFetching species for ${ids.length} observations in ${batches.length} batches…`)

  let fixed = 0, notFound = 0

  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i]
    process.stdout.write(`  Batch ${i + 1}/${batches.length} (ids ${batch[0]}–${batch[batch.length - 1]})… `)
    const speciesMap = await fetchSpeciesBatch(batch)

    for (const id of batch) {
      const f = byId.get(id)
      if (!f) continue
      const species = speciesMap.get(id)
      if (species) {
        f.properties.species = species
        // Re-derive genus from the corrected species name
        f.properties.genus = species.trim().split(/\s+/)[0]
        fixed++
      } else {
        notFound++
      }
    }

    console.log(`${speciesMap.size} resolved, ${batch.length - speciesMap.size} not found`)

    if (i < batches.length - 1) await sleep(DELAY_MS)
  }

  console.log(`\nResults: ${fixed} fixed, ${notFound} not found in iNat (deleted/obscured), ${withoutId.length} had no inat_id.`)

  // Save local backup first
  const backupPath = join(here, 'observations-repaired.geojson')
  writeFileSync(backupPath, JSON.stringify(geojson))
  console.log(`Local backup saved to ${backupPath}`)

  await upload(geojson)
  console.log('\nDone. The refreshed observations.geojson is live in Supabase Storage.')
}

main().catch((err) => { console.error(err); process.exit(1) })
