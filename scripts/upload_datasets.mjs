// Upload the exported datasets (public/data) to Supabase Storage so that
// Supabase becomes the single source of truth for observations.
//
// Run after export_geojson.py — this is the initial migration step that seeds
// observations.geojson into Supabase. Subsequent updates are written by
// netlify/functions/refresh-observations.mjs on its 6-hour schedule.
//
// No-ops (exit 0) when SUPABASE_* env is not configured, so it is safe to
// always include in the pipeline. Requires SUPABASE_SERVICE_ROLE_KEY to write.
//
// After running, set NUXT_PUBLIC_DATASETS_MANIFEST_URL to the printed URL so
// the client loads the manifest (and dataset GeoJSON paths) directly from
// Supabase Storage instead of the bundled /data/datasets.json file.
//
//   node scripts/upload_datasets.mjs

import { readFile, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { supabaseConfigured, publicUrl } from '../netlify/lib/supabase-storage.mjs'
import { uploadJson } from '../netlify/lib/datasets-store.mjs'

const DATA_DIR = 'public/data'

async function main() {
  if (!supabaseConfigured()) {
    console.log('Supabase not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY) — skipping upload.')
    return
  }

  const combined = JSON.parse(await readFile(join(DATA_DIR, 'observations.geojson'), 'utf8'))
  await uploadJson('observations.geojson', combined)
  console.log(`↑ observations.geojson (${combined.features?.length ?? 0} features)`)

  let speciesFiles = []
  try {
    speciesFiles = (await readdir(join(DATA_DIR, 'species'))).filter((f) => f.endsWith('.geojson'))
  } catch {
    // no species dir
  }
  for (const file of speciesFiles) {
    const fc = JSON.parse(await readFile(join(DATA_DIR, 'species', file), 'utf8'))
    await uploadJson(`species/${file}`, fc)
    console.log(`↑ species/${file} (${fc.features?.length ?? 0})`)
  }

  // Rewrite manifest paths to Supabase public URLs so the frontend can read
  // datasets directly from Storage.
  const manifest = JSON.parse(await readFile(join(DATA_DIR, 'datasets.json'), 'utf8'))
  const rewritten = manifest.map((d) => ({
    ...d,
    path: publicUrl(String(d.path).replace(/^\/data\//, '')) || d.path,
  }))
  await uploadJson('datasets.json', rewritten, 'application/json')
  console.log(`↑ datasets.json (${rewritten.length} datasets)`) // manifest last

  console.log('\n✅ Datasets uploaded. Point NUXT_PUBLIC_DATASETS_MANIFEST_URL at:')
  console.log(`   ${publicUrl('datasets.json')}`)
}

main().catch((e) => { console.error('[!] Upload failed:', e); process.exit(1) })
