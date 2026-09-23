#!/usr/bin/env node
// One-off script: rename an entry in the datasets.json manifest in Supabase Storage.
//
// Usage:
//   node scripts/rename-dataset.mjs <id> <new-title>
//
// Example:
//   node scripts/rename-dataset.mjs agaricus-abruptibulbus "Baseline"
//
// Reads SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY from .env in the project root.

import { readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const root = join(here, '..')

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

const [,, id, newTitle] = process.argv
if (!id || !newTitle) {
  console.error('Usage: node scripts/rename-dataset.mjs <id> <new-title>')
  process.exit(1)
}

const { readJson, uploadJson } = await import('../netlify/lib/datasets-store.mjs')

const manifest = await readJson('datasets.json')
if (!manifest) { console.error('Could not read datasets.json from Supabase Storage.'); process.exit(1) }

console.log('Current manifest:')
manifest.forEach((e) => console.log(`  ${e.id}: "${e.label}"`))

const idx = manifest.findIndex((e) => e.id === id)
if (idx < 0) { console.error(`No entry found with id "${id}".`); process.exit(1) }

const oldLabel = manifest[idx].label
manifest[idx] = { ...manifest[idx], label: newTitle }

await uploadJson('datasets.json', manifest, 'application/json')
console.log(`\nRenamed "${oldLabel}" → "${newTitle}"`)
