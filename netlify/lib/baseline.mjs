// Load the observations GeoJSON used as the bbox-job baseline.
//
// Supabase Storage is the authoritative source when configured (the
// refresh-observations function keeps it current). The committed file at
// public/data/observations.geojson is the local-dev / no-Supabase fallback;
// several path candidates are tried because cwd differs between local dev and
// the deployed Netlify runtime.

import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'
import { supabaseConfigured } from './supabase-storage.mjs'
import { readJson } from './datasets-store.mjs'

const REL = 'public/data/observations.geojson'
const here = dirname(fileURLToPath(import.meta.url))

const CANDIDATES = [
  join(process.cwd(), REL),
  join(here, '../../', REL),
  join(here, '../..', 'public/data/observations.geojson'),
  `/var/task/${REL}`,
]

export async function loadBaseline() {
  if (supabaseConfigured()) {
    const data = await readJson('observations.geojson')
    if (data) return data
  }
  for (const path of CANDIDATES) {
    try {
      const raw = await readFile(path, 'utf8')
      return JSON.parse(raw)
    } catch {
      // try the next candidate
    }
  }
  return null
}
