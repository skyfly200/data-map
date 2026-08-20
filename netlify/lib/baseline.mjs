// Load the committed, fully-enriched baseline GeoJSON that the offline Python
// pipeline produces (public/data/observations.geojson). Bundled into the
// functions via `included_files` in netlify.toml. Path resolution differs
// between local dev and the deployed runtime, so several candidates are tried.

import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const REL = 'public/data/observations.geojson'
const here = dirname(fileURLToPath(import.meta.url))

const CANDIDATES = [
  join(process.cwd(), REL),
  join(here, '../../', REL),
  join(here, '../..', 'public/data/observations.geojson'),
  `/var/task/${REL}`,
]

export async function loadBaseline() {
  for (const path of CANDIDATES) {
    try {
      const raw = await readFile(path, 'utf8')
      return JSON.parse(raw)
    } catch {
      // try the next candidate
    }
  }
  return { type: 'FeatureCollection', features: [] }
}
