// Admin load of GBIF fungi into the main observations dataset, no queue/timeout.
//   node scripts/load-gbif-fungi.mjs [--bbox w,s,e,n] [--from 1900] [--to 2026]
//        [--max-uncertainty 5000] [--min-year 1970] [--out file.geojson] [--write]
// Env (for --write, and to dedupe against the live dataset): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
//
// Pages GBIF's public occurrence search (no account needed), skips GBIF's copy of
// iNaturalist (we already have those) and merges everything else into
// observations.geojson. Dry run by default: prints counts and writes --out if
// given. Re-running replaces earlier GBIF features, so it is idempotent. A full
// pipeline re-export (upload_datasets.mjs) overwrites observations.geojson, so
// re-run this after one.

import { writeFile } from 'node:fs/promises'
import { REGIONS } from '../netlify/lib/access-regions.mjs'
import { supabaseConfigured } from '../netlify/lib/supabase-storage.mjs'
import { readJson, uploadJson } from '../netlify/lib/datasets-store.mjs'
import {
  GBIF_SEARCH, PER_PAGE, DEFAULT_MAX_UNCERTAINTY_M,
  searchParams, partitionYears, dropReason, gbifRecordToFeature, mergeGbif,
} from '../netlify/lib/gbif-bulk.mjs'

const args = process.argv.slice(2)
const flag = (k) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : undefined }
const num = (k, d) => (flag(k) != null ? Number(flag(k)) : d)

const bbox = flag('bbox') ? flag('bbox').split(',').map(Number) : REGIONS.colorado.bbox
const yearRange = [num('from', 1800), num('to', new Date().getUTCFullYear())]
const filters = { maxUncertaintyM: num('max-uncertainty', DEFAULT_MAX_UNCERTAINTY_M), minYear: num('min-year', null) }
const write = args.includes('--write')

async function getJson(url, tries = 4) {
  for (let i = 0; ; i++) {
    try {
      const res = await fetch(url)
      if (res.ok) return await res.json()
      if (res.status < 500 && res.status !== 429) throw new Error(`GBIF ${res.status}`)
    } catch (e) {
      if (i >= tries) throw e
    }
    if (i >= tries) throw new Error(`GBIF failed after ${tries + 1} tries`)
    await new Promise((r) => setTimeout(r, 2000 * 2 ** i))
  }
}

async function count(years) {
  const p = searchParams({ bbox, years })
  p.set('limit', '0')
  return (await getJson(`${GBIF_SEARCH}?${p}`)).count ?? 0
}

const parts = await partitionYears(count, yearRange)
const total = parts.reduce((a, p) => a + p.count, 0)
console.log(`GBIF fungi in [${bbox.join(', ')}], ${yearRange.join('–')}: ${total} records in ${parts.length} quer${parts.length === 1 ? 'y' : 'ies'}`)

const features = []
const dropped = {}
let seen = 0
for (const { years, count: n } of parts) {
  const p = searchParams({ bbox, years })
  p.set('limit', String(PER_PAGE))
  for (let offset = 0; offset < n; offset += PER_PAGE) {
    p.set('offset', String(offset))
    const data = await getJson(`${GBIF_SEARCH}?${p}`)
    for (const rec of data.results || []) {
      const why = dropReason(rec, filters)
      if (why) dropped[why] = (dropped[why] || 0) + 1
      else features.push(gbifRecordToFeature(rec))
    }
    seen += data.results?.length || 0
    process.stdout.write(`\r${seen}/${total} read, ${features.length} kept`)
    if (data.endOfRecords) break
  }
}
console.log(`\nDropped: ${JSON.stringify(dropped)}`)

const baseline = supabaseConfigured() ? await readJson('observations.geojson') : null
if (!baseline) console.warn('Live observations.geojson not readable: merging into an empty collection.')
const { collection, added, replaced } = mergeGbif(baseline, features)
console.log(`Merge: ${baseline?.features?.length ?? 0} existing, ${replaced} earlier GBIF replaced, ${added} GBIF added, ${collection.features.length} total`)

if (flag('out')) {
  await writeFile(flag('out'), JSON.stringify(collection))
  console.log(`Wrote ${flag('out')}`)
}
if (write) {
  if (!baseline) { console.error('Refusing to --write without the live dataset to merge into.'); process.exit(1) }
  await uploadJson('observations.geojson', collection)
  console.log('↑ observations.geojson')
} else {
  console.log('Dry run. Pass --write to upload.')
}
