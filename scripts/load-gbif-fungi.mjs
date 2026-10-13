// Admin load of GBIF fungi into the main observations dataset, no queue/timeout.
// The admin page runs the same thing as a gbif_ingest job; use this from a shell.
//   node scripts/load-gbif-fungi.mjs [--max-uncertainty 5000] [--min-year 1970] [--write]
// Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY.
//
// Dry run by default: prints counts. Re-running replaces earlier GBIF features,
// so it is idempotent. A full pipeline re-export (upload_datasets.mjs)
// overwrites observations.geojson, so re-run this after one.

import { readJson, uploadJson } from '../netlify/lib/datasets-store.mjs'
import { normaliseGbifSpec, runGbifIngest } from '../netlify/lib/gbif-bulk.mjs'

const args = process.argv.slice(2)
const flag = (k) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : undefined }

const spec = normaliseGbifSpec({
  maxUncertaintyM: flag('max-uncertainty'),
  minYear: flag('min-year'),
  dryRun: !args.includes('--write'),
})
const result = await runGbifIngest({
  spec, readJson, uploadJson, fetchImpl: fetch,
  onProgress: ({ message }) => process.stdout.write(`\r${message}`),
})
console.log(`\n${JSON.stringify(result, null, 2)}`)
if (spec.dryRun) console.log('Dry run. Pass --write to upload.')
