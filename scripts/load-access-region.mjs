// One-time admin load of an access region (default: colorado), no queue/timeout.
//   node scripts/load-access-region.mjs [--region colorado] [--bbox w,s,e,n --name label]
//        [--sources padus,osm,ridb] [--force]
// Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY; optional RIDB_API_KEY, PADUS_FEATURE_URL, OVERPASS_URL.
// Skips regions already loaded unless --force. Upserts are idempotent, so a failed run can be re-run.

import { adminClient } from '../netlify/lib/auth.mjs'
import { findCoveringRegion, listRegions, normaliseAccessSpec, runAccessIngest } from '../netlify/lib/access-regions.mjs'

const args = process.argv.slice(2)
const flag = (k) => { const i = args.indexOf(`--${k}`); return i >= 0 ? args[i + 1] : undefined }

const client = adminClient()
if (!client) { console.error('Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY.'); process.exit(1) }

const spec = normaliseAccessSpec({
  region: flag('name') || flag('region') || 'colorado',
  bbox: flag('bbox'),
  sources: flag('sources')?.split(','),
  force: args.includes('--force'),
})
const covering = findCoveringRegion(await listRegions(client), spec.bbox)
if (covering && !spec.force) {
  console.log(`Already loaded as "${covering.name}" (use --force to reload).`)
  process.exit(0)
}
if (!process.env.RIDB_API_KEY && spec.sources.includes('ridb')) console.warn('RIDB_API_KEY not set: fee overlay skipped, estimates only.')

const result = await runAccessIngest({
  client, spec, onProgress: async ({ fraction, message }) => console.log(`${Math.round(fraction * 100)}% ${message}`),
})
console.log(JSON.stringify({ region: spec.region, ...result }))
