// Render every Earth Engine layer against the real API, once, and report.
//
// The catalogue's asset IDs, band names and system:index values were written
// from documentation and from Code Editor scripts, never executed against Earth
// Engine from this codebase — the verification debt in docs/roadmap.md. A layer
// with a wrong band comes back blank, which on a map reads as ground with
// nothing on it rather than as an error, so the only way to discharge the debt
// is to mint each one and look.
//
// This is that pass, made repeatable. With Earth Engine credentials in the
// environment (EARTHENGINE_SERVICE_ACCOUNT_KEY and EARTHENGINE_PROJECT — see
// netlify/lib/ee-runner.mjs) it resolves each layer at its default parameters,
// runs its prepare step and its empty-window count where it has them, mints a
// tile template, and prints a pass/fail line per layer. A non-zero exit means at
// least one layer did not render.
//
//   node scripts/verify_ee_layers.mjs            every layer
//   node scripts/verify_ee_layers.mjs soil       only keys containing "soil"
//
// It does not fetch the tiles themselves — a mint that succeeds proves the
// asset, bands and visualisation are valid; whether the pixels look right is the
// eyeball step this cannot replace, and the script says so at the end.

import ee from '@google/earthengine'

import { earthEngineConfigured, initEarthEngine, evaluate } from '../netlify/lib/ee-runner.mjs'
import {
  EE_LAYER_KEYS, EE_TILE_LAYERS, resolveLayer, visParams,
} from '../netlify/lib/ee-tile-layers.mjs'

/** The layers this run will check, honouring an optional name filter. */
export function layersToCheck(filter = '') {
  const needle = String(filter || '').toLowerCase()
  return EE_LAYER_KEYS.filter((key) => !needle || key.toLowerCase().includes(needle))
}

/** Mint a tile template for an image, as a promise. getMapId is callback-shaped. */
function mint(image, vis) {
  return new Promise((resolve, reject) => {
    ee.data.getMapId({ image, ...visParams(vis) }, (result, err) => {
      if (err) return reject(new Error(String(err)))
      const template = result?.urlFormat || (result?.mapid ? `mapid:${result.mapid}` : null)
      if (!template) return reject(new Error('no tile URL returned'))
      return resolve(template)
    })
  })
}

/** Render one layer at its defaults and say whether it worked. */
async function checkLayer(key) {
  const layer = EE_TILE_LAYERS[key]
  try {
    const { params } = resolveLayer(key, {})
    const prepared = layer.prepare ? await evaluate(layer.prepare(ee)) : null
    if (layer.prepare && (!prepared || !Array.isArray(prepared.values) || !prepared.values.length)) {
      return { key, ok: false, error: 'prepare returned no class table' }
    }
    // A layer with a count and an empty window fails loudly rather than blank;
    // at defaults it should not be empty, so a zero here is a real fault.
    if (layer.count) {
      const n = await evaluate(layer.count(ee, params))
      if (!n) return { key, ok: false, error: 'count is zero at default parameters' }
    }
    const { image, vis } = layer.build(ee, params, prepared)
    await mint(image, vis)
    return { key, ok: true }
  } catch (err) {
    return { key, ok: false, error: String(err?.message || err).slice(0, 200) }
  }
}

async function main() {
  const filter = process.argv[2] || ''
  const keys = layersToCheck(filter)
  if (!keys.length) {
    console.error(`No layers match “${filter}”.`)
    process.exit(2)
  }
  if (!earthEngineConfigured()) {
    console.error('Earth Engine is not configured. Set EARTHENGINE_SERVICE_ACCOUNT_KEY and '
      + 'EARTHENGINE_PROJECT, then run this again.')
    process.exit(3)
  }

  await initEarthEngine()
  console.log(`Checking ${keys.length} layer${keys.length > 1 ? 's' : ''} against Earth Engine…\n`)

  const results = []
  for (const key of keys) {
    // Serial on purpose: a pile of concurrent getMapId calls is exactly the
    // throttling the tile path is careful to avoid, and this is a rare pass.
    const result = await checkLayer(key) // eslint-disable-line no-await-in-loop
    results.push(result)
    console.log(`${result.ok ? '  ok  ' : 'FAIL  '}${key}${result.ok ? '' : ` — ${result.error}`}`)
  }

  const failed = results.filter((r) => !r.ok)
  console.log(`\n${results.length - failed.length}/${results.length} layers rendered.`)
  console.log('A mint proves the asset, bands and palette are valid. It does not prove the '
    + 'pixels are right — open the ones that passed and look before trusting them.')
  process.exit(failed.length ? 1 : 0)
}

// Only run when invoked directly, so layersToCheck can be imported by a test.
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error(err)
    process.exit(1)
  })
}
