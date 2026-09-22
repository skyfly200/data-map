// A module-resolution hook so `node --test` resolves the composables the same
// way the bundler does.
//
// The composables are authored for Vite/Nuxt: they import siblings without an
// extension (`./ramps`) or with a `.js` extension that Vite maps onto the `.ts`
// file (`./ramps.js`). Node's own ESM resolver does neither — it needs a literal
// path — so a test importing one of those chains failed with ERR_MODULE_NOT_FOUND
// even though the app builds fine.
//
// Type stripping itself is already built in (Node 22), so this only fixes
// RESOLUTION: given a relative specifier, prefer a real `.ts` file where the
// literal path does not exist. Source stays bundler-idiomatic; the test runner
// learns to follow it. Registered with `node --import ./tests/loader.mjs`.

import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

/** Candidate paths to try for one relative specifier, in order. */
function candidates(specifier) {
  if (specifier.endsWith('.ts')) return [specifier]
  if (specifier.endsWith('.js')) return [specifier.slice(0, -3) + '.ts', specifier]
  // Extensionless: a TypeScript source first, then a JavaScript one.
  return [specifier + '.ts', specifier + '.js', specifier]
}

export async function resolve(specifier, context, nextResolve) {
  // Only relative imports are ours to remap; bare specifiers are packages.
  if (specifier.startsWith('.') && context.parentURL) {
    for (const candidate of candidates(specifier)) {
      let url
      try {
        url = new URL(candidate, context.parentURL)
      } catch {
        continue
      }
      if (url.protocol === 'file:' && existsSync(fileURLToPath(url))) {
        return nextResolve(candidate, context)
      }
    }
  }
  return nextResolve(specifier, context)
}
