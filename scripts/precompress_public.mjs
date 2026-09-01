/**
 * Pre-compress large public assets that Nitro will not compress on its own.
 *
 * Nitro's `compressPublicAssets` only touches a fixed allowlist of MIME types.
 * `.geojson` resolves to `application/geo+json`, which is not on it, so the
 * observation dataset — the single biggest thing the app serves, ~48 MB that
 * gzips to under 7 MB — shipped uncompressed while every `.json` beside it was
 * compressed.
 *
 * Writing `.gz`/`.br` siblings *before* `nuxt build` runs is enough: Nitro copies
 * them into `.output/public`, indexes them in its static-asset manifest with the
 * right `encoding`, and serves them to any client sending `Accept-Encoding`.
 * Nitro also skips re-compressing a file that already has siblings, so this
 * neither fights with nor duplicates its work.
 *
 * Wired up as the `prebuild` npm script, so `npm run build` (and therefore the
 * Netlify deploy) picks it up with no extra step.
 */
import { createReadStream, createWriteStream } from 'node:fs'
import { readdir, stat } from 'node:fs/promises'
import { join, extname } from 'node:path'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'
import zlib from 'node:zlib'

const ROOT = fileURLToPath(new URL('..', import.meta.url))
const PUBLIC_DIR = join(ROOT, 'public')

// Extensions Nitro leaves uncompressed but that are plain text underneath.
const EXTENSIONS = new Set(['.geojson'])
// Below this, the compressed sibling costs more than it saves.
const MIN_BYTES = 1024

async function* walk(dir) {
  let entries
  try {
    entries = await readdir(dir, { withFileTypes: true })
  } catch {
    return // no public/ dir yet — nothing to do
  }
  for (const entry of entries) {
    const full = join(dir, entry.name)
    if (entry.isDirectory()) yield* walk(full)
    else if (entry.isFile()) yield full
  }
}

async function isStale(source, target) {
  try {
    const [src, dst] = await Promise.all([stat(source), stat(target)])
    return src.mtimeMs > dst.mtimeMs
  } catch {
    return true // no sibling yet
  }
}

async function compress(source, target, stream) {
  await pipeline(createReadStream(source), stream, createWriteStream(target))
  return (await stat(target)).size
}

const mb = (n) => `${(n / 1048576).toFixed(1)} MB`

let compressed = 0
for await (const file of walk(PUBLIC_DIR)) {
  if (!EXTENSIONS.has(extname(file).toLowerCase())) continue
  const { size } = await stat(file)
  if (size < MIN_BYTES) continue

  const rel = file.slice(PUBLIC_DIR.length + 1)
  const targets = [
    [`${file}.gz`, () => zlib.createGzip({ level: zlib.constants.Z_BEST_COMPRESSION })],
    [`${file}.br`, () => zlib.createBrotliCompress({
      params: {
        [zlib.constants.BROTLI_PARAM_MODE]: zlib.constants.BROTLI_MODE_TEXT,
        [zlib.constants.BROTLI_PARAM_QUALITY]: zlib.constants.BROTLI_MAX_QUALITY,
        [zlib.constants.BROTLI_PARAM_SIZE_HINT]: size,
      },
    })],
  ]

  const written = []
  for (const [target, makeStream] of targets) {
    // Rebuilding a 48 MB brotli stream on every build is slow and pointless when
    // the dataset has not changed.
    if (!(await isStale(file, target))) continue
    written.push(`${target.endsWith('.br') ? 'br' : 'gz'} ${mb(await compress(file, target, makeStream()))}`)
  }

  if (written.length) {
    console.log(`  ${rel}: ${mb(size)} → ${written.join(', ')}`)
    compressed++
  }
}

console.log(compressed
  ? `Pre-compressed ${compressed} public asset(s) Nitro would have served raw.`
  : 'Public assets already pre-compressed; nothing to do.')
