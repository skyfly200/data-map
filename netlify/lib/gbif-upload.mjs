// Limits for the GBIF archive upload (server/routes/api/datasets/import-gbif).
//
// The route parses the whole file in memory, so every limit here is about
// keeping one request from exhausting the function: the upload itself, what
// the zip says it will expand to, and how many rows become features.

import { unzipSync } from 'fflate'

/** Netlify's synchronous function payload limit is 6 MB; refuse before reading. */
export const MAX_UPLOAD_BYTES = 6 * 1024 * 1024
/** Decompressed size of any one archive entry. */
export const MAX_ENTRY_BYTES = 60 * 1024 * 1024
/** Decompressed size of everything extracted from one archive. */
export const MAX_TOTAL_BYTES = 120 * 1024 * 1024
/** Occurrences kept from one upload. */
export const MAX_RECORDS = 50_000

export class GbifUploadError extends Error {
  constructor(message, status = 400) {
    super(message)
    this.status = status
  }
}

/**
 * Unzip only the entries `wanted(name)` accepts, refusing an entry or a total
 * that would expand past the limits. Sizes come from the zip's own headers and
 * fflate inflates into a buffer of exactly that size, so a lying header cannot
 * produce more than it declared.
 */
export function safeUnzip(data, wanted, { maxEntry = MAX_ENTRY_BYTES, maxTotal = MAX_TOTAL_BYTES } = {}) {
  let total = 0
  return unzipSync(data, {
    filter(file) {
      if (!wanted(file.name)) return false
      if (file.originalSize > maxEntry) {
        throw new GbifUploadError(`"${file.name}" expands to more than ${Math.round(maxEntry / 1048576)} MB. Narrow the GBIF download.`, 413)
      }
      total += file.originalSize
      if (total > maxTotal) {
        throw new GbifUploadError(`The archive expands to more than ${Math.round(maxTotal / 1048576)} MB. Narrow the GBIF download.`, 413)
      }
      return true
    },
  })
}

/** Does a zip entry path name this file, at the root or one folder down? */
export function entryMatches(path, name) {
  return path === name || path.endsWith(`/${name}`)
}
