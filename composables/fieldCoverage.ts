// How much of a view a sparsely-populated field actually covers.
//
// Taxonomy above species is resolved from ancestry, and that pass has stalled:
// family and the ranks above it are populated for a small slice of the store.
// Genus and species are complete (genus is split from the binomial when it is
// missing), so a note on them would be noise; the ranks below carry it.
//
// A view grouped by a rank most records lack is not wrong, but it is a slice, and
// it must say so — a bar chart of families over 4% of the data next to one over
// all of it look the same and mean very different things. This computes the
// fraction, and coverageNote turns it into the sentence a view shows.

// Inlined rather than imported from useObservations: that module reaches for
// runtime-only aliases, and this needs to stay importable on its own (and in a
// plain test). Same definition as useObservations.hasValue.
const hasValue = (v: any): boolean => v !== null && v !== undefined && v !== ''

/** The ranks whose resolution has stalled, so a coverage note is worth showing. */
export const SPARSE_RANKS = new Set(['kingdom', 'phylum', 'class', 'order', 'family'])

/** A record, either a GeoJSON feature ({properties}) or a flattened row. */
function valueOf(row: any, field: string): unknown {
  if (row && typeof row === 'object' && row.properties) return row.properties[field]
  return row ? row[field] : undefined
}

export interface Coverage {
  present: number
  total: number
  fraction: number
}

/** What fraction of `rows` carry a value for `field`. */
export function fieldCoverage(rows: any[], field: string): Coverage {
  const total = rows?.length || 0
  if (!total) return { present: 0, total: 0, fraction: 0 }
  let present = 0
  for (const row of rows) if (hasValue(valueOf(row, field))) present += 1
  return { present, total, fraction: present / total }
}

/**
 * The sentence a view grouped by `field` should show, or '' when there is
 * nothing worth saying — the field is not a stalled rank, the data is empty, or
 * it is populated past the threshold and the slice is effectively the whole.
 */
export function coverageNote(rows: any[], field: string, { threshold = 0.9 } = {}): string {
  if (!SPARSE_RANKS.has(field)) return ''
  const { present, total, fraction } = fieldCoverage(rows, field)
  if (!total || fraction >= threshold) return ''
  const pct = fraction < 0.01 ? '<1' : String(Math.round(fraction * 100))
  return `Grouped over the ${present.toLocaleString()} of ${total.toLocaleString()} records `
    + `(${pct}%) that carry a ${field}. Taxonomy above species is only partly resolved, `
    + 'so this is a slice, not the whole dataset.'
}
