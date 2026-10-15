// Checking a job's source while it is being picked, not after it is submitted.
//
// The server refuses a bad area or date range when the job is submitted, and an
// enrichment that gets past that now fails rather than hanging. Either way the
// member finds out two steps later than they could have. The shape checks are
// the server's own (normaliseSource), run in the browser, so the two cannot
// disagree; whether the area holds any observations needs the baseline and is
// asked of the server as a dry run.

import { SpecError, normaliseSource } from '../netlify/lib/ee-pipeline.mjs'

export interface BboxForm {
  north?: number | string | null
  south?: number | string | null
  east?: number | string | null
  west?: number | string | null
  dateFrom?: string | null
  dateTo?: string | null
  noDateFilter?: boolean
  taxon?: string | null
}

const blank = (v: unknown) => v === null || v === undefined || v === ''

/** The bbox source a form describes, in the shape a job spec carries. */
export function bboxSource(form: BboxForm) {
  return {
    type: 'bbox',
    bounds: { north: form.north, south: form.south, east: form.east, west: form.west },
    dateFrom: form.noDateFilter ? undefined : (form.dateFrom || undefined),
    dateTo: form.noDateFilter ? undefined : (form.dateTo || undefined),
    taxon: form.taxon || undefined,
  }
}

/**
 * What is wrong with a bbox form, or '' when nothing is.
 *
 * An emptied number input reads as '' and Number('') is 0, which the server
 * would take as an edge on the equator or the meridian, so a missing edge is
 * caught here first and named.
 */
export function bboxIssue(form: BboxForm): string {
  const missing = (['north', 'south', 'west', 'east'] as const).filter((k) => blank(form[k]))
  if (missing.length) return `Fill in the ${missing.join(', ')} edge${missing.length > 1 ? 's' : ''}.`
  try {
    normaliseSource(bboxSource(form))
    return ''
  } catch (err) {
    if (err instanceof SpecError) return err.message
    throw err
  }
}
