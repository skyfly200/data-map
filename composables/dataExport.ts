// Turning a set of observations into a file somebody can open somewhere else.
//
// Pure, and separate from the component that offers the button, because the
// parts worth being careful about are all in the encoding:
//
//   CSV has no specification anybody follows, but it does have one rule that
//   matters — a field containing a comma, a quote or a newline must be quoted,
//   and a quote inside it doubled. Getting that wrong shifts every column after
//   it, silently, in one row out of a thousand. A species note with a comma in
//   it is enough.
//
//   A spreadsheet treats a cell beginning =, +, - or @ as a formula. That makes
//   any text field a way to run something in the reader's spreadsheet, and the
//   text comes from iNaturalist, which is to say from the public. See quoteCsv.
//
//   GeoJSON has to stay valid when the caller narrows the columns. Dropping the
//   geometry to save width leaves something that is no longer GeoJSON, so the
//   column selection applies to properties only.

export interface ObservationFeature {
  type: 'Feature'
  geometry: {
    type: 'Point'
    coordinates: [number, number]
  }
  properties: Record<string, any>
}

/** Columns every row has, in the order a reader expects to meet them. */
export const IDENTITY_COLUMNS = ['id', 'date', 'scientific_name', 'common_name']

/** Where the coordinates go in a CSV, which has nowhere else to put them. */
export const GEOMETRY_COLUMNS = ['longitude', 'latitude']

/**
 * Every property key present across the features, identity columns first.
 *
 * Union rather than the first row's keys: an enriched set is ragged — a job
// that ran four stages over points another job had already enriched leaves rows
// with different columns, and a reader should get all of them.
 */
export function columnsOf(features: ObservationFeature[] = []): string[] {
  const seen = new Set<string>()
  for (const f of features) {
    for (const key of Object.keys(f?.properties || {})) seen.add(key)
  }
  const ordered = IDENTITY_COLUMNS.filter((c) => seen.has(c))
  for (const c of ordered) seen.delete(c)
  return [...ordered, ...[...seen].sort()]
}

/**
 * One CSV field.
 *
 * The leading apostrophe on a formula-looking value is the conventional defence
// and it is deliberately inside the quoting, so a reader that does not evaluate
// formulas shows the apostrophe and that the text. Stripping the
// character instead would quietly alter data — "-5" is a number somebody meant.
 */
export function quoteCsv(value: any): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'number') return Number.isFinite(value) ? String(value) : ''
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  if (typeof value === 'object') {
    return quoteCsv(JSON.stringify(value))
  }

  let text = String(value)
  if (/^[=+\-@\t\r]/.test(text)) text = `'${text}`
  if (/[",\n\r]/.test(text)) return `"${text.replace(/"/g, '""')}"`
  return text
}

/**
 * A CSV of the given features.
 *
 * Longitude and latitude are written as their own columns, because a CSV has
// nowhere else to carry a geometry and a reader opening this in a spreadsheet
// is looking for two numbers.
 *
 * CRLF line endings: that is what RFC 4180 says, and it is what stops Excel on
// Windows from running the whole file into one row.
 */
export function toCsv(features: ObservationFeature[] = [], { columns = null, geometry = true } = {}): string {
  const cols = columns || columnsOf(features)
  const header = geometry ? [...GEOMETRY_COLUMNS, ...cols] : [...cols]
  const lines = [header.map(quoteCsv).join(',')]

  for (const f of features) {
    const props = f?.properties || {}
    const co = f?.geometry?.coordinates || []
    const row = geometry ? [co[0] ?? '', co[1] ?? ''] : []
    for (const c of cols) row.push(props[c])
    lines.push(row.map(quoteCsv).join(','))
  }
  return `${lines.join('\r\n')}\r\n`
}

/**
 * A GeoJSON FeatureCollection, optionally with the properties narrowed.
 *
 * The geometry is never dropped. Narrowing columns is about width, and a
// feature without a geometry is not a feature.
 */
export function toGeoJson(features: ObservationFeature[] = [], { columns = null, pretty = false } = {}): string {
  const out = columns
    ? features.map((f) => {
      const props = f?.properties || {}
      const narrowed = {}
      for (const c of columns) {
        if (Object.prototype.hasOwnProperty.call(props, c)) narrowed[c] = props[c]
      }
      return { ...f, properties: narrowed }
    })
    : features

  const collection = { type: 'FeatureCollection', features: out }
  return JSON.stringify(collection, null, pretty ? 2 : 0)
}

/** What to call the file. */
export function exportFilename(stem: string, format: 'csv' | 'geojson', { stamp = new Date() } = {}): string {
  const date = stamp instanceof Date ? stamp.toISOString().slice(0, 10) : String(stamp)
  const slug = String(stem || 'export').toLowerCase()
    .replace(/[^\w\s-]/g, '').trim().replace(/\s+/g, '-')
    .slice(0, 60) || 'export'
  return `${slug}-${date}.${format === 'csv' ? 'csv' : 'geojson'}`
}

/** The blob for a format, ready to hand to a download. */
export function exportBlob(features: ObservationFeature[], format: 'csv' | 'geojson', options = {}): Blob {
  if (format === 'csv') {
    return new Blob(['﻿', toCsv(features, options)], { type: 'text/csv;charset=utf-8' })
  }
  return new Blob([toGeoJson(features, options)], { type: 'application/geo+json' })
}
