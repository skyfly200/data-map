// Turning a member's own CSV or GeoJSON file into a point FeatureCollection.
//
// Pure functions, no I/O: the datasets function validates with these, and the
// tests exercise them directly. Accepted: CSV with lat/lon columns (iNat, GBIF
// and generic headers are auto-detected, overridable), and GeoJSON
// (FeatureCollection, a single Feature, or a bare geometry). Output features
// are Points with `species`, `date` and a few carried-through properties.

export const MAX_UPLOAD_BYTES = 3 * 1024 * 1024

const LAT = ['latitude', 'lat', 'decimallatitude', 'decimal_latitude', 'y']
const LON = ['longitude', 'lon', 'lng', 'decimallongitude', 'decimal_longitude', 'x']
const DATE = ['observed_on', 'date', 'eventdate', 'event_date', 'observedon', 'dateidentified', 'date_observed']
const SPECIES = ['scientific_name', 'scientificname', 'taxon_name', 'species', 'taxon', 'name', 'taxon_species_name', 'verbatimscientificname']

export class UploadError extends Error {
  constructor(message, code = 'bad_upload') { super(message); this.code = code }
}

const detect = (headers, candidates) => candidates.find((c) => headers.includes(c)) || null

/** Parse CSV text into { headers, rows }. Handles quoted fields and quoted newlines. */
export function parseCsv(text) {
  const src = String(text).replace(/^﻿/, '')
  const records = []
  let cells = [], cur = '', inQuote = false
  for (let i = 0; i < src.length; i++) {
    const c = src[i]
    if (inQuote) {
      if (c === '"' && src[i + 1] === '"') { cur += '"'; i++ }
      else if (c === '"') inQuote = false
      else cur += c
    } else if (c === '"') inQuote = true
    else if (c === ',') { cells.push(cur); cur = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && src[i + 1] === '\n') i++
      cells.push(cur); cur = ''
      records.push(cells); cells = []
    } else cur += c
  }
  if (cur || cells.length) { cells.push(cur); records.push(cells) }
  const nonEmpty = records.filter((r) => r.some((v) => v.trim()))
  if (nonEmpty.length < 2) return { headers: [], rows: [] }
  const headers = nonEmpty[0].map((h) => h.trim().toLowerCase())
  const rows = nonEmpty.slice(1).map((cs) => {
    const row = {}
    headers.forEach((h, j) => { row[h] = (cs[j] || '').trim() })
    return row
  })
  return { headers, rows }
}

const validCoord = (lat, lon) => Number.isFinite(lat) && Number.isFinite(lon)
  && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180

function makeFeature(lon, lat, { date, species, extra = {} } = {}) {
  const props = { ...extra }
  if (date) props.date = String(date).slice(0, 10)
  if (species) props.species = String(species)
  return { type: 'Feature', geometry: { type: 'Point', coordinates: [lon, lat] }, properties: props }
}

/** CSV text -> { features, skipped, detectedCols }. Throws UploadError. */
export function csvToFeatures(text, cols = {}) {
  const { headers, rows } = parseCsv(text)
  if (!headers.length) throw new UploadError('Could not parse CSV: need a header row and at least one data row.', 'bad_csv')
  const pick = (given, cands) => String(given || detect(headers, cands) || '').toLowerCase()
  const latCol = pick(cols.lat_col, LAT)
  const lonCol = pick(cols.lon_col, LON)
  if (!latCol || !headers.includes(latCol)) {
    throw new UploadError(`Could not find a latitude column. Detected headers: ${headers.slice(0, 10).join(', ')}. Pass lat_col to specify one.`, 'no_lat_col')
  }
  if (!lonCol || !headers.includes(lonCol)) {
    throw new UploadError('Could not find a longitude column. Pass lon_col to specify one.', 'no_lon_col')
  }
  const dateCol = pick(cols.date_col, DATE)
  const speciesCol = pick(cols.species_col, SPECIES)

  const features = []
  let skipped = 0
  for (const row of rows) {
    const lat = parseFloat(row[latCol]), lon = parseFloat(row[lonCol])
    if (!validCoord(lat, lon)) { skipped++; continue }
    const extra = {}
    if (row.quality_grade) extra.quality_grade = row.quality_grade
    if (row.id) extra.source_id = row.id
    features.push(makeFeature(lon, lat, {
      date: dateCol && row[dateCol], species: speciesCol && row[speciesCol], extra,
    }))
  }
  return { features, skipped, detectedCols: { latCol, lonCol, dateCol: dateCol || null, speciesCol: speciesCol || null } }
}

const firstOf = (props, keys) => {
  const lower = {}
  for (const k of Object.keys(props || {})) lower[k.toLowerCase()] = props[k]
  for (const k of keys) if (lower[k] != null && lower[k] !== '') return lower[k]
  return null
}

/** GeoJSON (object or text) -> { features, skipped }. Non-point geometries use their first vertex. */
export function geojsonToFeatures(input) {
  let gj = input
  if (typeof input === 'string') {
    try { gj = JSON.parse(input) } catch { throw new UploadError('That file is not valid JSON.', 'bad_geojson') }
  }
  let items
  if (gj?.type === 'FeatureCollection' && Array.isArray(gj.features)) items = gj.features
  else if (gj?.type === 'Feature') items = [gj]
  else if (gj?.type && gj.coordinates) items = [{ type: 'Feature', geometry: gj, properties: {} }]
  else throw new UploadError('Expected a GeoJSON FeatureCollection.', 'bad_geojson')

  const features = []
  let skipped = 0
  for (const f of items) {
    let c = f?.geometry?.coordinates
    while (Array.isArray(c) && Array.isArray(c[0])) c = c[0]
    const lon = Array.isArray(c) ? Number(c[0]) : NaN
    const lat = Array.isArray(c) ? Number(c[1]) : NaN
    if (!validCoord(lat, lon)) { skipped++; continue }
    const p = f.properties || {}
    // Source properties are kept; normalized date/species keys win.
    features.push(makeFeature(lon, lat, {
      date: firstOf(p, DATE), species: firstOf(p, SPECIES), extra: p,
    }))
  }
  return { features, skipped }
}

/** Dispatch on format ('csv' | 'geojson'). Throws UploadError if nothing usable. */
export function toFeatureCollection({ format, content, cols = {} }) {
  const text = typeof content === 'string' ? content : JSON.stringify(content ?? '')
  if (!text.trim() || text === '""') throw new UploadError('The file is empty.', 'empty')
  if (text.length > MAX_UPLOAD_BYTES) {
    throw new UploadError('File is larger than 3 MB. Thin it to the most relevant records — MaxEnt works well with 100–2,000 presence points.', 'too_large')
  }
  let res
  if (format === 'csv') res = csvToFeatures(text, cols)
  else if (format === 'geojson') res = geojsonToFeatures(content)
  else throw new UploadError('Format must be "csv" or "geojson".', 'bad_format')
  if (!res.features.length) {
    throw new UploadError(`No valid coordinate pairs found${res.skipped ? ` (${res.skipped} rows skipped for bad or missing coordinates)` : ''}.`, 'no_features')
  }
  return { ...res, geojson: { type: 'FeatureCollection', features: res.features } }
}

/** Guess a format from a file name, falling back to the content. */
export function guessFormat(name = '', text = '') {
  if (/\.(geo)?json$/i.test(name)) return 'geojson'
  if (/\.csv$/i.test(name)) return 'csv'
  return text.trimStart().startsWith('{') ? 'geojson' : 'csv'
}
