import { defineEventHandler, readMultipartFormData, getHeader, createError } from 'h3'
import { parse as parseCsv } from 'csv-parse/sync'
import { strFromU8 } from 'fflate'
import { randomUUID } from 'node:crypto'
import { requireMemberFresh } from '~/netlify/lib/auth.mjs'
import {
  GbifUploadError, MAX_RECORDS, MAX_UPLOAD_BYTES, entryMatches, safeUnzip,
} from '~/netlify/lib/gbif-upload.mjs'
import { MAX_DATASETS_PER_MEMBER } from '~/netlify/lib/species-fetch.mjs'
import { uploadJson } from '~/netlify/lib/datasets-store.mjs'
import { serviceClient } from '~/netlify/lib/supabase-storage.mjs'
import { DEFAULT_VISIBILITY, slugify, nextFreeSlug } from '~/netlify/lib/dataset-access.mjs'
import { cleanProvenance } from '~/netlify/lib/dataset-detail.mjs'

const FIELDS = 'id, owner_id, slug, title, description, path, visibility, feature_count, bytes, provenance, created_at, updated_at'

// Extra DwC fields to carry forward verbatim if present.
const EXTRA_FIELDS = [
  'basisOfRecord', 'establishmentMeans', 'lifeStage', 'sex', 'individualCount',
  'country', 'stateProvince', 'county', 'taxonRank',
  'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'infraspecificEpithet',
  'taxonKey', 'speciesKey', 'institutionCode', 'collectionCode', 'catalogNumber',
  'recordedBy', 'identifiedBy', 'dateIdentified',
  'habitat', 'samplingProtocol', 'elevation', 'depth',
]

// DwC-A extension row-type URIs we handle.
const MULTIMEDIA_ROW_TYPE = 'http://rs.gbif.org/terms/1.0/Multimedia'
const DNA_ROW_TYPES = [
  'http://rs.gbif.org/terms/1.0/DNADerivedData',
  'http://data.ggbn.org/schemas/ggbn/terms/Amplification',
]

interface ExtensionMeta { file: string; rowType: string; delimiter: string }
interface ArchiveMeta { coreFile: string; coreDelimiter: string; extensions: ExtensionMeta[] }

function parseMetaXml(xml: string): ArchiveMeta {
  const coreBlock = xml.match(/<core[^>]*>([\s\S]*?)<\/core>/i)?.[1] ?? ''
  const coreFile = coreBlock.match(/<location>(.*?)<\/location>/i)?.[1]?.trim() ?? 'occurrence.txt'
  const coreDelimiter = (xml.match(/<core[^>]*fieldsTerminatedBy="([^"]+)"/)?.[1] ?? '\\t').replace('\\t', '\t')

  const extensions: ExtensionMeta[] = []
  for (const m of xml.matchAll(/<extension\s([^>]*)rowType="([^"]+)"([^>]*)>([\s\S]*?)<\/extension>/gi)) {
    const rowType = m[2]
    const block = m[4]
    const file = block.match(/<location>(.*?)<\/location>/i)?.[1]?.trim() ?? ''
    const fullTag = m[0]
    const delimiter = (fullTag.match(/fieldsTerminatedBy="([^"]+)"/)?.[1] ?? '\\t').replace('\\t', '\t')
    if (file) extensions.push({ file, rowType, delimiter })
  }
  return { coreFile, coreDelimiter, extensions }
}

function parseDsv(text: string, delimiter: string): Record<string, string>[] {
  return parseCsv(text, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
    delimiter,
    relax_quotes: true,
    relax_column_count: true,
  }) as Record<string, string>[]
}

// Find a file inside a DwC-A zip by its declared name, trying root and one-level subdirs.
function findEntry(files: Record<string, Uint8Array>, name: string): string | undefined {
  return Object.keys(files).find(k => entryMatches(k, name))
}

function buildImagesMap(rows: Record<string, string>[]): Map<string, object[]> {
  const map = new Map<string, object[]>()
  for (const row of rows) {
    const id = row.gbifID ?? row.coreid ?? row.id
    const url = row.identifier
    if (!id || !url) continue
    const img: Record<string, string> = { url }
    if (row.type) img.type = row.type
    if (row.format) img.format = row.format
    if (row.license) img.license = row.license
    if (row.creator) img.creator = row.creator
    if (row.rightsHolder) img.rightsHolder = row.rightsHolder
    if (row.title) img.title = row.title
    if (row.references) img.references = row.references
    if (!map.has(id)) map.set(id, [])
    map.get(id)!.push(img)
  }
  return map
}

function buildDnaMap(rows: Record<string, string>[]): Map<string, object[]> {
  const DNA_COPY = [
    // DNA derived data
    'target_gene', 'DNA_sequence', 'materialSampleID', 'pcr_primer_name_forward',
    'pcr_primer_name_reverse', 'env_medium', 'lib_layout', 'sop',
    // GGBN amplification
    'marker', 'markerSubfragment', 'consensusSequence', 'amplificationSuccess',
    'boldProcessID', 'genBankAccession',
  ]
  const map = new Map<string, object[]>()
  for (const row of rows) {
    const id = row.gbifID ?? row.coreid ?? row.id
    if (!id) continue
    const entry: Record<string, string> = {}
    for (const f of DNA_COPY) {
      if (row[f]) entry[f] = row[f]
    }
    if (!Object.keys(entry).length) continue
    if (!map.has(id)) map.set(id, [])
    map.get(id)!.push(entry)
  }
  return map
}

function rowToFeature(
  row: Record<string, string>,
  images: object[],
  dna: object[],
): object | null {
  const lat = parseFloat(row.decimalLatitude ?? row.lat ?? '')
  const lon = parseFloat(row.decimalLongitude ?? row.lon ?? '')
  if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) return null

  const props: Record<string, unknown> = {
    species: row.species || row.scientificName || 'Unknown',
    date: row.eventDate || (row.year ? `${row.year}` : null),
    gbifId: row.gbifID || row.id || null,
    occurrenceStatus: row.occurrenceStatus || 'PRESENT',
    coordinateUncertainty: row.coordinateUncertaintyInMeters
      ? parseFloat(row.coordinateUncertaintyInMeters) : null,
  }

  for (const f of EXTRA_FIELDS) {
    if (row[f]) props[f] = row[f]
  }

  if (images.length) props.images = images
  if (dna.length) props.dna = dna

  return { type: 'Feature', geometry: { type: 'Point', coordinates: [lon, lat] }, properties: props }
}

export default defineEventHandler(async (event) => {
  try {
    // Signed in and a member before anything is read: parsing an archive is the
    // expensive part, and saving a dataset is a membership benefit, as it is
    // for the CSV/GeoJSON upload in the datasets function.
    const authHeader = getHeader(event, 'authorization') || ''
    const auth: any = await requireMemberFresh({ headers: new Headers({ authorization: authHeader }) } as any)
    if (!auth.ok) {
      const body = await auth.response.json().catch(() => ({}))
      throw new GbifUploadError(body.error || 'Sign in required.', auth.response.status)
    }

    const declared = Number(getHeader(event, 'content-length') || 0)
    if (declared > MAX_UPLOAD_BYTES) {
      throw new GbifUploadError(`That file is over ${MAX_UPLOAD_BYTES / 1048576} MB. Narrow the GBIF download, or use "Import observations" to fetch from GBIF directly.`, 413)
    }

    const formData = await readMultipartFormData(event)
    if (!formData) throw new GbifUploadError('No file uploaded')

    const filePart = formData.find(p =>
      p.name === 'file' && (p.filename?.toLowerCase().endsWith('.zip') || p.filename?.toLowerCase().endsWith('.csv')))
    if (!filePart) {
      throw new GbifUploadError('Please upload a GBIF Darwin Core Archive (.zip) or simple CSV export')
    }
    if (filePart.data.length > MAX_UPLOAD_BYTES) {
      throw new GbifUploadError(`That file is over ${MAX_UPLOAD_BYTES / 1048576} MB. Narrow the GBIF download.`, 413)
    }

    const isDwca = filePart.filename!.toLowerCase().endsWith('.zip')

    let occurrenceRows: Record<string, string>[] = []
    let imagesMap = new Map<string, object[]>()
    let dnaMap = new Map<string, object[]>()

    if (isDwca) {
      // Two passes, each extracting only what it needs: meta.xml names the
      // files, then just those are inflated, within the size limits.
      const metaFiles = safeUnzip(filePart.data, n => entryMatches(n, 'meta.xml'))
      const metaEntry = findEntry(metaFiles, 'meta.xml')
      const meta = metaEntry ? parseMetaXml(strFromU8(metaFiles[metaEntry])) : null

      const coreFile = meta?.coreFile ?? 'occurrence.txt'
      const coreDelim = meta?.coreDelimiter ?? '\t'
      const mmExt = meta?.extensions.find(e => e.rowType === MULTIMEDIA_ROW_TYPE)
      const mmFile = mmExt?.file ?? 'multimedia.txt'
      const dnaExt = meta?.extensions.find(e => DNA_ROW_TYPES.includes(e.rowType))
      const names = [coreFile, 'occurrence.csv', 'occurrence.txt', mmFile, 'multimedia.csv', 'multimedia.txt']
      if (dnaExt?.file) names.push(dnaExt.file)
      const files = safeUnzip(filePart.data, n => names.some(name => entryMatches(n, name)))

      const coreEntry = findEntry(files, coreFile)
        ?? findEntry(files, 'occurrence.csv')
        ?? findEntry(files, 'occurrence.txt')
      if (!coreEntry) throw new GbifUploadError('No occurrence file found in the Darwin Core Archive')
      occurrenceRows = parseDsv(strFromU8(files[coreEntry]), coreDelim)

      const mmEntry = findEntry(files, mmFile)
        ?? findEntry(files, 'multimedia.csv')
        ?? findEntry(files, 'multimedia.txt')
      if (mmEntry) {
        imagesMap = buildImagesMap(parseDsv(strFromU8(files[mmEntry]), mmExt?.delimiter ?? '\t'))
      }

      if (dnaExt?.file) {
        const dnaEntry = findEntry(files, dnaExt.file)
        if (dnaEntry) {
          dnaMap = buildDnaMap(parseDsv(strFromU8(files[dnaEntry]), dnaExt.delimiter))
        }
      }
    } else {
      // Plain GBIF simple CSV. GBIF's "simple" export is tab-separated despite
      // the name; take whichever delimiter the header line uses.
      const text = filePart.data.toString('utf-8')
      const header = text.slice(0, text.indexOf('\n') >>> 0)
      occurrenceRows = parseDsv(text, header.includes('\t') ? '\t' : ',')
    }

    const features: object[] = []
    let skippedCount = 0
    let mediaCount = 0
    let dnaCount = 0
    let truncated = 0

    for (const row of occurrenceRows) {
      if (features.length >= MAX_RECORDS) { truncated++; continue }
      const id = row.gbifID ?? row.id ?? ''
      const images = imagesMap.get(id) ?? []
      const dna = dnaMap.get(id) ?? []
      const feature = rowToFeature(row, images, dna)
      if (!feature) { skippedCount++; continue }
      mediaCount += images.length
      dnaCount += dna.length
      features.push(feature)
    }

    if (features.length === 0) throw new GbifUploadError('No valid geographic records found in the file')

    const validCount = features.length
    const datasetId = `gbif_${randomUUID()}`
    const geojson = { type: 'FeatureCollection', features }
    const previewGeojson = { type: 'FeatureCollection', features: features.slice(0, 100) }

    let savedDataset: any = null
    const user = auth.user
    const client = serviceClient()

    if (user && client) {
      const { count } = await client.from('saved_datasets')
        .select('id', { count: 'exact', head: true }).eq('owner_id', user.id)
      if ((count || 0) >= MAX_DATASETS_PER_MEMBER) {
        throw new GbifUploadError(`You have ${count} saved datasets, which is the limit. Delete one to save another.`, 409)
      }

      const titlePart = formData.find(p => p.name === 'title')
      const rawTitle = titlePart?.data.toString('utf-8').trim() || ''
      const title = (rawTitle || `GBIF Import (${validCount} records)`).slice(0, 200)
      const description = isDwca
        ? `DwC-A import — ${validCount} occurrences, ${mediaCount} media, ${dnaCount} DNA records, ${skippedCount} skipped.`
        : `CSV import — ${validCount} occurrence records, ${skippedCount} skipped.`

      const base = slugify(title)
      const { data: clashes } = await client.from('saved_datasets')
        .select('slug').like('slug', `${base}%`)
      const slug = nextFreeSlug(base, (clashes || []).map((r: any) => r.slug))
      const path = `datasets/${user.id}/${slug}-${Date.now()}.geojson`
      const body = await uploadJson(path, geojson)
      const bytes = body.length

      const { data, error } = await client.from('saved_datasets').insert({
        owner_id: user.id, job_id: null, slug, title, description, path,
        visibility: DEFAULT_VISIBILITY, feature_count: validCount, bytes,
        provenance: cleanProvenance({
          origin: 'gbif_archive', file_name: filePart.filename, format: isDwca ? 'dwca' : 'csv',
          records: validCount, skipped: skippedCount, media: mediaCount, dna: dnaCount,
        }),
      }).select(FIELDS).single()
      if (!error) savedDataset = data
    }

    const capped = truncated ? ` Only the first ${MAX_RECORDS.toLocaleString()} were kept.` : ''
    return {
      success: true,
      datasetId,
      dataset: savedDataset,
      assetPath: null,
      stats: { totalRecords: occurrenceRows.length, validRecords: validCount, skippedRecords: skippedCount, mediaRecords: mediaCount, dnaRecords: dnaCount, truncatedRecords: truncated },
      message: (savedDataset
        ? `Saved ${validCount} records as "${savedDataset.title}".`
        : `Processed ${validCount} occurrence records.`) + capped,
      preview: previewGeojson,
    }

  } catch (error: any) {
    if (error instanceof GbifUploadError) {
      throw createError({ statusCode: error.status, statusMessage: error.message })
    }
    console.error('GBIF import error:', error)
    throw createError({ statusCode: 400, statusMessage: error?.message || 'Failed to process file' })
  }
})
