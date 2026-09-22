import { defineEventHandler, readMultipartFormData, getHeader, createError } from 'h3'
import { parse } from 'csv-parse/sync'
import { v4 as uuidv4 } from 'uuid'
import { verifyToken } from '~/netlify/lib/auth.mjs'
import { uploadJson } from '~/netlify/lib/datasets-store.mjs'
import { serviceClient } from '~/netlify/lib/supabase-storage.mjs'
import { DEFAULT_VISIBILITY, slugify, nextFreeSlug } from '~/netlify/lib/dataset-access.mjs'

const FIELDS = 'id, owner_id, slug, title, description, path, visibility, feature_count, bytes, created_at, updated_at'

export default defineEventHandler(async (event) => {
  try {
    const formData = await readMultipartFormData(event)
    if (!formData) {
      throw new Error('No file uploaded')
    }

    const csvFile = formData.find(part => part.name === 'file' && part.filename?.endsWith('.csv'))
    if (!csvFile) {
      throw new Error('Please upload a valid GBIF CSV export file')
    }

    const csvContent = csvFile.data.toString('utf-8')
    const records = parse(csvContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true
    })

    const features: any[] = []
    let skippedCount = 0

    for (const row of records) {
      const lat = parseFloat(row.decimalLatitude || row.lat || '')
      const lon = parseFloat(row.decimalLongitude || row.lon || '')

      if (isNaN(lat) || isNaN(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        skippedCount++
        continue
      }

      const feature: any = {
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [lon, lat] },
        properties: {
          species: row.species || row.scientificName || 'Unknown',
          date: row.eventDate || (row.year ? `${row.year}` : null),
          gbifId: row.gbifID || null,
          occurrenceStatus: row.occurrenceStatus || 'PRESENT',
          coordinateUncertainty: row.coordinateUncertaintyInMeters
            ? parseFloat(row.coordinateUncertaintyInMeters) : null,
          originalData: row
        }
      }

      const commonFields = ['basisOfRecord', 'establishmentMeans', 'lifeStage', 'sex',
        'individualCount', 'country', 'stateProvince', 'county']
      for (const field of commonFields) {
        if (row[field]) feature.properties[field] = row[field]
      }

      features.push(feature)
    }

    if (features.length === 0) {
      throw new Error('No valid geographic records found in the CSV file')
    }

    const validCount = features.length
    const datasetId = `gbif_${uuidv4()}`
    const geojson = { type: 'FeatureCollection', features }
    const previewGeojson = { type: 'FeatureCollection', features: features.slice(0, 100) }

    // Save as a proper dataset when the user is logged in
    let savedDataset: any = null
    const authHeader = getHeader(event, 'authorization') || ''
    const tokenMatch = /^Bearer\s+(.+)$/i.exec(authHeader.trim())
    const token = tokenMatch ? tokenMatch[1].trim() : null
    const user = token ? await verifyToken(token) : null
    const client = serviceClient()

    if (user && client) {
      const titlePart = formData.find(p => p.name === 'title')
      const rawTitle = titlePart?.data.toString('utf-8').trim() || ''
      const title = (rawTitle || `GBIF Import (${validCount} records)`).slice(0, 200)
      const description = `Imported from GBIF CSV — ${validCount} occurrence records, ${skippedCount} skipped.`

      const base = slugify(title)
      const { data: clashes } = await client.from('saved_datasets')
        .select('slug').like('slug', `${base}%`)
      const slug = nextFreeSlug(base, (clashes || []).map((r: any) => r.slug))

      const path = `datasets/${user.id}/${slug}-${Date.now()}.geojson`
      await uploadJson(path, geojson)

      const bytes = JSON.stringify(geojson).length
      const { data, error } = await client.from('saved_datasets').insert({
        owner_id: user.id,
        job_id: null,
        slug,
        title,
        description,
        path,
        visibility: DEFAULT_VISIBILITY,
        feature_count: validCount,
        bytes,
      }).select(FIELDS).single()

      if (!error) savedDataset = data
    }

    return {
      success: true,
      datasetId,
      dataset: savedDataset,
      assetPath: savedDataset ? null : `users/your-ee-project/gbif_imports/${datasetId}`,
      stats: {
        totalRecords: records.length,
        validRecords: validCount,
        skippedRecords: skippedCount
      },
      message: savedDataset
        ? `Saved ${validCount} occurrence records as dataset "${savedDataset.title}".`
        : `Successfully processed ${validCount} occurrence records. To complete the import, upload the generated GeoJSON to your Earth Engine assets folder.`,
      preview: previewGeojson
    }

  } catch (error: any) {
    console.error('GBIF import error:', error)
    throw createError({
      statusCode: 400,
      statusMessage: error.message || 'Failed to process GBIF export'
    })
  }
})
