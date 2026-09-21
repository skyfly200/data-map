import { defineEventHandler, readMultipartFormData } from 'h3'
import { parse } from 'csv-parse/sync'
import { v4 as uuidv4 } from 'uuid'

export default defineEventHandler(async (event) => {
  try {
    const formData = await readMultipartFormData(event)
    if (!formData) {
      throw new Error('No file uploaded')
    }

    // Find the CSV file
    const csvFile = formData.find(part => part.name === 'file' && part.filename?.endsWith('.csv'))
    if (!csvFile) {
      throw new Error('Please upload a valid GBIF CSV export file')
    }

    // Parse CSV content
    const csvContent = csvFile.data.toString('utf-8')
    const records = parse(csvContent, {
      columns: true,
      skip_empty_lines: true,
      trim: true
    })

    // Validate and transform GBIF data to GeoJSON
    const features = []
    let validCount = 0
    let skippedCount = 0

    for (const row of records) {
      // GBIF standard fields
      const lat = parseFloat(row.decimalLatitude || row.lat || '')
      const lon = parseFloat(row.decimalLongitude || row.lon || '')

      if (isNaN(lat) || isNaN(lon)) {
        skippedCount++
        continue
      }

      // Validate coordinates
      if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
        skippedCount++
        continue
      }

      const feature: any = {
        type: 'Feature',
        geometry: {
          type: 'Point',
          coordinates: [lon, lat]
        },
        properties: {
          species: row.species || row.scientificName || 'Unknown',
          date: row.eventDate || row.year ? `${row.year}` : null,
          gbifId: row.gbifID || null,
          occurrenceStatus: row.occurrenceStatus || 'PRESENT',
          coordinateUncertainty: row.coordinateUncertaintyInMeters ? parseFloat(row.coordinateUncertaintyInMeters) : null,
          originalData: row // Keep all original fields
        }
      }

      // Add common GBIF fields if present
      const commonFields = ['basisOfRecord', 'establishmentMeans', 'lifeStage', 'sex', 'individualCount', 'country', 'stateProvince', 'county']
      commonFields.forEach(field => {
        if (row[field]) {
          feature.properties[field] = row[field]
        }
      })

      features.push(feature)
      validCount++
    }

    if (features.length === 0) {
      throw new Error('No valid geographic records found in the CSV file')
    }

    const geojson = {
      type: 'FeatureCollection',
      features: features
    }

    // Generate unique dataset ID
    const datasetId = `gbif_${uuidv4()}`

    // In a real implementation, this would:
    // 1. Upload to Google Cloud Storage bucket linked to Earth Engine
    // 2. Trigger Earth Engine asset ingestion
    // 3. Return the asset path

    // For now, we'll store the GeoJSON in Supabase and provide instructions
    // for manual upload to EE, or simulate the asset path

    const mockAssetPath = `users/your-ee-project/gbif_imports/${datasetId}`

    return {
      success: true,
      datasetId,
      assetPath: mockAssetPath,
      stats: {
        totalRecords: records.length,
        validRecords: validCount,
        skippedRecords: skippedCount
      },
      message: `Successfully processed ${validCount} occurrence records. To complete the import, upload the generated GeoJSON to your Earth Engine assets folder.`,
      preview: geojson
    }

  } catch (error: any) {
    console.error('GBIF import error:', error)
    throw createError({
      statusCode: 400,
      statusMessage: error.message || 'Failed to process GBIF export'
    })
  }
})
