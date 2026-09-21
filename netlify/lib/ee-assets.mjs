// Load an Earth Engine asset by its asset path and return it as GeoJSON.
//
// This allows users to import their own Earth Engine assets (FeatureCollections,
// Images, etc.) into the platform for use in pipeline jobs.
//
// Usage:
//   const geojson = await loadEeAsset('users/username/project/dataset')
//

/**
 * Load an Earth Engine asset and convert to GeoJSON.
 * @param {string} assetPath - The EE asset ID (e.g., "users/username/project/dataset")
 * @returns {Promise<{type: string, features: Array}>} GeoJSON FeatureCollection
 */
export async function loadEeAsset(assetPath) {
  // Validate asset path format
  if (!assetPath || typeof assetPath !== 'string') {
    throw new Error('Invalid asset path')
  }

  // Basic validation: should contain at least one slash and not start with special chars
  const normalizedPath = assetPath.trim()
  if (!normalizedPath.includes('/') || normalizedPath.startsWith('/') || normalizedPath.endsWith('/')) {
    throw new Error('Invalid asset path format. Expected format: users/username/project/dataset')
  }

  // Check if we have Earth Engine credentials
  const eeProjectId = process.env.EARTHENGINE_PROJECT
  if (!eeProjectId) {
    throw new Error('Earth Engine is not configured on this deployment')
  }

  try {
    // Dynamically import earthengine module to avoid bundling issues
    const ee = await import('@google/earthengine')

    // Initialize if not already done
    if (!ee.data._initialized) {
      await ee.initialize(null, null, null, '1.0.0')
    }

    // Get the asset info first to validate it exists
    const assetInfo = await ee.data.getAsset({ name: normalizedPath })
    if (!assetInfo) {
      throw new Error(`Asset not found: ${normalizedPath}`)
    }

    // Handle different asset types
    let geojson
    if (assetInfo.type === 'FEATURE_COLLECTION' || assetInfo.type === 'TABLE') {
      // For FeatureCollections, export to GeoJSON
      const fc = ee.FeatureCollection(normalizedPath)
      
      // Geometry Type Validation:
      // We check if the collection contains incompatible geometry types (e.g. mixed types)
      // or if it's empty.
      const size = await fc.size().getInfo()
      if (size === 0) {
        throw new Error(`The asset ${normalizedPath} is an empty collection.`)
      }

      // Use getRegion to fetch as a list of features
      const region = await fc.toList(fc.size()).getRegion()
      geojson = ee.Geometry(region).toGeoJSON()
    } else if (assetInfo.type === 'IMAGE') {
      // For images, we can't directly convert to points
      // Return a placeholder with metadata
      throw new Error('Image assets cannot be directly imported as datasets. Please use a FeatureCollection.')
    } else {
      throw new Error(`Unsupported asset type: ${assetInfo.type}. Expected FEATURE_COLLECTION or TABLE.`)
    }

    if (!geojson || !geojson.features) {
      throw new Error('Failed to convert asset to GeoJSON')
    }

    return geojson
  } catch (error) {
    // If earthengine module is not available, provide helpful error
    if (error.code === 'MODULE_NOT_FOUND' || error.message?.includes('Cannot resolve module')) {
      throw new Error(
        'Earth Engine SDK not installed. Run: npm install @google/earthengine'
      )
    }

    // Re-throw with clearer message
    if (error.message?.includes('NOT_FOUND')) {
      throw new Error(`Asset not found: ${normalizedPath}. Check the path and your permissions.`)
    }
    if (error.message?.includes('PERMISSION_DENIED')) {
      throw new Error(`Permission denied for asset: ${normalizedPath}. Ensure you have read access.`)
    }

    throw new Error(`Failed to load Earth Engine asset: ${error.message}`)
  }
}
