const ee = require('@google/earthengine');
const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');

// Helper to safely parse private key from environment variables
const getPrivateKey = () => {
  const rawKey = process.env.EE_PRIVATE_KEY;
  if (!rawKey) throw new Error('EE_PRIVATE_KEY environment variable is missing.');
  
  try {
    // Attempt parsing as JSON string
    const parsed = JSON.parse(rawKey);
    return parsed.private_key || parsed;
  } catch (e) {
    // Fallback: handle as formatted string with newline escape characters
    return rawKey.replace(/\\n/g, '\n');
  }
};

// Authenticate Earth Engine using GCP Service Account
const authenticateEE = () => {
  return new Promise((resolve, reject) => {
    const clientEmail = process.env.EE_CLIENT_EMAIL;
    const privateKey = getPrivateKey();

    if (!clientEmail) {
      return reject(new Error('EE_CLIENT_EMAIL environment variable is missing.'));
    }

    ee.data.authenticateViaPrivateKey(
      {
        client_email: clientEmail,
        private_key: privateKey,
      },
      () => ee.initialize(null, null, resolve, (err) => reject(new Error(`EE Init Error: ${err}`))),
      (err) => reject(new Error(`EE Auth Error: ${err}`))
    );
  });
};

exports.handler = async (event, context) => {
  // Allow GET for browser/curl testing, POST for webhooks/cron
  if (event.httpMethod !== 'GET' && event.httpMethod !== 'POST') {
    return {
      statusCode: 405,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ error: 'Method Not Allowed' }),
    };
  }

  try {
    // 1. Check Supabase Config
    if (!process.env.SUPABASE_URL || !process.env.SUPABASE_SERVICE_ROLE_KEY) {
      throw new Error('Supabase environment variables are missing.');
    }
    const supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_SERVICE_ROLE_KEY
    );

    // 2. Authenticate with Earth Engine
    await authenticateEE();

    // 3. Fetch recent observation points from iNaturalist API
    // (Defaulting to Boletus edulis taxon_id=48701; tweak as needed)
    const inatRes = await fetch(
      'https://api.inaturalist.org/v1/observations?taxon_id=48701&per_page=10&order=desc&order_by=created_at'
    );
    
    if (!inatRes.ok) {
      throw new Error(`iNaturalist API responded with status ${inatRes.status}`);
    }
    
    const inatData = await inatRes.json();

    const points = (inatData.results || [])
      .filter((obs) => obs.geojson && obs.geojson.coordinates)
      .map((obs) => {
        const [lon, lat] = obs.geojson.coordinates;
        return ee.Feature(ee.Geometry.Point([lon, lat]), {
          inat_id: obs.id,
          observed_on: obs.observed_on || new Date().toISOString().split('T')[0],
        });
      });

    if (points.length === 0) {
      return {
        statusCode: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: 'No valid observation points found to process.' }),
      };
    }

    const featureCollection = ee.FeatureCollection(points);

    // 4. Earth Engine Raster Pipeline
    // Using single-image 3DEP 10m DEM asset
    const dem = ee.Image('USGS/3DEP/10m').select('elevation');
    const slopeRad = ee.Terrain.slope(dem).multiply(Math.PI / 180);
    const flowAcc = ee.Terrain.slope(dem).convolve(ee.Kernel.gaussian(30)).add(1);
    const twi = flowAcc.divide(slopeRad.tan().add(0.001)).log().rename('twi');

    const nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')
      .filter(ee.Filter.eq('system:index', '2021'))
      .first()
      .select('landcover')
      .rename('landcover');

    // Combine single image and collection bands into a multi-band sampling composite
    const composite = dem.addBands(twi).addBands(nlcd);

    // 5. Sample spatial rasters at point locations
    const sampledPoints = composite.reduceRegions({
      collection: featureCollection,
      reducer: ee.Reducer.first(),
      scale: 10,
    });

    // Evaluate GEE execution into JS Object asynchronously
    const enrichedResults = await new Promise((resolve, reject) => {
      sampledPoints.evaluate((data, err) => {
        if (err) reject(new Error(`GEE Evaluation Error: ${err}`));
        else resolve(data);
      });
    });

    // 6. Format and Upsert into Supabase PostGIS
    const records = enrichedResults.features.map((f) => ({
      inat_id: f.properties.inat_id,
      observed_on: f.properties.observed_on,
      elevation_m: f.properties.elevation || null,
      twi: f.properties.twi || null,
      nlcd_class: f.properties.landcover || null,
      location: `POINT(${f.geometry.coordinates[0]} ${f.geometry.coordinates[1]})`,
    }));

    const { data, error } = await supabase
      .from('observations')
      .upsert(records, { onConflict: 'inat_id' });

    if (error) {
      throw new Error(`Supabase Upsert Error: ${error.message}`);
    }

    return {
      statusCode: 200,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: 'success',
        processed_count: records.length,
        records,
      }),
    };
  } catch (err) {
    console.error('Pipeline Runtime Failure:', err.message);
    return {
      statusCode: 500,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: 'error',
        error: err.message || 'Internal Server Error',
      }),
    };
  }
};