const ee = require('@google/earthengine');
const { createClient } = require('@supabase/supabase-js');
const fetch = require('node-fetch');

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY
);

// Authenticate Earth Engine using Service Account
const authenticateEE = () => {
  return new Promise((resolve, reject) => {
    const privateKey = JSON.parse(process.env.EE_PRIVATE_KEY);
    ee.data.authenticateViaPrivateKey(
      privateKey,
      () => ee.initialize(null, null, resolve, reject),
      (err) => reject(err)
    );
  });
};

exports.handler = async (event, context) => {
  try {
    await authenticateEE();

    // 1. Fetch recent target observations from iNaturalist API
    const inatRes = await fetch(
      'https://api.inaturalist.org/v1/observations?taxon_id=48701&per_page=10&order=desc&order_by=created_at'
    );
    const inatData = await inatRes.json();

    const points = inatData.results
      .filter((obs) => obs.geojson)
      .map((obs) => {
        const [lon, lat] = obs.geojson.coordinates;
        return ee.Feature(ee.Geometry.Point([lon, lat]), {
          inat_id: obs.id,
          observed_on: obs.observed_on,
        });
      });

    if (points.length === 0) {
      return {
        statusCode: 200,
        body: JSON.stringify({ message: 'No new points to process.' }),
      };
    }

    const featureCollection = ee.FeatureCollection(points);

    // 2. Define Earth Engine Raster Pipeline (using fixed single-image DEM)
    const dem = ee.Image('USGS/3DEP/10m').select('elevation');
    const slopeRad = ee.Terrain.slope(dem).multiply(Math.PI / 180);
    const flowAcc = ee.Terrain.slope(dem).convolve(ee.Kernel.gaussian(30)).add(1);
    const twi = flowAcc.divide(slopeRad.tan().add(0.001)).log().rename('twi');

    const nlcd = ee.ImageCollection('USGS/NLCD_RELEASES/2021_REL/NLCD')
      .filter(ee.Filter.eq('system:index', '2021'))
      .first()
      .select('landcover')
      .rename('landcover');

    // Combine rasters into a single multi-band image
    const composite = dem.addBands(twi).addBands(nlcd);

    // 3. Extract raster values at point locations
    const sampledPoints = composite.reduceRegions({
      collection: featureCollection,
      reducer: ee.Reducer.first(),
      scale: 10,
    });

    // Evaluate GEE computation into JS Object
    const enrichedResults = await new Promise((resolve, reject) => {
      sampledPoints.evaluate((data, err) => {
        if (err) reject(err);
        else resolve(data);
      });
    });

    // 4. Format & Upsert into Supabase
    const records = enrichedResults.features.map((f) => ({
      inat_id: f.properties.inat_id,
      observed_on: f.properties.observed_on,
      elevation_m: f.properties.elevation,
      twi: f.properties.twi,
      nlcd_class: f.properties.landcover,
      location: `POINT(${f.geometry.coordinates[0]} ${f.geometry.coordinates[1]})`,
    }));

    const { data, error } = await supabase
      .from('observations')
      .upsert(records, { onConflict: 'inat_id' });

    if (error) throw error;

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: `Successfully enriched ${records.length} observations.`,
        records,
      }),
    };
  } catch (err) {
    console.error('Pipeline Error:', err);
    return {
      statusCode: 500,
      body: JSON.stringify({ error: err.message }),
    };
  }
};