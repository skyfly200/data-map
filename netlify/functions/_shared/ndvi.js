// NDVI via Google Earth Engine REST API.
// Requires GEE_SERVICE_ACCOUNT_KEY env var (service account JSON with EE access).
// Returns null when unconfigured — clustering proceeds without NDVI in that case.
//
// To enable: create a GEE service account, grant it Earth Engine access,
// set GEE_SERVICE_ACCOUNT_KEY to the JSON key string in Netlify env vars.

export async function fetchNDVI(lat, lon, date) {
  const keyJson = process.env.GEE_SERVICE_ACCOUNT_KEY
  if (!keyJson) return null

  // TODO: implement GEE REST API call
  // 1. Parse service account key and mint a JWT
  // 2. Exchange JWT for an access token via OAuth2
  // 3. POST to https://earthengine.googleapis.com/v1/projects/{project}/value:compute
  //    with a Sentinel-2 NDVI computation for the given point and date range
  //
  // Reference: https://developers.google.com/earth-engine/reference/rest

  return null
}
