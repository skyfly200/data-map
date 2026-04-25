// Fetches mushroom observations from the iNaturalist REST API.
// Supports multi-page results up to maxPages * per_page observations.
export async function fetchInatObservations(config) {
  const {
    taxon_name = 'morchella',
    quality_grade = 'research',
    lat = 39.5,
    lng = -105.5,
    radius = 500,
    per_page = 200,
    maxPages = 10
  } = config

  const all = []
  let page = 1
  let totalResults = Infinity

  while (all.length < totalResults && page <= maxPages) {
    const params = new URLSearchParams({
      taxon_name,
      quality_grade,
      lat,
      lng: String(lng),
      radius: String(radius),
      per_page: String(Math.min(per_page, 200)),
      page: String(page),
      captive: 'false',
      geo: 'true',
      order: 'desc',
      order_by: 'created_at'
    })

    const res = await fetch(`https://api.inaturalist.org/v1/observations?${params}`)
    if (!res.ok) throw new Error(`iNat API error ${res.status}`)

    const data = await res.json()
    totalResults = data.total_results

    const results = (data.results || [])
      .filter(obs => obs.geojson && obs.observed_on)
      .map(obs => ({
        inat_uuid: obs.uuid,
        taxon_name: obs.taxon?.name ?? '',
        // iNat GeoJSON uses [lon, lat] order
        lon: obs.geojson.coordinates[0],
        lat: obs.geojson.coordinates[1],
        observed_on: obs.observed_on,
        quality_grade: obs.quality_grade,
        place_guess: obs.place_guess ?? ''
      }))

    all.push(...results)

    if (results.length < per_page) break
    page++
  }

  return all
}
