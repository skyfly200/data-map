// Batch elevation lookup via Open-Elevation API.
// Accepts an array of {lat, lon} objects, returns array of elevation values
// in the same order. Returns null for any failed lookups.
export async function fetchElevationBatch(locations) {
  if (!locations.length) return []

  try {
    const res = await fetch('https://api.open-elevation.com/api/v1/lookup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({
        locations: locations.map(({ lat, lon }) => ({ latitude: lat, longitude: lon }))
      }),
      signal: AbortSignal.timeout(30_000)
    })

    if (!res.ok) return locations.map(() => null)

    const data = await res.json()
    return (data.results || []).map(r => r.elevation ?? null)
  } catch {
    return locations.map(() => null)
  }
}
