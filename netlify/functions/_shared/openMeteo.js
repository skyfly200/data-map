// Fetches historical weather and 7-day precipitation lookback from Open-Meteo.
// Replaces the Python pipeline's ERA5-Land + CHIRPS raster-based approach.
// Free, no auth required. https://open-meteo.com/en/docs/historical-weather-api

function subtractDays(dateStr, n) {
  const d = new Date(dateStr)
  d.setUTCDate(d.getUTCDate() - n)
  return d.toISOString().slice(0, 10)
}

export async function fetchWeatherAndPrecip(lat, lon, observedOn) {
  // Fetch 7 days before + the observation day
  const startDate = subtractDays(observedOn, 7)
  const endDate = observedOn

  const params = new URLSearchParams({
    latitude: String(lat),
    longitude: String(lon),
    start_date: startDate,
    end_date: endDate,
    daily: [
      'temperature_2m_mean',
      'precipitation_sum',
      'wind_speed_10m_max'
    ].join(','),
    timezone: 'auto'
  })

  try {
    const res = await fetch(
      `https://archive-api.open-meteo.com/v1/archive?${params}`,
      { signal: AbortSignal.timeout(15_000) }
    )
    if (!res.ok) return nullWeather()

    const data = await res.json()
    if (!data.daily?.time) return nullWeather()

    const times = data.daily.time
    const obsIndex = times.indexOf(observedOn)
    if (obsIndex === -1) return nullWeather()

    const precip = {}
    for (let d = 0; d <= 6; d++) {
      const idx = obsIndex - d
      precip[`prcp_d${d}`] = idx >= 0 ? (data.daily.precipitation_sum[idx] ?? null) : null
    }

    return {
      temp_avg: data.daily.temperature_2m_mean[obsIndex] ?? null,
      precip_total: data.daily.precipitation_sum[obsIndex] ?? null,
      wind_speed: data.daily.wind_speed_10m_max[obsIndex] ?? null,
      ...precip
    }
  } catch {
    return nullWeather()
  }
}

function nullWeather() {
  return {
    temp_avg: null, precip_total: null, wind_speed: null,
    prcp_d0: null, prcp_d1: null, prcp_d2: null, prcp_d3: null,
    prcp_d4: null, prcp_d5: null, prcp_d6: null
  }
}
