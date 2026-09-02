// What an observation drawer shows, and how each value is worded.
//
// Kept out of the component so the two drawers cannot drift apart, and so the
// parts with actual logic in them — the compass bearing, the rain total, the
// judgement about which readings to trust — can be tested without a browser.

// No framework imports, deliberately: reaching for useObservations here would
// drag Nuxt's auto-imports in and make this module unloadable outside the app,
// which is exactly what makes the logic below testable in plain Node.
const hasValue = (v) => v !== null && v !== undefined && v !== ''

const COMPASS = [
  'N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW',
]

/**
 * The 16-point compass name for a bearing. "213°" is a number; "SSW" is the
 * thing a forager actually wants to know about a slope.
 */
export function compassPoint(deg) {
  // Not just Number.isFinite: Number(null) and Number('') are both 0, so a
  // missing bearing would confidently report that the slope faces north.
  if (!hasValue(deg) || typeof deg === 'boolean') return ''
  const n = Number(deg)
  if (!Number.isFinite(n)) return ''
  // Round to the nearest of 16 sectors, wrapping so 350° reads N, not NNW.
  const i = Math.round(((n % 360) + 360) % 360 / 22.5) % 16
  return COMPASS[i]
}

/** Day offsets of the precipitation history columns. */
export const PRCP_OFFSETS = [0, 1, 2, 3, 4, 5, 6]

/**
 * Total rain in the week before the find, and how many of the seven days
 * actually carried a reading — a total over two days is not a week's rain, and
 * saying so is the difference between a number and a misleading one.
 */
export function rainLeadUp(props = {}) {
  let total = 0
  let days = 0
  for (const o of PRCP_OFFSETS) {
    const v = props[`prcp_d${o}`]
    if (!hasValue(v)) continue
    const n = Number(v)
    if (!Number.isFinite(n)) continue
    total += n
    days += 1
  }
  return days ? { total, days } : null
}

/** A 0–1 index as a percentage of its scale, for the little bar in the drawer. */
export function indexFraction(value, [lo, hi] = [0, 1]) {
  const n = Number(value)
  if (!Number.isFinite(n) || hi === lo) return null
  return Math.min(1, Math.max(0, (n - lo) / (hi - lo)))
}

const num = (v, dp = 2) => Number(v).toFixed(dp)

/**
 * The rows of the drawer, grouped, with empty groups dropped.
 *
 * `ctx` carries the unit-aware formatters, because elevation and temperature
 * follow the ft/m and °F/°C toggles and this module must not reach for global
 * state to find them.
 */
export function detailSections(props, ctx = {}) {
  if (!props) return []
  const { elevLabel = (v) => `${v} m`, tempLabel = (v) => `${v}°`, precisionLabel = null } = ctx
  const row = (label, value, extra = {}) => ({ label, value, ...extra })
  const sections = []

  // ── Record ────────────────────────────────────────────────────────────────
  const record = []
  if (hasValue(props.date)) {
    const doy = hasValue(props.day_of_year) ? ` · day ${Math.round(Number(props.day_of_year))}` : ''
    record.push(row('Observed', `${props.date}${doy}`))
  }
  if (hasValue(props.location)) record.push(row('Place', props.location))
  const lat = props.lat ?? props.latitude
  const lon = props.lon ?? props.longitude
  if (hasValue(lat) && hasValue(lon)) {
    record.push(row('Coordinates', `${num(lat, 4)}, ${num(lon, 4)}`))
  }
  if (hasValue(props.location_precision) && precisionLabel) {
    record.push(row('Precision', precisionLabel(props.location_precision), {
      // The one field that changes how everything below it should be read.
      warn: props.location_precision !== 'precise',
      hint: props.location_precision === 'precise'
        ? null
        : 'Terrain below was sampled at this point, which iNaturalist may have moved.',
    }))
  }
  if (hasValue(props.num_identification_agreements)) {
    const n = Number(props.num_identification_agreements)
    record.push(row('Agreements', `${n} ${n === 1 ? 'identifier' : 'identifiers'}`, {
      hint: n === 0 ? 'Nobody has confirmed this identification.' : null,
    }))
  }
  if (record.length) sections.push({ title: 'Record', rows: record })

  // ── Terrain ───────────────────────────────────────────────────────────────
  const terrain = []
  if (hasValue(props.elevation)) terrain.push(row('Elevation', elevLabel(props.elevation)))
  if (hasValue(props.slope)) terrain.push(row('Slope', `${num(props.slope, 1)}°`))
  if (hasValue(props.aspect)) {
    const point = compassPoint(props.aspect)
    terrain.push(row('Faces', `${point} · ${num(props.aspect, 0)}°`))
  }
  if (hasValue(props.land_cover_label)) terrain.push(row('Land cover', props.land_cover_label))
  for (const [key, label, bounds, hint] of [
    ['ndvi', 'NDVI', [-1, 1], 'Greenness. Higher is denser living vegetation.'],
    ['soil_moisture', 'Soil moisture', [0, 1], null],
    ['water_retention', 'Wetness index', [0, 1], 'How much water the terrain funnels here.'],
    ['solar_exposure', 'Solar exposure', [0, 1], null],
    ['wind_exposure', 'Wind exposure', [0, 1], null],
  ]) {
    if (!hasValue(props[key])) continue
    terrain.push(row(label, num(props[key], 2), { bar: indexFraction(props[key], bounds), hint }))
  }
  if (terrain.length) sections.push({ title: 'Terrain', rows: terrain })

  // ── Weather ───────────────────────────────────────────────────────────────
  const weather = []
  if (hasValue(props.tmax)) weather.push(row('High that day', tempLabel(props.tmax)))
  if (hasValue(props.tmin)) weather.push(row('Low that day', tempLabel(props.tmin)))
  const rain = rainLeadUp(props)
  if (rain) {
    weather.push(row('Rain, 7 days before', `${num(rain.total, 1)} mm`, {
      hint: rain.days < PRCP_OFFSETS.length
        ? `Only ${rain.days} of the 7 days carried a reading.`
        : null,
    }))
  }
  if (weather.length) sections.push({ title: 'Weather', rows: weather })

  return sections
}

/**
 * Which enrichment stages have not reached this record. Saying so beats leaving
 * a gap the reader has to notice for themselves.
 */
export function missingEnrichment(props) {
  if (!props) return []
  const missing = []
  if (!hasValue(props.slope) && !hasValue(props.aspect)) missing.push('terrain')
  if (!hasValue(props.ndvi) && !hasValue(props.soil_moisture)) missing.push('satellite')
  if (!rainLeadUp(props)) missing.push('weather')
  return missing
}
