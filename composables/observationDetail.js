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

/**
 * Above this many metres of accuracy radius, a point is too vague to sample
 * terrain at. Matches COARSE_ACCURACY_M in scripts/iNat.py, which uses the same
 * threshold to classify a record as coarse — the two must agree or the drawer
 * would warn about a record the pipeline called precise.
 */
export const COARSE_ACCURACY_M = 1000

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

/**
 * What each row in the drawer means, for the hover explanation.
 *
 * Every row gets one. A drawer full of numbers with no units and no provenance
 * is a wall: "0.62" for wind exposure says nothing about whether that is a
 * ridge or a hollow, or whether it was measured or modelled. Where a value is
 * derived rather than observed, the tip says so — several of these are indices
 * computed from terrain, and reading them as measurements would be wrong.
 *
 * Kept beside the rows rather than in optionDocs.js, which documents CONTROLS.
 * These describe a record's own fields, and the two lists have no overlap.
 */
export const STAT_TIPS = {
  Observed: 'The date the fungus was seen, and its day of the year. Day of year is what the seasonal charts bin on.',
  Place: 'The locality iNaturalist recorded. Often a place name rather than the exact spot.',
  Coordinates: 'Where the terrain and weather below were sampled. Check the precision row before trusting them.',
  Precision: 'How exact the position is. iNaturalist blurs the location of sensitive taxa and of records whose owner asked it to, and everything sampled below was sampled at the blurred point.',
  Accuracy: 'The radius iNaturalist publishes for these coordinates: the fungus was somewhere within this distance of the point. Everything sampled below was sampled at the point itself.',
  Agreements: 'How many identifiers agreed with this name. Research grade needs two who agree; one or none means the identification is unconfirmed.',
  Elevation: 'Height above sea level at the recorded point, from the SRTM digital elevation model.',
  Slope: 'How steep the ground is, in degrees. 0° is flat, 45° is a scramble.',
  Faces: 'The compass direction the slope faces. In the northern hemisphere a south face is the warm dry one and a north face holds moisture longest.',
  'Land cover': 'The ESA WorldCover class at this point, at 10 m. Describes the surrounding pixel, not the individual tree the fungus was under.',
  NDVI: 'Greenness from satellite, between -1 and 1. Higher is denser living vegetation. Bare rock and water are near zero or below.',
  'Soil moisture': 'Modelled water in the top layer of soil on the day of the find, from ERA5-Land. A model at roughly 11 km, so it describes the district rather than the patch.',
  'Wetness index': 'How much uphill ground drains through this spot, from terrain shape alone. High means a gully or a seep; low means a ridge. Computed, not measured.',
  'Solar exposure': 'Modelled sunlight the ground receives across a year, from slope and aspect. High is an open south face, low is a shaded north one.',
  'Wind exposure': 'How exposed the spot is to wind, from terrain shape. High is a ridge or spur, low is a sheltered hollow. Computed, not measured.',
  'High that day': 'The daily maximum air temperature on the day of the find, from ERA5-Land.',
  'Low that day': 'The daily minimum air temperature on the day of the find, from ERA5-Land.',
  'Rain, 7 days before': 'Total rainfall over the week up to and including the day of the find, from CHIRPS. The lead-up that most often precedes a flush.',
  Genus: 'The genus this identification sits in, from the iNaturalist taxonomy.',
  Cluster: 'Which environmental cluster this record fell into. Clusters group records with similar terrain and weather, and the color matches the map.',
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
  // The tip is attached here rather than at each call site, so a row added
  // later carries its explanation automatically or shows up in the test that
  // asserts every row has one.
  const row = (label, value, extra = {}) => ({ label, value, tip: STAT_TIPS[label] || '', ...extra })
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
  // The accuracy radius in metres, beside the precision word. "Coarse" says
  // the point cannot be trusted for terrain; "±2,400 m" says how far off it
  // could be, which is the difference between "ignore this" and "this is the
  // right hillside but maybe the wrong gully".
  const accuracy = props.public_positional_accuracy ?? props.positional_accuracy
  if (hasValue(accuracy) && Number.isFinite(Number(accuracy))) {
    const m = Number(accuracy)
    record.push(row('Accuracy', m >= 1000
      ? `±${(m / 1000).toFixed(m >= 10000 ? 0 : 1)} km`
      : `±${Math.round(m)} m`, {
      // A radius wider than the terrain sampling is worth flagging: at that
      // point the elevation and slope below describe a different place.
      warn: m > COARSE_ACCURACY_M,
      hint: m > COARSE_ACCURACY_M
        ? 'Wider than the terrain below was sampled at, so treat those as the district, not the spot.'
        : null,
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
  // No per-field hint here any more: these were definitions, and definitions
  // now live in STAT_TIPS where every row has one and they can be read on
  // hover. The always-visible hint is reserved for a caveat about THIS record —
  // a blurred position, an unconfirmed identification, a partial rain week —
  // which is worth the space precisely because it is not always true.
  for (const [key, label, bounds] of [
    ['ndvi', 'NDVI', [-1, 1]],
    ['soil_moisture', 'Soil moisture', [0, 1]],
    ['water_retention', 'Wetness index', [0, 1]],
    ['solar_exposure', 'Solar exposure', [0, 1]],
    ['wind_exposure', 'Wind exposure', [0, 1]],
  ]) {
    if (!hasValue(props[key])) continue
    terrain.push(row(label, num(props[key], 2), { bar: indexFraction(props[key], bounds) }))
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
