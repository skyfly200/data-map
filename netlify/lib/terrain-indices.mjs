// The derived terrain indices, computed at points rather than over a raster.
//
// scripts/terrain_pipeline.py computes these across a whole DEM scene; the
// Earth Engine path samples slope, aspect and TPI at the observation points and
// derives the same three indices from those samples. The formulas below are
// deliberately the same ones, so a dataset enriched through the app carries
// columns that mean what the shipped dataset's columns mean.
//
// One difference is inherent and is worth stating wherever these values are
// shown: the final min-max normalisation spans the points in this job, not a
// whole scene. The indices stay comparable BETWEEN observations in one dataset
// — which is what the map, the charts and the clustering use them for — but an
// absolute value is not comparable across two jobs over different points.

const DEG = Math.PI / 180

/** Min-max scale to 0..1. A constant input maps to 0.5, non-finite stays null. */
export function normalise(values) {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (!Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!Number.isFinite(lo)) return values.map(() => null)
  const span = hi - lo
  return values.map((v) => {
    if (!Number.isFinite(v)) return null
    // A dataset where every point has the same value carries no information
    // about which is more exposed; 0.5 says that honestly, where 0 or 1 would
    // imply an extreme.
    return span < 1e-12 ? 0.5 : (v - lo) / span
  })
}

/**
 * Sun positions over a day, at several declinations, as [altitude, azimuth].
 *
 * Averaging across the solstices and the equinox gives an annual potential-
 * radiation proxy, which is a stable property of the terrain rather than of the
 * day an observation happened to be made.
 */
export function sunPositions(latitude, declinations = [-23.44, 0, 23.44], nHours = 13) {
  const lat = latitude * DEG
  const out = []
  for (const decDeg of declinations) {
    const dec = decDeg * DEG
    for (let i = 0; i < nHours; i += 1) {
      // Hour angle from sunrise to sunset in even steps; 0 is solar noon.
      const H = (-90 + (180 * i) / (nHours - 1)) * DEG
      const sinAlt = Math.sin(lat) * Math.sin(dec) + Math.cos(lat) * Math.cos(dec) * Math.cos(H)
      if (sinAlt <= 0) continue                      // below the horizon
      const alt = Math.asin(sinAlt)
      let cosAz = (Math.sin(dec) - Math.sin(alt) * Math.sin(lat))
        / (Math.cos(alt) * Math.cos(lat) + 1e-9)
      cosAz = Math.max(-1, Math.min(1, cosAz))
      let az = Math.acos(cosAz)                      // clockwise from north
      if (H > 0) az = 2 * Math.PI - az               // afternoon: sun in the west
      out.push([alt, az])
    }
  }
  return out
}

/**
 * Potential incoming solar radiation, 0..1.
 *
 * For each modelled sun position, the cosine of the incidence angle on the
 * sloped surface, with faces turned away from the sun contributing nothing,
 * weighted by sin(altitude) so a high sun counts for more than a low one.
 */
export function solarExposure(slopeDeg = [], aspectDeg = [], latitudes = []) {
  const n = slopeDeg.length
  const raw = new Array(n).fill(null)
  // Sun geometry depends on latitude, and a job can span several degrees of it.
  // One set of positions per whole degree is far finer than the index's own
  // resolution and saves recomputing the model per point.
  const cache = new Map()

  for (let i = 0; i < n; i += 1) {
    const slope = slopeDeg[i]
    const aspect = aspectDeg[i]
    const lat = latitudes[i]
    if (!Number.isFinite(slope) || !Number.isFinite(aspect) || !Number.isFinite(lat)) continue

    const key = Math.round(lat)
    if (!cache.has(key)) cache.set(key, sunPositions(key))
    const positions = cache.get(key)

    const slopeRad = slope * DEG
    const aspectRad = aspect * DEG
    let total = 0
    let weight = 0
    for (const [alt, az] of positions) {
      const zenith = Math.PI / 2 - alt
      const cosInc = Math.cos(zenith) * Math.cos(slopeRad)
        + Math.sin(zenith) * Math.sin(slopeRad) * Math.cos(az - aspectRad)
      const flux = Math.sin(alt)
      total += Math.max(0, cosInc) * flux
      weight += flux
    }
    raw[i] = weight > 0 ? total / weight : null
  }
  return normalise(raw)
}

/**
 * Topographic wind exposure, 0..1.
 *
 * Two effects: openness, from the multi-scale TPI — a ridge sits above its
 * surroundings and catches wind where a valley is sheltered — and windward
 * aspect, since a slope facing into the prevailing wind is more exposed than a
 * lee slope of the same steepness.
 *
 * `prevailingWindDeg` is the direction the wind blows FROM (270 = westerly).
 */
export function windExposure(tpi = [], slopeDeg = [], aspectDeg = [], prevailingWindDeg = 270) {
  const openness = normalise(tpi)
  const windFrom = prevailingWindDeg * DEG
  const windward = normalise(slopeDeg.map((slope, i) => {
    const aspect = aspectDeg[i]
    if (!Number.isFinite(slope) || !Number.isFinite(aspect)) return null
    return 0.5 * (1 + Math.cos(aspect * DEG - windFrom)) * Math.sin(slope * DEG)
  }))
  return normalise(openness.map((o, i) => {
    const w = windward[i]
    if (o === null || w === null) return null
    return 0.7 * o + 0.3 * w
  }))
}

/**
 * Water retention, 0..1, as a topographic wetness proxy.
 *
 * The raster pipeline computes a true TWI from D8 flow accumulation, which
 * needs a full DEM and cannot be done from sampled points. Sampling MERIT
 * Hydro's upstream drainage area gives the same quantity's main input, so this
 * is ln(area / tan(slope)) — the TWI formula — over the sampled values.
 */
export function waterRetention(upstreamArea = [], slopeDeg = []) {
  const raw = upstreamArea.map((area, i) => {
    const slope = slopeDeg[i]
    if (!Number.isFinite(area) || !Number.isFinite(slope)) return null
    // Flat ground has no downhill and would divide by zero; the floor stands in
    // for the smallest gradient the DEM can express.
    const tan = Math.max(Math.tan(slope * DEG), 1e-4)
    return Math.log(Math.max(area, 1) / tan)
  })
  return normalise(raw)
}

/** All three, given the sampled columns. Returns an object of arrays. */
export function derivedIndices({ slope = [], aspect = [], lat = [], tpi = [], upstream = [] } = {},
                               { prevailingWindDeg = 270 } = {}) {
  return {
    solar_exposure: solarExposure(slope, aspect, lat),
    wind_exposure: windExposure(tpi, slope, aspect, prevailingWindDeg),
    water_retention: waterRetention(upstream, slope),
  }
}
