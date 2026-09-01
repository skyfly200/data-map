// Pure statistics behind the Analysis page.
//
// Deliberately free of any framework import: these are the parts where being
// subtly wrong is invisible — a correlation that reads 0.9 looks just as
// convincing whether or not the maths behind it is right — so they are kept
// importable and testable on their own.

/** A value that is actually there — not null, undefined or blank. */
export const isPresent = (v) => v !== null && v !== undefined && v !== ''

// Environmental variables worth relating to each other and to species. `unit`
// marks the ones that follow the ft/m and °F/°C toggles.
export const ANALYSIS_FIELDS = [
  { key: 'elevation', label: 'Elevation', unit: 'elev' },
  { key: 'day_of_year', label: 'Day of year' },
  { key: 'tmax', label: 'High temp', unit: 'temp' },
  { key: 'tmin', label: 'Low temp', unit: 'temp' },
  { key: 'rain7', label: '7-day rain' },
  { key: 'ndvi', label: 'NDVI' },
  { key: 'soil_moisture', label: 'Soil moisture' },
  { key: 'solar_exposure', label: 'Solar exposure' },
  { key: 'wind_exposure', label: 'Wind exposure' },
  { key: 'water_retention', label: 'Wetness index' },
  { key: 'slope', label: 'Slope' },
]

const PRCP = [0, 1, 2, 3, 4, 5, 6]

/** Numeric value of a field for one row, or null. `rain7` is a derived sum. */
export function fieldValue(row, key) {
  if (key === 'rain7') {
    let sum = 0
    let any = false
    for (const d of PRCP) {
      const v = row[`prcp_d${d}`]
      if (isPresent(v)) { sum += Number(v); any = true }
    }
    return any ? sum : null
  }
  const v = row[key]
  return isPresent(v) && Number.isFinite(Number(v)) ? Number(v) : null
}

/**
 * Spearman rank correlation between two aligned numeric arrays.
 *
 * Rank-based rather than Pearson because these relationships are monotonic but
 * not linear — elevation against temperature bends, and a Pearson coefficient
 * would understate it. Ties get average ranks, which is what makes the
 * coefficient correct on data with repeated values (day-of-year has many).
 */
export function spearman(xs, ys) {
  const n = xs.length
  if (n < 3) return null
  const rank = (values) => {
    const order = values.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0])
    const ranks = new Array(n)
    let i = 0
    while (i < n) {
      let j = i
      while (j + 1 < n && order[j + 1][0] === order[i][0]) j++
      const avg = (i + j) / 2 + 1
      for (let k = i; k <= j; k++) ranks[order[k][1]] = avg
      i = j + 1
    }
    return ranks
  }
  const rx = rank(xs)
  const ry = rank(ys)
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length
  const mx = mean(rx)
  const my = mean(ry)
  let num = 0
  let dx = 0
  let dy = 0
  for (let i = 0; i < n; i++) {
    const a = rx[i] - mx
    const b = ry[i] - my
    num += a * b
    dx += a * a
    dy += b * b
  }
  const den = Math.sqrt(dx * dy)
  return den ? num / den : null
}

/** Mean and (population) standard deviation, ignoring nulls. */
export function meanSd(values) {
  const vals = values.filter((v) => v !== null && Number.isFinite(v))
  if (!vals.length) return { n: 0, mean: null, sd: null }
  const mean = vals.reduce((s, v) => s + v, 0) / vals.length
  const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length
  return { n: vals.length, mean, sd: Math.sqrt(variance) }
}

export function median(values) {
  const vals = values.filter((v) => v !== null && Number.isFinite(v)).sort((a, b) => a - b)
  if (!vals.length) return null
  const mid = vals.length >> 1
  return vals.length % 2 ? vals[mid] : (vals[mid - 1] + vals[mid]) / 2
}
