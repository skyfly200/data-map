// The shape of a built chart, and how one travels in a URL.
//
// A chart is worth sharing on its own — "look at elevation against day of year,
// coloured by cluster" is a claim someone else should be able to open, check and
// argue with. That means the configuration has to survive a link, and the link
// has to be short enough to fit in a QR code alongside the filters that produced
// the data.
//
// So: one table, in a fixed order, holding the short code, the default and the
// type of every field. Encoding writes only what differs from the default, and
// decoding validates each value before it lands on the builder — a truncated or
// hand-edited link opens the app with less of the chart restored rather than a
// broken one.

// Separators chosen so URLSearchParams leaves them alone: `.` and `*` survive
// form encoding untouched, and no field key or value contains either. `-` and
// `_` do appear in values ("value-desc", "day_of_year"), so they cannot be used.
const PAIR_SEP = '*'
const KV_SEP = '.'

// [key, short code, default, kind, range?]
//
// The integer fields carry the same range their control offers. Without it a
// truncated link decodes `b.` to Number('') === 0 — a histogram with no bins —
// and a hand-edited one could ask for 50,000 of them.
const FIELDS = [
  ['type', 't', 'scatter', 'str'],
  ['xField', 'x', 'day_of_year', 'str'],
  ['yField', 'y', 'elevation', 'str'],
  ['colorField', 'c', 'cluster', 'str'],
  ['shapeField', 'sh', '', 'str'],
  ['sizeField', 'sz', '', 'str'],
  ['seriesField', 'se', '', 'str'],
  ['groupField', 'g', 'species', 'str'],
  ['valueField', 'v', 'elevation', 'str'],
  ['measure', 'm', 'count', 'str'],
  ['rowField', 'r', 'species', 'str'],
  ['colField', 'cl', 'land_cover_label', 'str'],
  ['bins', 'b', 10, 'int', [4, 30]],
  ['granularity', 'gr', 24, 'int', [4, 60]],
  ['horizontal', 'h', false, 'bool'],
  ['showToday', 'td', false, 'bool'],
  ['sortBy', 'so', 'value-desc', 'str'],
]

/** A fresh configuration with every field at its default. */
export function defaultChartConfig() {
  return Object.fromEntries(FIELDS.map(([key, , fallback]) => [key, fallback]))
}

/** The chart types the builder can render, for validating a decoded link. */
export const CHART_TYPES = [
  'scatter', 'bar', 'line', 'area', 'box', 'histogram', 'heatmap', 'radar', 'donut',
]

/**
 * A config as a compact string, carrying only what differs from the defaults.
 * A chart left at its defaults encodes to `''`, which the caller omits entirely.
 */
export function encodeChartConfig(config = {}) {
  const parts = []
  for (const [key, code, fallback, kind] of FIELDS) {
    const value = config[key]
    if (value === undefined || value === null) continue
    if (value === fallback) continue
    const out = kind === 'bool' ? (value ? '1' : '0') : String(value)
    if (out === '') continue // an empty string is only ever the default
    parts.push(`${code}${KV_SEP}${out}`)
  }
  return parts.join(PAIR_SEP)
}

/**
 * The inverse. Unknown codes and malformed values are dropped rather than
 * throwing, so a mangled link still opens a chart — just a more default one.
 */
export function decodeChartConfig(encoded) {
  const config = defaultChartConfig()
  if (typeof encoded !== 'string' || !encoded) return config

  const byCode = new Map(FIELDS.map((f) => [f[1], f]))
  for (const part of encoded.split(PAIR_SEP)) {
    const at = part.indexOf(KV_SEP)
    if (at < 1) continue
    const field = byCode.get(part.slice(0, at))
    if (!field) continue
    const [key, , , kind, range] = field
    const raw = part.slice(at + 1)

    if (kind === 'bool') {
      config[key] = raw === '1'
    } else if (kind === 'int') {
      // Not Number(raw): the empty string converts to 0, which is finite.
      const n = raw.trim() === '' ? NaN : Number(raw)
      if (Number.isFinite(n)) {
        const [lo, hi] = range
        config[key] = Math.min(hi, Math.max(lo, Math.round(n)))
      }
    } else {
      config[key] = raw
    }
  }
  if (!CHART_TYPES.includes(config.type)) config.type = 'scatter'
  return config
}

/** Just the configuration fields of a saved chart, without its id or title. */
export function chartConfigOf(chart = {}) {
  return Object.fromEntries(FIELDS.map(([key]) => [key, chart[key]])
    .filter(([, v]) => v !== undefined))
}

/**
 * A readable name for a chart, for the card heading and the share text.
 * Uses the saved title when there is one, and otherwise describes the chart
 * from its own configuration.
 */
export function describeChart(config = {}, labelFor = (k) => k) {
  if (config.title) return config.title
  const t = config.type || 'scatter'
  if (t === 'scatter' || t === 'line' || t === 'area') {
    return `${labelFor(config.yField)} vs. ${labelFor(config.xField)}`
  }
  if (t === 'histogram') return `Distribution of ${labelFor(config.valueField)}`
  if (t === 'box') return `${labelFor(config.valueField)} by ${labelFor(config.groupField)}`
  if (t === 'heatmap') return `${labelFor(config.rowField)} × ${labelFor(config.colField)}`
  const measure = config.measure && config.measure !== 'count'
    ? `Mean ${labelFor(config.measure)}` : 'Count'
  return `${measure} by ${labelFor(config.groupField)}`
}
