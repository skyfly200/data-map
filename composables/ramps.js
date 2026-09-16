// Colour ramps, as a list of stops rather than a pair of ends.
//
// Everything that shades a number by colour — the grid heatmaps, the point
// gradient, the Earth Engine layer keys — used to be two colours with a linear
// interpolation between them. Two is enough for "more is darker" and not enough
// for anything else: a diverging scale needs a neutral middle, and the layer
// palettes this app already borrows from run to six and ten stops because that
// is what makes their middles readable.
//
// A two-stop ramp is just an N-stop ramp with N = 2, so the pair form keeps
// working and nothing had to be migrated to gain the rest.

/** Fewer than two is not a ramp. */
export const MIN_STOPS = 2

/**
 * More than this is a palette nobody can read as an ordering.
 *
 * Not a technical limit — the interpolation does not care — but a ramp with
 * fifteen stops stops reading as "low to high" and starts reading as a set of
 * categories, at which point the reader is decoding rather than seeing.
 */
export const MAX_STOPS = 8

const HEX = /^#[0-9a-f]{6}$/i

/** Six-digit hex, or null. Three-digit shorthand is expanded rather than lost. */
export function toHex(value) {
  const raw = String(value ?? '').trim()
  if (HEX.test(raw)) return raw.toLowerCase()
  const short = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(raw)
  if (short) return `#${short[1]}${short[1]}${short[2]}${short[2]}${short[3]}${short[3]}`.toLowerCase()
  return null
}

/**
 * A usable ramp, or null.
 *
 * Null rather than a repaired guess: these come from stored preferences that
 * sync between devices and from versions of this app that only knew about
 * pairs. A ramp that cannot be read should fall back to the default, not to
 * something invented from the half of it that parsed.
 */
export function normaliseStops(stops) {
  if (!Array.isArray(stops)) return null
  const clean = stops.map(toHex).filter(Boolean)
  if (clean.length < MIN_STOPS) return null
  return clean.slice(0, MAX_STOPS)
}

/** Clamped interpolation between two colours. */
export function mix(a, b, t) {
  const k = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0))
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16))
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16))
  return `#${pa.map((v, i) => Math.round(v + (pb[i] - v) * k).toString(16).padStart(2, '0')).join('')}`
}

/**
 * The colour at `t` along a ramp of any length.
 *
 * Stops are spaced evenly, which is what makes a ramp editable by adding one:
 * a stop dropped into the middle of a two-colour scale lands in the middle,
 * where the person who added it is looking.
 */
export function rampColor(stops, t) {
  if (!Array.isArray(stops) || !stops.length) return '#888888'
  if (stops.length === 1) return stops[0]
  const k = Math.max(0, Math.min(1, Number.isFinite(t) ? t : 0))
  const scaled = k * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(scaled))
  return mix(stops[i], stops[i + 1], scaled - i)
}

/** The CSS for a swatch of the whole ramp. */
export function gradientCss(stops, angle = '90deg') {
  const clean = normaliseStops(stops) || ['#888888', '#888888']
  return `linear-gradient(${angle}, ${clean.join(', ')})`
}

/**
 * A stop added after `index`, coloured as the ramp already is there.
 *
 * Inserted at the midpoint's own colour rather than at black or white, so
 * adding a stop changes nothing until it is moved — the ramp a person was
 * looking at is still the ramp they have.
 */
export function addStop(stops, index) {
  const clean = normaliseStops(stops) || ['#ffffff', '#000000']
  if (clean.length >= MAX_STOPS) return clean
  const at = Math.max(0, Math.min(clean.length - 2, Number(index) || 0))
  const middle = mix(clean[at], clean[at + 1], 0.5)
  return [...clean.slice(0, at + 1), middle, ...clean.slice(at + 1)]
}

/** A stop removed, unless that would leave fewer than two. */
export function removeStop(stops, index) {
  const clean = normaliseStops(stops) || []
  if (clean.length <= MIN_STOPS) return clean
  const at = Number(index)
  if (!Number.isInteger(at) || at < 0 || at >= clean.length) return clean
  return clean.filter((_, i) => i !== at)
}

/** One stop recoloured. Anything unparseable leaves the ramp as it was. */
export function setStop(stops, index, color) {
  const clean = normaliseStops(stops) || []
  const hex = toHex(color)
  const at = Number(index)
  if (!hex || !Number.isInteger(at) || at < 0 || at >= clean.length) return clean
  return clean.map((c, i) => (i === at ? hex : c))
}

/** The same ramp the other way up. */
export function reverseStops(stops) {
  const clean = normaliseStops(stops)
  return clean ? [...clean].reverse() : clean
}
