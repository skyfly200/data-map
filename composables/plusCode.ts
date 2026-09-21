// Open Location Code (plus code) encoding, ported from the reference
// implementation in utils/openlocationcode.py so the map can label a dropped
// point with the same code the enrichment pipeline speaks. Encode only: the map
// never needs to decode one.
//
// Pure and framework-free, so the arithmetic is unit-tested against the
// reference test vectors without a browser.

const ALPHABET = '23456789CFGHJMPQRVWX'
const SEPARATOR = '+'
const SEPARATOR_POSITION = 8
const ENCODING_BASE = 20
const LATITUDE_MAX = 90
const LONGITUDE_MAX = 180
const MAX_DIGITS = 15
const PAIR_CODE_LENGTH = 10
const GRID_COLUMNS = 4
const GRID_ROWS = 5
const GRID_CODE_LENGTH = MAX_DIGITS - PAIR_CODE_LENGTH // 5

// The precision multipliers, latitude and longitude, at the finest code length.
const FINAL_LAT_PRECISION = ENCODING_BASE ** 3 * GRID_ROWS ** GRID_CODE_LENGTH
const FINAL_LNG_PRECISION = ENCODING_BASE ** 3 * GRID_COLUMNS ** GRID_CODE_LENGTH

/** Location in degrees to the pair of positive integers the encoder works in. */
function locationToIntegers(latitude: number, longitude: number): [number, number] {
  let latVal = Math.floor(latitude * FINAL_LAT_PRECISION) + LATITUDE_MAX * FINAL_LAT_PRECISION
  if (latVal < 0) latVal = 0
  else if (latVal >= 2 * LATITUDE_MAX * FINAL_LAT_PRECISION) latVal = 2 * LATITUDE_MAX * FINAL_LAT_PRECISION - 1

  const lngRange = 2 * LONGITUDE_MAX * FINAL_LNG_PRECISION
  let lngVal = Math.floor(longitude * FINAL_LNG_PRECISION) + LONGITUDE_MAX * FINAL_LNG_PRECISION
  // Wrap the longitude rather than clamp it: 181° is a real place, 91° is not.
  if (lngVal < 0) lngVal = ((lngVal % lngRange) + lngRange) % lngRange
  else if (lngVal >= lngRange) lngVal %= lngRange
  return [latVal, lngVal]
}

/**
 * A location as an Open Location Code.
 *
 * `codeLength` is significant digits, not counting the `+`; 10 is the common
// "plus code" at roughly 14 m, 11 is about 3 m.
 */
export function encodePlusCode(latitude: number, longitude: number, codeLength = 10): string {
  codeLength = Math.min(codeLength, MAX_DIGITS)
  let [latVal, lngVal] = locationToIntegers(latitude, longitude)
  let code = ''

  if (codeLength > PAIR_CODE_LENGTH) {
    for (let i = 0; i < GRID_CODE_LENGTH; i += 1) {
      const latDigit = latVal % GRID_ROWS
      const lngDigit = lngVal % GRID_COLUMNS
      code = ALPHABET[latDigit * GRID_COLUMNS + lngDigit] + code
      latVal = Math.floor(latVal / GRID_ROWS)
      lngVal = Math.floor(lngVal / GRID_COLUMNS)
    }
  } else {
    latVal = Math.floor(latVal / GRID_ROWS ** GRID_CODE_LENGTH)
    lngVal = Math.floor(lngVal / GRID_COLUMNS ** GRID_CODE_LENGTH)
  }

  for (let i = 0; i < PAIR_CODE_LENGTH / 2; i += 1) {
    code = ALPHABET[lngVal % ENCODING_BASE] + code
    code = ALPHABET[latVal % ENCODING_BASE] + code
    latVal = Math.floor(latVal / ENCODING_BASE)
    lngVal = Math.floor(lngVal / ENCODING_BASE)
  }

  code = code.slice(0, SEPARATOR_POSITION) + SEPARATOR + code.slice(SEPARATOR_POSITION)
  if (codeLength >= SEPARATOR_POSITION) return code.slice(0, codeLength + 1)
  return code.slice(0, codeLength) + '0'.repeat(SEPARATOR_POSITION - codeLength) + SEPARATOR
}
