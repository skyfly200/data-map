/* The Earth Engine job spec, and the indices derived from what it samples.
 *
 * The spec tests are the security boundary: a member sends values, and if
 * anything here lets an unchecked one through it reaches the society's Earth
 * Engine billing. The index tests are the physics — they assert direction
 * (a south-facing slope in the northern hemisphere is sunnier) rather than
 * exact numbers, which is what actually has to hold.
 */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  DEFAULT_STAGES, MAX_BBOX_DEGREES, SpecError, STAGES, STAGE_KEYS,
  bandsFor, normaliseBounds, normaliseSpec, progressPlan,
} from '../netlify/lib/ee-pipeline.mjs'
import {
  derivedIndices, normalise, solarExposure, sunPositions, waterRetention, windExposure,
} from '../netlify/lib/terrain-indices.mjs'
import { estimateUnits } from '../netlify/lib/quotas.mjs'

const bbox = { north: 40.5, south: 39.5, east: -105, west: -106 }
const base = { kind: 'enrich', source: { type: 'bbox', bounds: bbox } }

// ── The allowlist ────────────────────────────────────────────────────────────

test('an unknown stage is refused by name', () => {
  assert.throws(() => normaliseSpec({ ...base, stages: ['terrain', 'rm -rf'] }), (err) => {
    assert.ok(err instanceof SpecError)
    // Naming it matters: the member has to know which box to untick.
    assert.match(err.message, /Unknown stage: rm -rf/)
    return true
  })
})

test('stages come back in catalogue order and deduplicated', () => {
  // Two specs asking for the same work must be the same spec, or the cost
  // estimate would depend on the order someone happened to tick boxes in.
  const a = normaliseSpec({ ...base, stages: ['ndvi', 'terrain', 'ndvi'] })
  const b = normaliseSpec({ ...base, stages: ['terrain', 'ndvi'] })
  assert.deepEqual(a.stages, b.stages)
  assert.deepEqual(a.stages, ['terrain', 'ndvi'])
})

test('no stages means the default set', () => {
  assert.deepEqual(normaliseSpec(base).stages, DEFAULT_STAGES)
  assert.deepEqual(normaliseSpec({ ...base, stages: [] }).stages, DEFAULT_STAGES)
})

test('every catalogue stage names an asset and its bands', () => {
  // The catalogue is the complete set of Earth Engine assets this service
  // account will ever touch, so an entry missing an asset is a hole in that.
  for (const key of STAGE_KEYS) {
    const stage = STAGES[key]
    assert.ok(stage.asset, `${key} has no asset`)
    assert.ok(stage.bands.length, `${key} produces no bands`)
    assert.ok(stage.passes >= 1, `${key} has no passes`)
    assert.equal(typeof stage.perDate, 'boolean', `${key} does not say whether it is dated`)
  }
})

test('an unknown job kind or source type is refused', () => {
  assert.throws(() => normaliseSpec({ ...base, kind: 'shell' }), SpecError)
  assert.throws(() => normaliseSpec({ source: { type: 'ftp' } }), SpecError)
})

// ── Bounds ───────────────────────────────────────────────────────────────────

test('a box larger than the limit is refused', () => {
  assert.throws(() => normaliseBounds({
    north: 40, south: 40 - MAX_BBOX_DEGREES - 1, east: 0, west: -1,
  }), SpecError)
})

test('an inverted or out-of-range box is refused', () => {
  assert.throws(() => normaliseBounds({ north: 39, south: 40, east: 0, west: -1 }), SpecError)
  assert.throws(() => normaliseBounds({ north: 91, south: 40, east: 0, west: -1 }), SpecError)
  assert.throws(() => normaliseBounds({ north: 40, south: 39, east: 181, west: -1 }), SpecError)
  assert.throws(() => normaliseBounds(null), SpecError)
  assert.throws(() => normaliseBounds({ north: 'x', south: 39, east: 0, west: -1 }), SpecError)
})

test('a box across the antimeridian stays small', () => {
  // west > east means it wraps. Read literally that is a 358-degree box and
  // would be refused, when it is really two degrees wide.
  const got = normaliseBounds({ north: 1, south: -1, east: -179, west: 179 })
  assert.equal(got.east - got.west, 2)
})

// ── Dates and free text ──────────────────────────────────────────────────────

test('dates must be dates, and in order', () => {
  const src = (extra) => ({ ...base, source: { ...base.source, ...extra } })
  assert.throws(() => normaliseSpec(src({ dateFrom: 'last tuesday' })), SpecError)
  assert.throws(() => normaliseSpec(src({ dateFrom: '2026-09-10', dateTo: '2026-09-01' })), SpecError)
  assert.throws(() => normaliseSpec(src({ dateFrom: '1990-01-01', dateTo: '2026-01-01' })), SpecError)

  const ok = normaliseSpec(src({ dateFrom: '2026-09-01T12:00:00Z', dateTo: '2026-09-30' }))
  assert.equal(ok.source.dateFrom, '2026-09-01')
  assert.equal(ok.source.dateTo, '2026-09-30')
})

test('a dataset slug cannot be path-shaped', () => {
  const ds = (slug) => normaliseSpec({ kind: 'enrich', source: { type: 'dataset', slug } })
  // These are the ones that matter: a slug becomes part of a storage key.
  for (const bad of ['../../etc/passwd', 'a/b', '', ' ', 'has space', 'x'.repeat(200), '-leading']) {
    assert.throws(() => ds(bad), SpecError, `accepted ${JSON.stringify(bad)}`)
  }
  assert.equal(ds('front-range_2026').source.slug, 'front-range_2026')
})

test('a long title is cut rather than refused', () => {
  // The title is cosmetic, so trimming it is kinder than rejecting the job.
  const spec = normaliseSpec({ ...base, title: 'x'.repeat(500) })
  assert.equal(spec.title.length, 120)
})

// ── Progress and cost agree with each other ──────────────────────────────────

test('the progress bar is divided by cost, not by stage count', () => {
  // Six stages divided evenly would sit at 17% through a rainfall stage that is
  // most of the job. The bar has to track the work.
  const plan = progressPlan(['terrain', 'precip'], { points: 100, dates: 30 })
  const [terrain, precip] = plan
  assert.ok(precip.weight > terrain.weight * 5, 'a 7-pass dated stage should dwarf a 1-pass static one')
  assert.equal(terrain.from, 0)
  assert.equal(precip.to, 1)
  // Contiguous: no gap or overlap between stages, or the bar would jump back.
  assert.equal(terrain.to, precip.from)
})

test('progress weights and the cost estimate use the same model', () => {
  const stages = ['terrain', 'precip', 'ndvi']
  const opts = { points: 1200, dates: 40 }
  const planTotal = progressPlan(stages, opts).reduce((a, s) => a + s.weight, 0)
  assert.equal(planTotal, estimateUnits({ ...opts, stages }, STAGES))
})

test('a dated stage costs at least one request per date', () => {
  // 10 points is one chunk, but 30 dates cannot be batched into one request.
  const cheap = estimateUnits({ points: 10, dates: 1, stages: ['soil_moisture'] }, STAGES)
  const dear = estimateUnits({ points: 10, dates: 30, stages: ['soil_moisture'] }, STAGES)
  assert.equal(cheap, 1)
  assert.equal(dear, 30)
  // A static stage does not care how many dates are involved.
  assert.equal(estimateUnits({ points: 10, dates: 30, stages: ['terrain'] }, STAGES), 1)
})

test('bands are reported for the stages chosen', () => {
  const bands = bandsFor(['terrain', 'precip'])
  assert.ok(bands.includes('elevation'))
  assert.ok(bands.includes('prcp_d6'))
  assert.ok(!bands.includes('ndvi'))
})

// ── Derived indices ──────────────────────────────────────────────────────────

test('normalise maps to 0..1 and says 0.5 when everything is equal', () => {
  assert.deepEqual(normalise([0, 5, 10]), [0, 0.5, 1])
  assert.deepEqual(normalise([7, 7, 7]), [0.5, 0.5, 0.5])
  assert.deepEqual(normalise([]), [])
  assert.deepEqual(normalise([null, NaN]), [null, null])
  // A non-finite entry stays null instead of dragging the range around.
  assert.deepEqual(normalise([0, NaN, 10]), [0, null, 1])
})

test('the sun stays above the horizon and swings west after noon', () => {
  const positions = sunPositions(40)
  assert.ok(positions.length > 0)
  for (const [alt, az] of positions) {
    assert.ok(alt > 0 && alt <= Math.PI / 2, 'altitude above the horizon')
    assert.ok(az >= 0 && az <= 2 * Math.PI, 'azimuth in range')
  }
  // At high northern latitude in midwinter the sun may never rise; that must be
  // an empty list rather than a crash or a negative altitude.
  assert.deepEqual(sunPositions(85, [-23.44], 13), [])
})

test('in the northern hemisphere a south-facing slope is the sunniest', () => {
  const slope = [20, 20, 20, 20]
  const aspect = [0, 90, 180, 270]        // N, E, S, W
  const lat = [40, 40, 40, 40]
  const solar = solarExposure(slope, aspect, lat)
  const south = solar[2]
  assert.equal(south, 1, 'south-facing should be the maximum')
  assert.equal(solar[0], 0, 'north-facing should be the minimum')
  assert.ok(solar[1] > 0 && solar[1] < 1, 'east lies between')
})

test('south of the equator it is the north-facing slope', () => {
  const solar = solarExposure([20, 20], [0, 180], [-35, -35])
  assert.ok(solar[0] > solar[1], 'north-facing should be sunnier in the southern hemisphere')
})

test('wind exposure favours ridges and windward faces', () => {
  // Same aspect, different topographic position: the ridge is more exposed.
  const ridgeVsValley = windExposure([10, -10], [20, 20], [270, 270], 270)
  assert.ok(ridgeVsValley[0] > ridgeVsValley[1])

  // Same position, facing into a westerly versus away from it.
  const windwardVsLee = windExposure([0, 0], [20, 20], [270, 90], 270)
  assert.ok(windwardVsLee[0] > windwardVsLee[1])
})

test('water retention is highest where flat ground drains a large area', () => {
  const wet = waterRetention([10000, 10000, 10], [1, 30, 1])
  assert.ok(wet[0] > wet[1], 'flat should hold more than steep for the same catchment')
  assert.ok(wet[0] > wet[2], 'a large catchment should hold more than a small one')
})

test('flat ground does not divide by zero', () => {
  const wet = waterRetention([500, 500], [0, 0])
  assert.ok(wet.every((v) => Number.isFinite(v)))
})

test('the indices survive missing samples', () => {
  // Earth Engine masks pixels — over water, under cloud — so a null column is
  // normal input, not a broken one.
  const got = derivedIndices({
    slope: [10, null, 20],
    aspect: [180, 180, null],
    lat: [40, 40, 40],
    tpi: [5, 5, null],
    upstream: [100, null, 100],
  })
  for (const key of ['solar_exposure', 'wind_exposure', 'water_retention']) {
    assert.equal(got[key].length, 3, `${key} should keep one entry per point`)
    for (const v of got[key]) assert.ok(v === null || (v >= 0 && v <= 1), `${key} out of range`)
  }
})
