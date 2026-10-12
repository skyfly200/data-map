import test from 'node:test'
import assert from 'node:assert/strict'
import { phenologyWeight, scoreCell, seasonWeights, resolveTimeSelection, monthMidDay, normaliseScores, rankCells, scoreBand, adjustForSample } from '../composables/forayScore.ts'
import { inSeasonSpecies, inSeasonDayOfYear } from '../composables/useInSeason.ts'

test('phenology weight: closer and tighter is heavier, bounded 0..1', () => {
  assert.equal(phenologyWeight(0, 0), 1)
  assert.ok(phenologyWeight(0, 0) > phenologyWeight(10, 0))
  assert.ok(phenologyWeight(5, 4) > phenologyWeight(5, 40))
  assert.ok(phenologyWeight(1000, 1000) > 0)
})

const weights = seasonWeights([{ name: 'A', dist: 0, iqr: 0 }, { name: 'B', dist: 14, iqr: 0 }])

test('scoreCell: weighted share of the cell own finds, effort-neutral', () => {
  const s = scoreCell({ n: 10, speciesInWindow: new Map([['A', 4], ['B', 2], ['Z', 3]]) }, weights)
  assert.equal(s.seasonalN, 6)
  assert.ok(Math.abs(s.share - 0.6) < 1e-9)
  assert.ok(Math.abs(s.score - (4 * 1 + 2 * 0.5) / 10) < 1e-9)
  assert.deepEqual(s.components.map((c) => c.species), ['A', 'B'])
  const big = scoreCell({ n: 100, speciesInWindow: new Map([['A', 40], ['B', 20]]) }, weights)
  assert.ok(Math.abs(big.score - s.score) < 1e-9)
})

test('scoreCell: thin samples are flagged, empty cells score 0', () => {
  assert.equal(scoreCell({ n: 2, speciesInWindow: new Map([['A', 2]]) }, weights).thin, true)
  const z = scoreCell({ n: 0, speciesInWindow: new Map() }, weights)
  assert.equal(z.score, 0)
  assert.equal(z.thin, true)
})

test('time selection: now vs month', () => {
  assert.deepEqual(resolveTimeSelection('now', 200), { day: 200, window: 14, label: 'Now' })
  const m = resolveTimeSelection(9, 200)
  assert.equal(m.label, 'Sep')
  assert.equal(m.day, monthMidDay(9))
  assert.equal(m.day, 258)
  assert.equal(resolveTimeSelection(99, 1).label, 'Dec')
})

test('normalise and rank', () => {
  const cells = [{ key: 'a', score: 0.2, n: 5 }, { key: 'b', score: 0.8, n: 5 }, { key: 'c', score: 0.8, n: 9 }]
  const norm = normaliseScores(cells)
  assert.equal(norm[0].t, 0)
  assert.equal(norm[1].t, 1)
  assert.deepEqual(rankCells(cells).map((c) => c.key), ['c', 'b', 'a'])
  assert.equal(rankCells(cells, 1).length, 1)
  assert.equal(normaliseScores([{ score: 1 }])[0].t, 0.5)
  assert.equal(scoreBand(0.9), 'Great')
  assert.equal(scoreBand(0.1), 'Low')
})

test('inSeasonSpecies: windowing, ordering, topN (dashboard behaviour)', () => {
  const day = inSeasonDayOfYear(new Date(2026, 8, 15))
  const mk = (species, date, elevation) => ({ species, date, elevation })
  const rows = [
    ...['2025-09-14', '2024-09-16', '2023-09-15'].map((d) => mk('Tight', d, 2500)),
    ...['2025-08-25', '2024-09-30', '2023-09-10', '2022-09-20'].map((d) => mk('Wide', d)),
    mk('Winter', '2025-01-10'), { species: 'NoDate' },
  ]
  const out = inSeasonSpecies(rows, { day, modelSpecies: new Set(['Tight']) })
  assert.deepEqual(out.map((s) => s.name), ['Tight', 'Wide'])
  assert.equal(out[0].hasModel, true)
  assert.equal(out[1].hasModel, false)
  assert.equal(out[0].count, 3)
  assert.deepEqual(out[0].elevBand, { loM: 2500, hiM: 2500 })
  assert.equal(out[1].elevBand, null)
  assert.equal(inSeasonSpecies(rows, { day, topN: 1 }).length, 1)
})

test('normaliseScores: all-zero cells read as Low, not Good', () => {
  const t = normaliseScores([{ score: 0 }, { score: 0 }]).map((c) => c.t)
  assert.deepEqual(t, [0, 0])
  assert.equal(scoreBand(t[0]), 'Low')
})

test('adjustForSample: a 3-find fluke ranks below a well-sampled good cell', () => {
  const cells = [
    { key: 'fluke', score: 1, n: 3 },
    { key: 'solid', score: 0.8, n: 40 },
    { key: 'poor', score: 0.05, n: 60 },
  ]
  const adj = adjustForSample(cells)
  assert.deepEqual(rankCells(adj, Infinity, 'adj').map((c) => c.key), ['solid', 'fluke', 'poor'])
  assert.equal(adj[0].score, 1, 'raw score is kept')
  for (const c of adj) assert.ok(c.adj >= 0 && c.adj <= 1)
})

import {
  ABSOLUTE_FULL, DEFAULT_PARAMS, absoluteT, blendScore, blendWeight, explainCell, gradeScores, paramsAreDefault, paramsToQuery,
  parseParams, scoreUnits,
} from '../composables/forayScore.ts'

test('phenology half-scale is tunable', () => {
  assert.ok(Math.abs(phenologyWeight(14, 0) - 0.5) < 1e-9)
  assert.ok(Math.abs(phenologyWeight(7, 0, 7) - 0.5) < 1e-9)
  assert.ok(seasonWeights([{ name: 'A', dist: 10, iqr: 0 }], 30).get('A') > seasonWeights([{ name: 'A', dist: 10, iqr: 0 }]).get('A'))
})

test('params: defaults, URL round trip, clamping and junk', () => {
  assert.deepEqual(parseParams(''), DEFAULT_PARAMS)
  assert.ok(paramsAreDefault(DEFAULT_PARAMS))
  const p = { ...DEFAULT_PARAMS, halfScale: 21, grading: 'absolute', modelMix: 0.3 }
  const q = paramsToQuery(p)
  assert.deepEqual(q, { fh: '21', fg: 'absolute', fx: '0.3' })
  assert.deepEqual(parseParams(new URLSearchParams(q)), p)
  const c = parseParams('fh=999&fp=-4&fm=abc&fg=weird&fx=0.33')
  assert.equal(c.halfScale, 45); assert.equal(c.priorFinds, 0); assert.equal(c.minSample, DEFAULT_PARAMS.minSample)
  assert.equal(c.grading, 'relative'); assert.equal(c.modelMix, 0.35)
})

test('absolute grading: fixed cut-offs; relative: min-max', () => {
  assert.equal(scoreUnits(ABSOLUTE_FULL), 1)
  assert.equal(absoluteT(ABSOLUTE_FULL * 2), 1)
  const cells = [{ score: 0, final: 0.1 }, { score: 0, final: 0.2 }]
  assert.deepEqual(gradeScores(cells, 'final', 'absolute').map((c) => c.t), [0.1, 0.2])
  assert.deepEqual(gradeScores(cells, 'final', 'relative').map((c) => c.t), [0, 1])
  assert.equal(scoreBand(gradeScores([{ score: 0, final: 0.2 }], 'final', 'absolute')[0].t), 'Low')
})

test('model blend: finds lead with many finds, model leads with none; mix 0 is finds only', () => {
  assert.equal(blendWeight(10, 0), 1)
  assert.equal(blendWeight(0, 0.5), 0.5)
  assert.equal(blendWeight(0, 1), 0)
  assert.ok(blendWeight(40, 0.5) > blendWeight(4, 0.5))
  assert.ok(blendWeight(10, 0.2) > blendWeight(10, 0.8))
  const none = blendScore(0, 0, 0.8, 0.5)
  assert.ok(Math.abs(none.blend - 0.4) < 1e-9)
  const many = blendScore(0.25, 200, 0.0, 0.5)
  assert.ok(many.blend > 0.45 && many.blend < 0.5)
  assert.deepEqual(blendScore(0.25, 5, null, 0.5), { findsT: 0.5, model: null, findsWeight: 1, blend: 0.5 })
  assert.equal(blendScore(0.25, 5, 0.9, 0).blend, 0.5)
})

test('explainCell: mean, rank, blend and per-species timing', () => {
  const cell = {
    key: 'b', n: 10, seasonalN: 6, score: 0.5, adj: 0.45, t: 0.8, final: 0.9,
    components: [{ species: 'A', weight: 1, count: 4, contribution: 0.4 }],
    blend: { findsT: 0.9, model: 0.7, findsWeight: 0.6, blend: 0.82 },
  }
  const e = explainCell(cell, {
    cells: [{ key: 'a', score: 0.1, n: 10 }, { key: 'b', score: 0.5, n: 10 }],
    ranked: [{ key: 'b' }, { key: 'a' }],
    species: [{ name: 'A', dist: 3, iqr: 12 }],
    params: { ...DEFAULT_PARAMS, grading: 'absolute' },
  })
  assert.equal(e.band, 'Great'); assert.equal(e.rank, 1); assert.equal(e.of, 2)
  assert.ok(Math.abs(e.mean - 0.3) < 1e-9)
  assert.equal(e.grading, 'absolute'); assert.equal(e.blend.model, 0.7); assert.equal(e.value, 0.9)
  assert.deepEqual(e.species[0], { species: 'A', count: 4, weight: 1, contribution: 0.4, dist: 3, iqr: 12 })
  assert.equal(explainCell({ ...cell, blend: { model: null } }, { cells: [], ranked: [], species: [], params: DEFAULT_PARAMS }).blend, null)
})
