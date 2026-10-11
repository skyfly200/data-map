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
