/* Which models feed the foray layers, the candidate grid, and the
 * under-sampled ranking (WANT-17 phases 2, 3 and 5). */

import test from 'node:test'
import assert from 'node:assert/strict'

import {
  MAX_LAYER_MODELS, candidateCells, defaultPicks, findsPerCell, modelSpecies, modelsParam, rankUnderSampled, taxonCovers,
} from '../composables/forayModels.ts'
import { cellKeyAt } from '../composables/gridCells.ts'

const model = (id, o = {}) => ({ id, title: id, taxon: null, auc: 0.8, own: true, registered: false, ranges: true, contributions: true, usable: true, ...o })

test('taxonCovers matches a species or its genus, not a lookalike prefix', () => {
  assert.ok(taxonCovers('Cantharellus', 'Cantharellus cibarius'))
  assert.ok(taxonCovers('cantharellus cibarius', 'Cantharellus cibarius'))
  assert.ok(!taxonCovers('Boletus', 'Boletinellus merulioides'))
  assert.ok(!taxonCovers(null, 'Boletus edulis'))
})

test('modelSpecies falls back to a species named in the title', () => {
  assert.deepEqual(modelSpecies({ taxon: null, title: 'Morchella esculenta — Front Range' }, ['Morchella esculenta', 'Boletus edulis']), ['Morchella esculenta'])
  assert.deepEqual(modelSpecies({ taxon: 'Boletus', title: 'x' }, ['Boletus edulis', 'Boletus rubriceps', 'Suillus']), ['Boletus edulis', 'Boletus rubriceps'])
})

test('defaultPicks weights by the best in-season species and respects the layer', () => {
  const weights = new Map([['Boletus edulis', 0.9], ['Cantharellus cibarius', 0.4]])
  const models = [
    model('a', { taxon: 'Cantharellus' }),
    model('b', { taxon: 'Boletus' }),
    model('c', { taxon: 'Morchella' }),
    model('d', { taxon: 'Boletus edulis', ranges: false }),
  ]
  assert.deepEqual(defaultPicks(models, weights, 'habitat').map((p) => [p.id, p.weight]), [['b', 0.9], ['a', 0.4]])
  assert.deepEqual(defaultPicks(models, weights, 'ensemble').map((p) => p.id), ['b', 'd', 'a'])
  const many = Array.from({ length: 10 }, (_, i) => model(`m${i}`, { taxon: 'Boletus' }))
  assert.equal(defaultPicks(many, weights, 'ensemble').length, MAX_LAYER_MODELS)
  assert.equal(modelsParam([{ id: 'a', weight: 0.4, species: [] }]), 'a:0.400')
})

test('candidateCells covers the view once per cell and refuses a view too large', () => {
  const bounds = { west: -105.3, south: 39.9, east: -105.2, north: 40.0 }
  const { cells, tooMany } = candidateCells(bounds, 0.02, 'square')
  assert.equal(tooMany, false)
  assert.equal(new Set(cells.map((c) => c.key)).size, cells.length)
  assert.ok(cells.length >= 20 && cells.length <= 36, `got ${cells.length}`)
  assert.ok(cells.every((c) => c.lat >= 39.9 && c.lat <= 40 && c.lon >= -105.3 && c.lon <= -105.2))
  const hex = candidateCells(bounds, 0.02, 'hex')
  assert.ok(hex.cells.length > 10)
  assert.equal(candidateCells({ west: -109, south: 37, east: -102, north: 41 }, 0.02, 'hex').tooMany, true)
})

test('findsPerCell counts every cell with finds, honouring land cover', () => {
  const f = (lon, lat, lc = 'Forest') => ({ geometry: { coordinates: [lon, lat] }, properties: { land_cover_label: lc } })
  const n = findsPerCell([f(-105.25, 39.95), f(-105.25, 39.951), f(-105.25, 39.95, 'Grass')], 0.02, 'square', 'Forest')
  assert.equal(n.get(cellKeyAt(-105.25, 39.95, 0.02, 'square')), 2)
})

test('rankUnderSampled ranks promise over finds and drops the out-of-envelope', () => {
  const cells = [
    { key: 'a', n: 0, promise: 0.6, outside: false },
    { key: 'b', n: 2, promise: 0.9, outside: false },  // 0.3
    { key: 'c', n: 0, promise: 0.95, outside: true },  // extrapolation
    { key: 'd', n: 1, promise: 0.4, outside: false },  // below the bar
    { key: 'e', n: 5, promise: 0.99, outside: false }, // not under-sampled
    { key: 'f', n: 0, promise: null, outside: null },  // masked
  ]
  const r = rankUnderSampled(cells)
  assert.deepEqual(r.ranked.map((o) => o.cell.key), ['a', 'b'])
  assert.equal(r.outside, 1)
  assert.equal(r.considered, 4)
})
