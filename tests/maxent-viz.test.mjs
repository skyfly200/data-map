import { test, describe } from 'node:test'
import assert from 'node:assert'
import { 
  mapValue, 
  scalePoint, 
  generateTicks, 
  matrixCellColor, 
  matrixTextColor, 
  matrixCellCoord, 
  scaleContribution, 
  DEFAULT_DIMS 
} from '../composables/maxentViz.js'

describe('MaxEnt Visualization Utilities', () => {
  
  describe('mapValue()', () => {
    test('maps midpoint of domain to midpoint of viewport', () => {
      // Domain [0, 100], padL=50, padR=50, total=200 -> range is 100px
      // v=50 should be 50 + (50/100)*100 = 100
      const result = mapValue(50, [0, 100], 50, 50, 200)
      assert.strictEqual(result, 100)
    })

    test('handles zero-width domain (division by zero)', () => {
      const result = mapValue(10, [10, 10], 0, 0, 100)
      assert.strictEqual(result, 0)
    })
  })

  describe('scalePoint()', () => {
    test('scales point correctly within default dims', () => {
      const p = { x: 0.5, y: 0.5 }
      const xDom = [0, 1]
      const yDom = [0, 1]
      const scaled = scalePoint(p, xDom, yDom)
      
      // cx: padL + 0.5 * (W - padL - padR) = 52 + 0.5 * (640 - 52 - 16) = 52 + 286 = 338
      assert.strictEqual(scaled.cx, 338)
      // cy: (H - padB) - 0.5 * (H - padT - padB) = (260 - 34) - 0.5 * (260 - 12 - 34) = 226 - 107 = 119
      assert.strictEqual(scaled.cy, 119)
    })
  })

  describe('generateTicks()', () => {
    test('generates correct number of ticks', () => {
      const ticks = generateTicks([0, 1], 4, true)
      assert.strictEqual(ticks.length, 5)
      assert.strictEqual(ticks[0].v, 0)
      assert.strictEqual(ticks[4].v, 1)
    })
  })

  describe('Confusion Matrix Utilities', () => {
    test('matrixCellColor returns expected RGB', () => {
      // lo=0, hi=100, v=0 -> t=0 -> [232, 241, 251]
      assert.strictEqual(matrixCellColor(0, 0, 100), 'rgb(232, 241, 251)')
      // lo=0, hi=100, v=100 -> t=1 -> [11, 61, 145]
      assert.strictEqual(matrixCellColor(100, 0, 100), 'rgb(11, 61, 145)')
    })

    test('matrixTextColor flips at threshold', () => {
      assert.strictEqual(matrixTextColor(0, 0, 100), 'var(--text)')
      assert.strictEqual(matrixTextColor(100, 0, 100), '#fff')
    })

    test('matrixCellCoord calculates positions', () => {
      const dims = DEFAULT_DIMS
      const coord = matrixCellCoord(0, 0, 5, 5, dims)
      assert.ok(coord.cx >= 90)
      assert.strictEqual(coord.cy, 26) // padT
    })
  })

  describe('scaleContribution()', () => {
    test('scales based on max value', () => {
      // v=5, maxV=10, padL=52, W=640, padR=16 -> 52 + 0.5 * 572 = 52 + 286 = 338
      const result = scaleContribution(5, 10)
      assert.strictEqual(result, 338)
    })
  })
})
