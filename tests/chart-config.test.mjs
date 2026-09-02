import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  CHART_TYPES,
  chartConfigOf,
  decodeChartConfig,
  defaultChartConfig,
  describeChart,
  encodeChartConfig,
} from '../composables/chartConfig.js'

test('a default chart encodes to nothing', () => {
  assert.equal(encodeChartConfig(defaultChartConfig()), '')
})

test('only what differs from the default is written', () => {
  const cfg = { ...defaultChartConfig(), type: 'bar', horizontal: true }
  const encoded = encodeChartConfig(cfg)
  assert.equal(encoded, 't.bar*h.1')
})

test('a configuration survives the round trip', () => {
  const cfg = {
    ...defaultChartConfig(),
    type: 'heatmap',
    rowField: 'species',
    colField: 'cluster',
    measure: 'elevation',
    bins: 22,
    granularity: 40,
    horizontal: true,
    showToday: true,
    sortBy: 'label-asc',
    sizeField: 'ndvi',
  }
  assert.deepEqual(decodeChartConfig(encodeChartConfig(cfg)), cfg)
})

test('the encoding survives URL round-tripping unescaped', () => {
  // The point of the separators: a link stays readable and short, which is what
  // lets a shared chart fit in a QR code.
  const encoded = encodeChartConfig({ ...defaultChartConfig(), type: 'box', valueField: 'day_of_year', sortBy: 'value-asc' })
  const qs = new URLSearchParams({ cfg: encoded }).toString()
  assert.equal(qs, `cfg=${encoded}`)
  assert.equal(new URLSearchParams(qs).get('cfg'), encoded)
})

test('a mangled link still opens a chart', () => {
  const base = defaultChartConfig()
  assert.deepEqual(decodeChartConfig(''), base)
  assert.deepEqual(decodeChartConfig(null), base)
  assert.deepEqual(decodeChartConfig('garbage'), base)
  assert.deepEqual(decodeChartConfig('*.*..*'), base)
  // Unknown codes are skipped, known ones still land.
  assert.equal(decodeChartConfig('zz.9*t.donut').type, 'donut')
  // A truncated numeric keeps its default rather than becoming Number('') === 0,
  // which would ask a histogram for no bins at all.
  assert.equal(decodeChartConfig('b.').bins, base.bins)
  assert.equal(decodeChartConfig('b.abc').bins, base.bins)
})

test('an out-of-range number is clamped to what the control offers', () => {
  assert.equal(decodeChartConfig('b.50000').bins, 30)
  assert.equal(decodeChartConfig('b.0').bins, 4)
  assert.equal(decodeChartConfig('gr.-9').granularity, 4)
  assert.equal(decodeChartConfig('gr.999').granularity, 60)
})

test('an unknown chart type falls back rather than rendering nothing', () => {
  assert.equal(decodeChartConfig('t.notachart').type, 'scatter')
  for (const t of CHART_TYPES) {
    assert.equal(decodeChartConfig(`t.${t}`).type, t)
  }
})

test('chartConfigOf drops the identity fields a saved chart carries', () => {
  const saved = { id: 'c123', title: 'mine', type: 'bar', groupField: 'species' }
  const cfg = chartConfigOf(saved)
  assert.equal(cfg.id, undefined)
  assert.equal(cfg.title, undefined)
  assert.equal(cfg.type, 'bar')
  assert.equal(cfg.groupField, 'species')
})

test('a chart describes itself when it has no title', () => {
  const label = (k) => ({ elevation: 'Elevation', day_of_year: 'Day of year', species: 'Species' }[k] || k)
  assert.equal(describeChart({ title: 'My chart', type: 'bar' }, label), 'My chart')
  assert.equal(
    describeChart({ type: 'scatter', xField: 'day_of_year', yField: 'elevation' }, label),
    'Elevation vs. Day of year',
  )
  assert.equal(describeChart({ type: 'histogram', valueField: 'elevation' }, label), 'Distribution of Elevation')
  assert.equal(describeChart({ type: 'bar', groupField: 'species', measure: 'count' }, label), 'Count by Species')
  assert.equal(
    describeChart({ type: 'bar', groupField: 'species', measure: 'elevation' }, label),
    'Mean Elevation by Species',
  )
})
