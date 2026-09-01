import test from 'node:test'
import assert from 'node:assert/strict'

import {
  SETTINGS_KEYS, CHARTS_KEY,
  mergeSettings, chartsToRows, rowsToCharts,
} from '../composables/useCloudSync.js'

// The network calls need a live Supabase; what is worth pinning without one is
// the logic that decides what gets written where — a wrong merge silently
// discards someone's settings, and a wrong row mapping loses a saved chart.

test('first sign-in keeps device-only settings and lets the account win elsewhere', () => {
  const local = { appearance: { palette: 'earth' }, units: { elev: 'ft' } }
  const remote = { appearance: { palette: 'okabe' } }

  const merged = mergeSettings(local, remote)
  // The account is the shared truth where both have an opinion...
  assert.deepEqual(merged.appearance, { palette: 'okabe' })
  // ...but work done on this device before signing up is not thrown away.
  assert.deepEqual(merged.units, { elev: 'ft' })
})

test('a fresh account adopts everything from the device', () => {
  const local = { appearance: { palette: 'vivid' }, 'chart-layout': { hidden: ['aspect'] } }
  assert.deepEqual(mergeSettings(local, null), local)
  assert.deepEqual(mergeSettings(local, {}), local)
})

test('a fresh device adopts everything from the account', () => {
  const remote = { appearance: { palette: 'pastel' } }
  assert.deepEqual(mergeSettings({}, remote), remote)
  assert.deepEqual(mergeSettings(null, remote), remote)
})

test('merging nothing with nothing is empty, not a crash', () => {
  assert.deepEqual(mergeSettings(null, null), {})
  assert.deepEqual(mergeSettings(undefined, undefined), {})
})

test('charts become rows with order preserved as position', () => {
  const charts = [
    { id: 'local1', type: 'scatter', xField: 'day_of_year', yField: 'elevation', title: 'A' },
    { id: 'local2', type: 'bar', groupField: 'species', title: 'B' },
  ]
  const rows = chartsToRows(charts, 'user-123')

  assert.deepEqual(rows.map((r) => r.position), [0, 1])
  assert.ok(rows.every((r) => r.user_id === 'user-123'))
  assert.deepEqual(rows.map((r) => r.title), ['A', 'B'])
  // The local id must NOT be stored: the database assigns the real one, and
  // writing the old one back would let two devices collide.
  assert.ok(rows.every((r) => !('id' in r.config)))
  // Everything else about the chart survives untouched.
  assert.equal(rows[0].config.xField, 'day_of_year')
  assert.equal(rows[1].config.groupField, 'species')
})

test('a chart with no title stores null rather than undefined', () => {
  const [row] = chartsToRows([{ id: 'x', type: 'bar' }], 'u')
  assert.equal(row.title, null)
})

test('rows become charts addressed by the row id', () => {
  const rows = [
    { id: 'uuid-1', config: { type: 'scatter', xField: 'tmax' }, title: 'Temp', position: 0 },
    { id: 'uuid-2', config: { type: 'bar', title: 'From config' }, title: null, position: 1 },
  ]
  const charts = rowsToCharts(rows)

  assert.deepEqual(charts.map((c) => c.id), ['uuid-1', 'uuid-2'])
  assert.equal(charts[0].title, 'Temp')
  // A row with no title column falls back to one inside the config blob.
  assert.equal(charts[1].title, 'From config')
  assert.equal(charts[0].xField, 'tmax')
})

test('charts survive a full round trip', () => {
  const original = [
    { id: 'a', type: 'scatter', xField: 'day_of_year', yField: 'elevation', colorField: 'species', title: 'One' },
    { id: 'b', type: 'box', valueField: 'elevation', groupField: 'genus', title: 'Two' },
  ]
  // Simulate the database assigning ids on insert.
  const stored = chartsToRows(original, 'u').map((r, i) => ({ ...r, id: `srv-${i}` }))
  const back = rowsToCharts(stored)

  assert.equal(back.length, original.length)
  back.forEach((c, i) => {
    const { id: _oldId, ...rest } = original[i]
    for (const [k, v] of Object.entries(rest)) {
      assert.deepEqual(c[k], v, `${k} did not survive the round trip`)
    }
    assert.equal(c.id, `srv-${i}`)
  })
})

test('empty and malformed inputs do not throw', () => {
  assert.deepEqual(chartsToRows([], 'u'), [])
  assert.deepEqual(chartsToRows(undefined, 'u'), [])
  assert.deepEqual(rowsToCharts([]), [])
  assert.deepEqual(rowsToCharts(null), [])
  assert.deepEqual(rowsToCharts(undefined), [])
})

test('the synced key list covers every preference the app persists', () => {
  // A preference missing here silently fails to follow the account, which is
  // the kind of gap nobody notices until they switch devices.
  for (const key of ['appearance', 'chart-layout', 'map-overlay', 'units']) {
    assert.ok(SETTINGS_KEYS.includes(key), `${key} is not synced`)
  }
  // Saved charts have their own table and must not also ride in the blob.
  assert.ok(!SETTINGS_KEYS.includes(CHARTS_KEY))
})
