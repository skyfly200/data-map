// End-to-end style tests for dashboard customization and the widget system.
//
// These exercise the full lifecycle of dashboard state without a browser:
// adding widgets, configuring them, reordering via drag-and-drop, removing,
// persisting and restoring from saved config. Covers DEBT-2.3.

import test, { describe } from 'node:test'
import assert from 'node:assert/strict'
import { DashboardStateManager } from '../composables/useDashboardState.js'

describe('Dashboard widget lifecycle', () => {
  test('add a widget → it appears at the end of the ordered list', () => {
    const m = new DashboardStateManager()
    const w1 = m.addWidget({ type: 'recent-jobs', label: 'Recent Jobs' })
    const w2 = m.addWidget({ type: 'species-list', label: 'Species' })

    assert.strictEqual(m.activeWidgets.length, 2)
    assert.deepEqual(m.widgetOrder, [w1.id, w2.id])
    assert.strictEqual(m.activeWidgets[1].type, 'species-list')
  })

  test('widget gets an id, type, label and empty settings by default', () => {
    const m = new DashboardStateManager()
    const w = m.addWidget({ type: 'env-stats' })

    assert.ok(w.id, 'id assigned')
    assert.strictEqual(w.type, 'env-stats')
    assert.deepEqual(w.settings, {})
    assert.strictEqual(w.label, undefined)
  })

  test('add with settings copies settings defensively', () => {
    const m = new DashboardStateManager()
    const src = { chartId: 'abc', size: 'large' }
    const w = m.addWidget({ type: 'saved-charts', settings: src })
    src.chartId = 'mutated'

    assert.strictEqual(w.settings.chartId, 'abc', 'mutation after add does not affect widget')
  })

  test('remove widget by id removes from both lists', () => {
    const m = new DashboardStateManager()
    const w1 = m.addWidget({ type: 'a' })
    const w2 = m.addWidget({ type: 'b' })
    const w3 = m.addWidget({ type: 'c' })

    m.removeWidget(w2.id)

    assert.strictEqual(m.activeWidgets.length, 2)
    assert.ok(!m.activeWidgets.some((w) => w.id === w2.id))
    assert.deepEqual(m.widgetOrder, [w1.id, w3.id])
  })

  test('remove unknown id is a no-op', () => {
    const m = new DashboardStateManager()
    const w = m.addWidget({ type: 'a' })

    m.removeWidget('does-not-exist')

    assert.strictEqual(m.activeWidgets.length, 1)
    assert.deepEqual(m.widgetOrder, [w.id])
  })
})

describe('Dashboard drag-and-drop reorder', () => {
  test('move first widget to last position', () => {
    const m = new DashboardStateManager()
    const [a, b, c] = ['a', 'b', 'c'].map((t) => m.addWidget({ type: t }))

    // reorder(moving, target): moving is placed at target's current index
    // Start [a, b, c]. reorder(a, c): targetIdx=2, after splice(0,1)=[b,c],
    // then splice(2,0,a)=[b,c,a].
    m.reorder(a.id, c.id)
    assert.deepEqual(m.widgetOrder, [b.id, c.id, a.id])
  })

  test('move last widget to first position', () => {
    const m = new DashboardStateManager()
    const [a, b, c] = ['a', 'b', 'c'].map((t) => m.addWidget({ type: t }))

    m.reorder(c.id, a.id)
    assert.deepEqual(m.widgetOrder, [c.id, a.id, b.id])
  })

  test('move adjacent widget does not duplicate or lose items', () => {
    const m = new DashboardStateManager()
    const [a, b, c] = ['a', 'b', 'c'].map((t) => m.addWidget({ type: t }))

    m.reorder(b.id, a.id)

    const ids = m.widgetOrder
    assert.strictEqual(ids.length, 3)
    assert.ok(ids.includes(a.id))
    assert.ok(ids.includes(b.id))
    assert.ok(ids.includes(c.id))
  })

  test('reorder with invalid source id is a no-op', () => {
    const m = new DashboardStateManager()
    const a = m.addWidget({ type: 'a' })
    const b = m.addWidget({ type: 'b' })
    const before = [...m.widgetOrder]

    m.reorder('ghost', b.id)

    assert.deepEqual(m.widgetOrder, before)
  })

  test('reorder with invalid target id is a no-op', () => {
    const m = new DashboardStateManager()
    const a = m.addWidget({ type: 'a' })
    const b = m.addWidget({ type: 'b' })
    const before = [...m.widgetOrder]

    m.reorder(a.id, 'ghost')

    assert.deepEqual(m.widgetOrder, before)
  })

  test('multiple reorders compose correctly', () => {
    const m = new DashboardStateManager()
    const [a, b, c, d] = ['a', 'b', 'c', 'd'].map((t) => m.addWidget({ type: t }))

    // Simulate a user rearranging several times
    // reorder(moving, target): moving lands at target's index before the splice.
    // Start: [a,b,c,d]
    // reorder(d,a): targetIdx=0, splice(3,1)=[a,b,c], splice(0,0,d)=[d,a,b,c]
    // reorder(b,c): [d,a,b,c], targetIdx=3, splice(2,1)=[d,a,c], splice(3,0,b)=[d,a,c,b]
    // reorder(a,c): [d,a,c,b], targetIdx=2, splice(1,1)=[d,c,b], splice(2,0,a)=[d,c,a,b]
    m.reorder(d.id, a.id)
    m.reorder(b.id, c.id)
    m.reorder(a.id, c.id)

    assert.strictEqual(m.widgetOrder.length, 4)
    assert.deepEqual(m.widgetOrder, [d.id, c.id, a.id, b.id])
  })
})

describe('Dashboard persist and restore', () => {
  test('getConfig round-trips through setConfig', () => {
    const m1 = new DashboardStateManager()
    const w1 = m1.addWidget({ type: 'recent-jobs', label: 'Jobs', settings: { max: 5 } })
    const w2 = m1.addWidget({ type: 'species-list' })
    m1.reorder(w2.id, w1.id)

    const cfg = m1.getConfig()

    const m2 = new DashboardStateManager()
    m2.setConfig(cfg)

    assert.strictEqual(m2.activeWidgets.length, 2)
    assert.deepEqual(m2.widgetOrder, [w2.id, w1.id])
    assert.strictEqual(m2.activeWidgets.find((w) => w.id === w1.id)?.settings.max, 5)
  })

  test('setConfig with null clears the dashboard', () => {
    const m = new DashboardStateManager()
    m.addWidget({ type: 'a' })
    m.setConfig(null)

    assert.strictEqual(m.activeWidgets.length, 0)
    assert.strictEqual(m.widgetOrder.length, 0)
  })

  test('setConfig with missing order falls back to widget insertion order', () => {
    const m = new DashboardStateManager()
    m.setConfig({
      widgets: [
        { id: 'x', type: 'a', settings: {} },
        { id: 'y', type: 'b', settings: {} },
      ],
      order: [],
    })

    assert.deepEqual(m.widgetOrder, ['x', 'y'])
  })

  test('setConfig ignores stale ids in order that have no matching widget', () => {
    const m = new DashboardStateManager()
    m.setConfig({
      widgets: [{ id: 'real', type: 'a', settings: {} }],
      order: ['stale', 'real', 'also-stale'],
    })

    assert.deepEqual(m.widgetOrder, ['real'])
  })

  test('setConfig with extra widgets not in order appends them', () => {
    const m = new DashboardStateManager()
    m.setConfig({
      widgets: [
        { id: '1', type: 'a', settings: {} },
        { id: '2', type: 'b', settings: {} },
      ],
      order: ['2'],
    })

    assert.strictEqual(m.widgetOrder.length, 2)
    assert.strictEqual(m.widgetOrder[0], '2')
    assert.strictEqual(m.widgetOrder[1], '1')
  })

  test('config is a snapshot: mutating the returned object does not change state', () => {
    const m = new DashboardStateManager()
    m.addWidget({ type: 'env-stats' })

    const cfg = m.getConfig()
    cfg.widgets[0].type = 'mutated'
    cfg.order.push('injected')

    assert.strictEqual(m.activeWidgets[0].type, 'env-stats')
    assert.strictEqual(m.widgetOrder.length, 1)
  })
})

describe('Dashboard widget settings', () => {
  test('each widget has its own settings object', () => {
    const m = new DashboardStateManager()
    const w1 = m.addWidget({ type: 'a', settings: { x: 1 } })
    const w2 = m.addWidget({ type: 'b', settings: { x: 2 } })

    assert.notStrictEqual(w1.settings, w2.settings)
    assert.strictEqual(w1.settings.x, 1)
    assert.strictEqual(w2.settings.x, 2)
  })

  test('all widget ids are unique across multiple adds', () => {
    const m = new DashboardStateManager()
    const ids = Array.from({ length: 20 }, () => m.addWidget({ type: 'a' }).id)
    const unique = new Set(ids)
    assert.strictEqual(unique.size, ids.length)
  })
})
