// Dashboard pages: tabs that each hold their own widgets and order, and the
// migration of a layout saved before pages existed.

import test, { describe } from 'node:test'
import assert from 'node:assert/strict'
import { DashboardStateManager } from '../composables/useDashboardState.js'

describe('Dashboard pages', () => {
  test('a new dashboard has one page, and widgets go on the active page', () => {
    const m = new DashboardStateManager()
    assert.strictEqual(m.pages.length, 1)
    const a = m.addWidget({ type: 'a' })
    const p2 = m.addPage('Data')
    assert.strictEqual(m.activePageId, p2.id)
    const b = m.addWidget({ type: 'b' })
    assert.deepEqual(m.pages[0].order, [a.id])
    assert.deepEqual(m.pages[1].order, [b.id])
    assert.deepEqual(m.widgetOrder, [b.id])
  })

  test('a page added without a name gets a numbered one; rename ignores blanks', () => {
    const m = new DashboardStateManager()
    const p = m.addPage()
    assert.strictEqual(p.name, 'Page 2')
    m.renamePage(p.id, '  Models ')
    assert.strictEqual(p.name, 'Models')
    m.renamePage(p.id, '   ')
    assert.strictEqual(p.name, 'Models')
  })

  test('reorder only touches the active page', () => {
    const m = new DashboardStateManager()
    const a = m.addWidget({ type: 'a' })
    const b = m.addWidget({ type: 'b' })
    m.addPage()
    m.reorder(b.id, a.id)
    assert.deepEqual(m.pages[0].order, [a.id, b.id])
  })

  test('moving a widget to another page appends it there', () => {
    const m = new DashboardStateManager()
    const a = m.addWidget({ type: 'a' })
    const b = m.addWidget({ type: 'b' })
    const p2 = m.addPage()
    const c = m.addWidget({ type: 'c' })
    m.moveWidgetToPage(a.id, p2.id)
    assert.deepEqual(m.pages[0].order, [b.id])
    assert.deepEqual(m.pages[1].order, [c.id, a.id])
    m.moveWidgetToPage(a.id, p2.id) // already there
    m.moveWidgetToPage('ghost', p2.id)
    m.moveWidgetToPage(b.id, 'ghost')
    assert.deepEqual(m.pages[1].order, [c.id, a.id])
    assert.deepEqual(m.pages[0].order, [b.id])
  })

  test('removing a widget removes it from its page, even when not active', () => {
    const m = new DashboardStateManager()
    const a = m.addWidget({ type: 'a' })
    m.addPage()
    m.removeWidget(a.id)
    assert.deepEqual(m.pages[0].order, [])
    assert.strictEqual(m.activeWidgets.length, 0)
  })

  test('removing a page drops its widgets and activates a neighbour; the last page stays', () => {
    const m = new DashboardStateManager()
    m.addWidget({ type: 'a' })
    const p2 = m.addPage()
    m.addWidget({ type: 'b' })
    m.removePage(p2.id)
    assert.strictEqual(m.pages.length, 1)
    assert.strictEqual(m.activeWidgets.length, 1)
    assert.strictEqual(m.activePageId, m.pages[0].id)
    m.removePage(m.pages[0].id)
    assert.strictEqual(m.pages.length, 1)
  })

  test('pages move left and right, and not off either end', () => {
    const m = new DashboardStateManager()
    const first = m.pages[0]
    const p2 = m.addPage('B')
    const p3 = m.addPage('C')
    m.movePage(p3.id, -1)
    assert.deepEqual(m.pages.map((p) => p.id), [first.id, p3.id, p2.id])
    m.movePage(first.id, -1)
    m.movePage(p2.id, 1)
    assert.deepEqual(m.pages.map((p) => p.id), [first.id, p3.id, p2.id])
  })
})

describe('Dashboard pages persistence', () => {
  test('pages and the active page round-trip through config', () => {
    const m1 = new DashboardStateManager()
    m1.addWidget({ type: 'a' })
    const p2 = m1.addPage('Data')
    const b = m1.addWidget({ type: 'b' })
    const cfg = JSON.parse(JSON.stringify(m1.getConfig()))
    const m2 = new DashboardStateManager()
    m2.setConfig(cfg)
    assert.deepEqual(m2.pages.map((p) => p.name), ['Home', 'Data'])
    assert.strictEqual(m2.activePageId, p2.id)
    assert.deepEqual(m2.widgetOrder, [b.id])
    assert.strictEqual(cfg.order.length, 2, 'flat order still lists every widget')
  })

  test('a layout saved before pages becomes one Home page in its saved order', () => {
    const m = new DashboardStateManager()
    m.setConfig({
      widgets: [{ id: 'x', type: 'a', settings: {} }, { id: 'y', type: 'b', settings: {} }],
      order: ['y', 'x'],
    })
    assert.strictEqual(m.pages.length, 1)
    assert.strictEqual(m.pages[0].name, 'Home')
    assert.deepEqual(m.widgetOrder, ['y', 'x'])
  })

  test('stale, duplicated and orphaned ids are cleaned up across pages', () => {
    const m = new DashboardStateManager()
    m.setConfig({
      widgets: [{ id: 'x', type: 'a', settings: {} }, { id: 'y', type: 'b', settings: {} }, { id: 'z', type: 'c', settings: {} }],
      order: [],
      pages: [
        { id: 'p1', name: 'One', order: ['stale', 'x'] },
        { id: 'p2', name: '', order: ['x', 'y'] },
      ],
      activePage: 'gone',
    })
    assert.deepEqual(m.pages[0].order, ['x', 'z'])
    assert.deepEqual(m.pages[1].order, ['y'])
    assert.strictEqual(m.pages[1].name, 'Page 2')
    assert.strictEqual(m.activePageId, 'p1')
  })

  test('a config with an empty pages list falls back to its flat order', () => {
    const m = new DashboardStateManager()
    m.setConfig({ widgets: [{ id: 'x', type: 'a', settings: {} }], order: ['x'], pages: [] })
    assert.strictEqual(m.pages.length, 1)
    assert.deepEqual(m.widgetOrder, ['x'])
  })
})
