import test, { describe } from 'node:test'
import assert from 'node:assert/strict'
import { DashboardStateManager } from '../composables/useDashboardState.js'

describe('Dashboard State Manager', () => {
  
  test('adding a widget updates active list and order', () => {
    const manager = new DashboardStateManager()
    const def = { type: 'recent-jobs', label: 'Recent Jobs' }
    
    const widget = manager.addWidget(def)
    
    assert.strictEqual(manager.activeWidgets.length, 1)
    assert.strictEqual(manager.widgetOrder.length, 1)
    assert.strictEqual(manager.activeWidgets[0].type, 'recent-jobs')
    assert.strictEqual(manager.widgetOrder[0], widget.id)
  })

  test('removing a widget cleans up both lists', () => {
    const manager = new DashboardStateManager()
    const w1 = manager.addWidget({ type: 'a', label: 'A' })
    const w2 = manager.addWidget({ type: 'b', label: 'B' })
    
    manager.removeWidget(w1.id)
    
    assert.strictEqual(manager.activeWidgets.length, 1)
    assert.strictEqual(manager.widgetOrder.length, 1)
    assert.strictEqual(manager.activeWidgets[0].id, w2.id)
    assert.strictEqual(manager.widgetOrder[0], w2.id)
  })

  test('reordering widgets (drag and drop logic)', () => {
    const manager = new DashboardStateManager()
    const w1 = manager.addWidget({ type: 'a', label: 'A' })
    const w2 = manager.addWidget({ type: 'b', label: 'B' })
    const w3 = manager.addWidget({ type: 'c', label: 'C' })
    
    // Initial order: [w1, w2, w3]
    // Move w1 to after w2 (index 1)
    manager.reorder(w1.id, w2.id)
    
    // Expected order: [w2, w1, w3]
    assert.deepEqual(manager.widgetOrder, [w2.id, w1.id, w3.id])
    
    // Move w3 to front
    manager.reorder(w3.id, w2.id)
    
    // Expected order: [w3, w2, w1]
    assert.deepEqual(manager.widgetOrder, [w3.id, w2.id, w1.id])
  })

  test('reordering with invalid IDs does nothing', () => {
    const manager = new DashboardStateManager()
    const w1 = manager.addWidget({ type: 'a', label: 'A' })
    const orderBefore = [...manager.widgetOrder]
    
    manager.reorder(w1.id, 'non-existent')
    assert.deepEqual(manager.widgetOrder, orderBefore)
    
    manager.reorder('non-existent', w1.id)
    assert.deepEqual(manager.widgetOrder, orderBefore)
  })

  test('setConfig restores state correctly', () => {
    const manager = new DashboardStateManager()
    const config = {
      widgets: [{ id: '1', type: 'a', label: 'A', settings: {} }],
      order: ['1']
    }
    
    manager.setConfig(config)
    
    assert.strictEqual(manager.activeWidgets.length, 1)
    assert.strictEqual(manager.widgetOrder[0], '1')
  })
})
