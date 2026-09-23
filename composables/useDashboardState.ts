// The dashboard's layout: which widgets are on it, and in what order.
//
// Kept as a small plain class so the arithmetic of adding, removing and
// reordering is testable without a component, and so the same logic drives the
// page and its persistence. The composable below wraps one instance in reactive
// state and saves it per viewer.
//
// Two lists rather than one: `activeWidgets` holds the widget objects, and
// `widgetOrder` holds their ids in display order. Order is a list of ids, not a
// sort of the widgets, because a drag reorders positions, and moving an id is
// cheaper and less error-prone than re-sorting objects — and it is exactly what
// a persisted layout stores.

export interface DashboardWidget {
  id: string
  type: string
  label?: string
  settings: Record<string, any>
}

export interface DashboardConfig {
  widgets: DashboardWidget[]
  order: string[]
}

let seq = 0
function newId(): string {
  seq += 1
  try {
    if (typeof globalThis.crypto?.randomUUID === 'function') return `w-${globalThis.crypto.randomUUID()}`
  } catch { /* fall through */ }
  return `w-${Date.now().toString(36)}-${seq}`
}

export class DashboardStateManager {
  activeWidgets: DashboardWidget[] = []

  widgetOrder: string[] = []

  /** Add a widget from a definition ({ type, label, settings? }), return it. */
  addWidget(def: { type: string; label?: string; settings?: Record<string, any> }): DashboardWidget {
    const widget: DashboardWidget = {
      id: newId(),
      type: def.type,
      label: def.label,
      settings: def.settings ? { ...def.settings } : {},
    }
    this.activeWidgets.push(widget)
    this.widgetOrder.push(widget.id)
    return widget
  }

  /** Remove a widget by id from both lists. A missing id is a no-op. */
  removeWidget(id: string): void {
    this.activeWidgets = this.activeWidgets.filter((w) => w.id !== id)
    this.widgetOrder = this.widgetOrder.filter((w) => w !== id)
  }

  /**
   * Move `movingId` to `targetId`'s slot — the drop-onto-target gesture.
   *
   * The target's index is read before the move, so dropping a widget from below
   * onto a target lands it just above that target, and from above lands it just
   * below. Either id being unknown is a no-op, so a drop onto nothing leaves the
   * order untouched.
   */
  reorder(movingId: string, targetId: string): void {
    const targetIndex = this.widgetOrder.indexOf(targetId)
    const movingIndex = this.widgetOrder.indexOf(movingId)
    if (targetIndex === -1 || movingIndex === -1 || movingId === targetId) return
    this.widgetOrder.splice(movingIndex, 1)
    this.widgetOrder.splice(targetIndex, 0, movingId)
  }

  /** The layout as a plain object, for saving. */
  getConfig(): DashboardConfig {
    return { widgets: this.activeWidgets.map((w) => ({ ...w })), order: [...this.widgetOrder] }
  }

  /**
   * Restore a saved layout. Order is rebuilt from the widgets when it is missing
   * or has drifted, so a config that lost its order (or gained a widget without
   * one) still renders every widget exactly once.
   */
  setConfig(config: DashboardConfig | null | undefined): void {
    const widgets = Array.isArray(config?.widgets) ? config!.widgets : []
    this.activeWidgets = widgets.map((w) => ({ ...w, settings: w.settings || {} }))
    const ids = new Set(this.activeWidgets.map((w) => w.id))
    const savedOrder = Array.isArray(config?.order) ? config!.order.filter((id) => ids.has(id)) : []
    const missing = this.activeWidgets.map((w) => w.id).filter((id) => !savedOrder.includes(id))
    this.widgetOrder = [...savedOrder, ...missing]
  }
}

// ─── The composable ──────────────────────────────────────────────────────────
// A reactive dashboard for the page, persisted per viewer under one key so it
// follows the account through cloud sync (see SETTINGS_KEYS in useCloudSync).

const STORAGE_KEY = 'dashboard-config'

export function useDashboardState() {
  const widgets = useState<DashboardWidget[]>('dashboard-widgets', () => [])
  const order = useState<string[]>('dashboard-order', () => [])
  const loaded = useState('dashboard-loaded', () => false)

  const manager = new DashboardStateManager()
  const sync = () => {
    widgets.value = manager.activeWidgets.map((w) => ({ ...w }))
    order.value = [...manager.widgetOrder]
  }

  function persist() {
    if (!import.meta.client) return
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(manager.getConfig())) } catch { /* private mode */ }
  }

  function load(defaults?: DashboardConfig) {
    manager.setConfig({ widgets: widgets.value, order: order.value })
    if (import.meta.client && !manager.activeWidgets.length) {
      let saved: DashboardConfig | null = null
      try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') } catch { saved = null }
      manager.setConfig(saved || defaults || { widgets: [], order: [] })
      sync()
    }
    loaded.value = true
  }

  /** The widgets in display order, ready to render. */
  const orderedWidgets = computed(() => order.value
    .map((id) => widgets.value.find((w) => w.id === id))
    .filter((w): w is DashboardWidget => Boolean(w)))

  function add(def: { type: string; label?: string; settings?: Record<string, any> }) {
    const w = manager.addWidget(def)
    sync(); persist()
    return w
  }
  function remove(id: string) { manager.removeWidget(id); sync(); persist() }
  function reorder(movingId: string, targetId: string) { manager.reorder(movingId, targetId); sync(); persist() }
  function reset(config: DashboardConfig) { manager.setConfig(config); sync(); persist() }

  function updateSettings(id: string, settings: Record<string, any>) {
    const w = manager.activeWidgets.find((x) => x.id === id)
    if (!w) return
    w.settings = { ...w.settings, ...settings }
    sync(); persist()
  }

  return { widgets, order, orderedWidgets, loaded, load, add, remove, reorder, reset, updateSettings }
}
