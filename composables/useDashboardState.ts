// The dashboard's layout: which widgets are on it, which page (tab) each is on,
// and in what order.
//
// Kept as a small plain class so the arithmetic of adding, removing and
// reordering is testable without a component, and so the same logic drives the
// page and its persistence. The composable below wraps one instance in reactive
// state and saves it per viewer.
//
// Widgets and their placement are kept apart: `activeWidgets` holds the widget
// objects, and each page holds its widgets' ids in display order (`widgetOrder`
// is the active page's). Order is a list of ids, not a
// sort of the widgets, because a drag reorders positions, and moving an id is
// cheaper and less error-prone than re-sorting objects — and it is exactly what
// a persisted layout stores.

export interface DashboardWidget {
  id: string
  type: string
  label?: string
  settings: Record<string, any>
}

/** A tab of the dashboard: a name and the ids of its widgets, in display order. */
export interface DashboardPage {
  id: string
  name: string
  order: string[]
}

export interface DashboardConfig {
  widgets: DashboardWidget[]
  /** Every page's ids concatenated, in page order. A layout saved before pages
   *  existed has only this, and it becomes the single first page. */
  order: string[]
  pages?: DashboardPage[]
  activePage?: string
}

const FIRST_PAGE_NAME = 'Home'

let seq = 0
function newId(prefix = 'w'): string {
  seq += 1
  try {
    if (typeof globalThis.crypto?.randomUUID === 'function') return `${prefix}-${globalThis.crypto.randomUUID()}`
  } catch { /* fall through */ }
  return `${prefix}-${Date.now().toString(36)}-${seq}`
}

export class DashboardStateManager {
  activeWidgets: DashboardWidget[] = []

  /** Never empty: a dashboard always has at least one page to put widgets on. */
  pages: DashboardPage[] = [{ id: 'page-home', name: FIRST_PAGE_NAME, order: [] }]

  activePageId = 'page-home'

  get activePage(): DashboardPage {
    return this.pages.find((p) => p.id === this.activePageId) || this.pages[0]
  }

  /** The active page's widget ids in display order. */
  get widgetOrder(): string[] { return this.activePage.order }
  set widgetOrder(ids: string[]) { this.activePage.order = ids }

  /** Add a widget to the active page from a definition ({ type, label, settings? }), return it. */
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

  /** Remove a widget by id, from whichever page holds it. A missing id is a no-op. */
  removeWidget(id: string): void {
    this.activeWidgets = this.activeWidgets.filter((w) => w.id !== id)
    for (const p of this.pages) p.order = p.order.filter((w) => w !== id)
  }

  /**
   * Move `movingId` to `targetId`'s slot on the active page — the drop-onto-target gesture.
   *
   * The target's index is read before the move, so dropping a widget from below
   * onto a target lands it just above that target, and from above lands it just
   * below. Either id being unknown is a no-op, so a drop onto nothing leaves the
   * order untouched.
   */
  reorder(movingId: string, targetId: string): void {
    const order = this.widgetOrder
    const targetIndex = order.indexOf(targetId)
    const movingIndex = order.indexOf(movingId)
    if (targetIndex === -1 || movingIndex === -1 || movingId === targetId) return
    order.splice(movingIndex, 1)
    order.splice(targetIndex, 0, movingId)
  }

  // ── Pages ──────────────────────────────────────────────────────────────────

  /** Add an empty page after the others, make it active, return it. */
  addPage(name?: string): DashboardPage {
    const page: DashboardPage = { id: newId('page'), name: (name || '').trim() || `Page ${this.pages.length + 1}`, order: [] }
    this.pages.push(page)
    this.activePageId = page.id
    return page
  }

  /** Rename a page. A blank name keeps the old one. */
  renamePage(id: string, name: string): void {
    const page = this.pages.find((p) => p.id === id)
    const trimmed = (name || '').trim()
    if (page && trimmed) page.name = trimmed
  }

  /**
   * Delete a page and the widgets on it. The last page cannot be deleted.
   * When the active page goes, its neighbour (the one before, else after) becomes active.
   */
  removePage(id: string): void {
    const index = this.pages.findIndex((p) => p.id === id)
    if (index === -1 || this.pages.length <= 1) return
    const [page] = this.pages.splice(index, 1)
    const gone = new Set(page.order)
    this.activeWidgets = this.activeWidgets.filter((w) => !gone.has(w.id))
    if (this.activePageId === id) this.activePageId = this.pages[Math.max(0, index - 1)].id
  }

  /** Shift a page left (-1) or right (+1) among the tabs. Off either end is a no-op. */
  movePage(id: string, delta: number): void {
    const from = this.pages.findIndex((p) => p.id === id)
    const to = from + delta
    if (from === -1 || to < 0 || to >= this.pages.length) return
    const [page] = this.pages.splice(from, 1)
    this.pages.splice(to, 0, page)
  }

  setActivePage(id: string): void {
    if (this.pages.some((p) => p.id === id)) this.activePageId = id
  }

  /** The page a widget is on, or undefined. */
  pageOf(widgetId: string): DashboardPage | undefined {
    return this.pages.find((p) => p.order.includes(widgetId))
  }

  /** Move a widget to the end of another page. Unknown ids, or the page it is already on, are a no-op. */
  moveWidgetToPage(widgetId: string, pageId: string): void {
    const from = this.pageOf(widgetId)
    const to = this.pages.find((p) => p.id === pageId)
    if (!from || !to || from === to) return
    from.order = from.order.filter((id) => id !== widgetId)
    to.order.push(widgetId)
  }

  // ── Persistence ────────────────────────────────────────────────────────────

  /** The layout as a plain object, for saving. */
  getConfig(): DashboardConfig {
    return {
      widgets: this.activeWidgets.map((w) => ({ ...w, settings: { ...w.settings } })),
      order: this.pages.flatMap((p) => p.order),
      pages: this.pages.map((p) => ({ ...p, order: [...p.order] })),
      activePage: this.activePageId,
    }
  }

  /**
   * Restore a saved layout. A config without pages (saved before pages existed)
   * becomes one page holding its whole order. Ids are rebuilt so every widget
   * renders exactly once: stale ids are dropped, an id on two pages stays on the
   * first, and a widget on no page is appended to the first page.
   */
  setConfig(config: DashboardConfig | null | undefined): void {
    const widgets = Array.isArray(config?.widgets) ? config!.widgets : []
    this.activeWidgets = widgets.map((w) => ({ ...w, settings: w.settings || {} }))
    const ids = new Set(this.activeWidgets.map((w) => w.id))

    const savedPages = Array.isArray(config?.pages) && config!.pages.length
      ? config!.pages
      : [{ id: 'page-home', name: FIRST_PAGE_NAME, order: Array.isArray(config?.order) ? config!.order : [] }]

    const placed = new Set<string>()
    const pageIds = new Set<string>()
    this.pages = savedPages.map((p, i) => {
      let id = typeof p?.id === 'string' && p.id ? p.id : newId('page')
      if (pageIds.has(id)) id = newId('page')
      pageIds.add(id)
      const order = (Array.isArray(p?.order) ? p.order : []).filter((wid) => {
        if (!ids.has(wid) || placed.has(wid)) return false
        placed.add(wid)
        return true
      })
      const name = typeof p?.name === 'string' && p.name.trim() ? p.name.trim() : (i === 0 ? FIRST_PAGE_NAME : `Page ${i + 1}`)
      return { id, name, order }
    })
    const missing = this.activeWidgets.map((w) => w.id).filter((id) => !placed.has(id))
    this.pages[0].order.push(...missing)

    this.activePageId = this.pages.some((p) => p.id === config?.activePage) ? config!.activePage! : this.pages[0].id
  }
}

// ─── The composable ──────────────────────────────────────────────────────────
// A reactive dashboard for the page, persisted per viewer under one key so it
// follows the account through cloud sync (see SETTINGS_KEYS in useCloudSync).

const STORAGE_KEY = 'dashboard-config'

export function useDashboardState() {
  const layout = useState<DashboardConfig>('dashboard-layout', () => ({ widgets: [], order: [], pages: [] }))
  const loaded = useState('dashboard-loaded', () => false)

  // Each caller (the page, a chart widget saving its settings) gets its own
  // manager, so it is reseeded from the shared layout before every change rather
  // than trusting what it held when the component mounted.
  const manager = new DashboardStateManager()
  const fresh = () => { manager.setConfig(layout.value); return manager }
  const sync = () => { layout.value = manager.getConfig() }

  function persist() {
    if (!import.meta.client) return
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(manager.getConfig())) } catch { /* private mode */ }
  }

  const commit = () => { sync(); persist() }

  function load(defaults?: DashboardConfig) {
    if (import.meta.client && !fresh().activeWidgets.length) {
      let saved: DashboardConfig | null = null
      try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null') } catch { saved = null }
      manager.setConfig(saved || defaults || { widgets: [], order: [] })
      sync()
    }
    loaded.value = true
  }

  const widgets = computed(() => layout.value.widgets)
  const pages = computed(() => layout.value.pages || [])
  const activePageId = computed(() => layout.value.activePage || pages.value[0]?.id || '')
  const activePage = computed(() => pages.value.find((p) => p.id === activePageId.value) || pages.value[0])
  const order = computed(() => activePage.value?.order || [])

  /** The active page's widgets in display order, ready to render. */
  const orderedWidgets = computed(() => order.value
    .map((id) => widgets.value.find((w) => w.id === id))
    .filter((w): w is DashboardWidget => Boolean(w)))

  function add(def: { type: string; label?: string; settings?: Record<string, any> }) {
    const w = fresh().addWidget(def)
    commit()
    return w
  }

  /** Add a new widget and insert it before `beforeId` in the order. */
  function addBefore(def: { type: string; label?: string; settings?: Record<string, any> }, beforeId: string) {
    const w = fresh().addWidget(def)
    // addWidget appends; move the new id to just before beforeId.
    manager.reorder(w.id, beforeId)
    commit()
    return w
  }

  function remove(id: string) { fresh().removeWidget(id); commit() }
  function reorder(movingId: string, targetId: string) { fresh().reorder(movingId, targetId); commit() }
  function reset(config: DashboardConfig) { manager.setConfig(config); commit() }

  function updateSettings(id: string, settings: Record<string, any>) {
    const w = fresh().activeWidgets.find((x) => x.id === id)
    if (!w) return
    w.settings = { ...w.settings, ...settings }
    commit()
  }

  function addPage(name?: string) { const p = fresh().addPage(name); commit(); return p }
  function renamePage(id: string, name: string) { fresh().renamePage(id, name); commit() }
  function removePage(id: string) { fresh().removePage(id); commit() }
  function movePage(id: string, delta: number) { fresh().movePage(id, delta); commit() }
  function setActivePage(id: string) { fresh().setActivePage(id); commit() }
  function moveWidgetToPage(widgetId: string, pageId: string) { fresh().moveWidgetToPage(widgetId, pageId); commit() }

  return {
    widgets, order, orderedWidgets, loaded, load, add, addBefore, remove, reorder, reset, updateSettings,
    pages, activePage, activePageId, addPage, renamePage, removePage, movePage, setActivePage, moveWidgetToPage,
  }
}
