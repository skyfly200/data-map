export interface Widget {
  id: string
  type: string
  label: string
  component?: string
  settings: Record<string, any>
}

export interface DashboardConfig {
  widgets: Widget[]
  order: string[]
}

export class DashboardStateManager {
  activeWidgets: Widget[] = []
  widgetOrder: string[] = []
  
  constructor(initialConfig?: DashboardConfig) {
    if (initialConfig) {
      this.activeWidgets = initialConfig.widgets || []
      this.widgetOrder = initialConfig.order || initialConfig.widgets.map(w => w.id)
    }
  }

  addWidget(widgetDef: { type: string; label: string; component?: string }) {
    const newWidget: Widget = {
      id: `${widgetDef.type}-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
      type: widgetDef.type,
      label: widgetDef.label,
      component: widgetDef.component,
      settings: {}
    }
    this.activeWidgets.push(newWidget)
    this.widgetOrder.push(newWidget.id)
    return newWidget
  }

  removeWidget(id: string) {
    this.activeWidgets = this.activeWidgets.filter(w => w.id !== id)
    this.widgetOrder = this.widgetOrder.filter(oid => oid !== id)
  }

  reorder(draggingId: string, targetId: string) {
    if (!draggingId || !targetId || draggingId === targetId) return
    
    const fromIndex = this.widgetOrder.indexOf(draggingId)
    const toIndex = this.widgetOrder.indexOf(targetId)

    if (fromIndex !== -1 && toIndex !== -1) {
      const newOrder = [...this.widgetOrder]
      newOrder.splice(fromIndex, 1)
      newOrder.splice(toIndex, 0, draggingId)
      this.widgetOrder = newOrder
    }
  }

  getConfig(): DashboardConfig {
    return {
      widgets: this.activeWidgets,
      order: this.widgetOrder
    }
  }

  setConfig(config: DashboardConfig) {
    this.activeWidgets = config.widgets || []
    this.widgetOrder = config.order || (config.widgets || []).map(w => w.id)
  }
}
