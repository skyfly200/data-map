<template>
  <div v-if="!isAuthed" class="dashboard-auth-guard">
    <div class="dag-message">
      <h2>Dashboard</h2>
      <p>Please log in to access your personalized dashboard.</p>
      <NuxtLink to="/login" class="dag-login-btn">Log In</NuxtLink>
    </div>
  </div>

  <div v-else class="dashboard">
    <header class="dashboard-header">
      <h1>My Dashboard</h1>
      <div class="dashboard-actions">
        <button 
          v-if="!isEditing" 
          class="dash-btn edit-btn" 
          @click="isEditing = true"
          title="Customize your dashboard"
        >
          ✏️ Edit Dashboard
        </button>
        <button 
          v-else 
          class="dash-btn save-btn" 
          @click="saveDashboard"
          title="Save your changes"
        >
          💾 Save Changes
        </button>
        <button 
          v-if="isEditing" 
          class="dash-btn cancel-btn" 
          @click="cancelEdit"
          title="Discard changes"
        >
          ✕ Cancel
        </button>
      </div>
    </header>

    <div class="dashboard-grid" :class="{ 'editing-mode': isEditing }">
      <!-- Available widgets to add -->
      <div v-if="isEditing" class="widget-palette">
        <h3>Add Widgets</h3>
        <div class="widget-options">
          <button 
            v-for="widget in availableWidgets" 
            :key="widget.type"
            class="widget-option"
            @click="addWidget(widget.type)"
            :disabled="activeWidgets.some(w => w.type === widget.type)"
          >
            <span class="widget-icon">{{ widget.icon }}</span>
            <span class="widget-name">{{ widget.label }}</span>
          </button>
        </div>
      </div>

      <!-- Active widgets grid -->
      <div 
        v-for="widget in activeWidgets" 
        :key="widget.id"
        class="dashboard-widget"
        :style="{ order: widgetOrder.indexOf(widget.id) }"
        @dragstart="onDragStart($event, widget.id)"
        @dragover="onDragOver"
        @drop="onDrop"
      >
        <div v-if="isEditing" class="widget-header">
          <span class="widget-drag-handle">⋮⋮</span>
          <button class="widget-remove" @click="removeWidget(widget.id)" title="Remove widget">×</button>
        </div>
        
        <component 
          :is="widget.component" 
          :widget="widget"
          :is-editing="isEditing"
        />
      </div>

      <div v-if="activeWidgets.length === 0" class="no-widgets">
        <p>No widgets yet. Click "Edit Dashboard" to add widgets.</p>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useAuth } from '~/composables/useAuth'
import { useCloudSync } from '~/composables/useCloudSync'
import { useDragReorder } from '~/composables/useDragReorder'

// Widget types available
const WIDGET_TYPES = [
  { type: 'saved-charts', label: 'Saved Charts', icon: '📊', component: 'DashboardSavedCharts' },
  { type: 'recent-jobs', label: 'Recent Jobs', icon: '⚙️', component: 'DashboardRecentJobs' },
  { type: 'species-list', label: 'Species List', icon: '🍄', component: 'DashboardSpeciesList' },
  { type: 'quick-filters', label: 'Quick Filters', icon: '🔍', component: 'DashboardQuickFilters' },
  { type: 'env-stats', label: 'Environmental Stats', icon: '🌡️', component: 'DashboardEnvStats' },
]

const { isAuthed } = useAuth()
const cloudSync = useCloudSync()

const isEditing = ref(false)
const activeWidgets = ref([])
const widgetOrder = ref([])

// Widgets that aren't already added
const availableWidgets = computed(() => {
  const activeTypes = new Set(activeWidgets.value.map(w => w.type))
  return WIDGET_TYPES.filter(w => !activeTypes.has(w.type))
})

// Drag and drop for reordering
const draggingId = ref('')

function onDragStart(event, id) {
  draggingId.value = id
  event.dataTransfer.effectAllowed = 'move'
}

function onDragOver(event) {
  event.preventDefault()
  event.dataTransfer.dropEffect = 'move'
}

function onDrop(event) {
  event.preventDefault()
  if (!draggingId.value) return
  
  const targetWidget = event.target.closest('.dashboard-widget')
  if (!targetWidget) return
  
  const targetId = targetWidget.querySelector('.dashboard-widget').dataset?.id || 
                   targetWidget.dataset?.id
  
  if (targetId && targetId !== draggingId.value) {
    const fromIndex = widgetOrder.value.indexOf(draggingId.value)
    const toIndex = widgetOrder.value.indexOf(targetId)
    
    if (fromIndex !== -1 && toIndex !== -1) {
      const newOrder = [...widgetOrder.value]
      newOrder.splice(fromIndex, 1)
      newOrder.splice(toIndex, 0, draggingId.value)
      widgetOrder.value = newOrder
    }
  }
  
  draggingId.value = ''
}

function addWidget(type) {
  const widgetDef = WIDGET_TYPES.find(w => w.type === type)
  if (!widgetDef) return
  
  const newWidget = {
    id: `${type}-${Date.now()}`,
    type: widgetDef.type,
    component: widgetDef.component,
    label: widgetDef.label,
    settings: {}
  }
  
  activeWidgets.value.push(newWidget)
  widgetOrder.value.push(newWidget.id)
}

function removeWidget(id) {
  activeWidgets.value = activeWidgets.value.filter(w => w.id !== id)
  widgetOrder.value = widgetOrder.value.filter(id_ => id_ !== id)
}

async function saveDashboard() {
  if (!import.meta.client) return
  
  const config = {
    widgets: activeWidgets.value,
    order: widgetOrder.value
  }
  
  try {
    localStorage.setItem('dashboard-config', JSON.stringify(config))
    await cloudSync.schedulePush()
    isEditing.value = false
  } catch (err) {
    console.error('Failed to save dashboard:', err)
  }
}

function cancelEdit() {
  loadDashboard()
  isEditing.value = false
}

function loadDashboard() {
  if (!import.meta.client) return
  
  try {
    const saved = localStorage.getItem('dashboard-config')
    if (saved) {
      const config = JSON.parse(saved)
      activeWidgets.value = config.widgets || []
      widgetOrder.value = config.order || activeWidgets.value.map(w => w.id)
    }
  } catch (err) {
    console.error('Failed to load dashboard:', err)
  }
}

onMounted(() => {
  if (isAuthed.value) {
    loadDashboard()
  }
})
</script>

<style scoped>
.dashboard-auth-guard {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 60vh;
}

.dag-message {
  text-align: center;
  padding: 2rem;
  background: var(--surface, #fff);
  border: 1px solid var(--border, #ddd);
  border-radius: 10px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.dag-message h2 {
  margin-bottom: 1rem;
  color: var(--text, #222);
}

.dag-login-btn {
  display: inline-block;
  margin-top: 1rem;
  padding: 0.75rem 1.5rem;
  background: var(--primary, #2a78d6);
  color: white;
  text-decoration: none;
  border-radius: 6px;
  font-weight: 600;
  transition: background 0.2s;
}

.dag-login-btn:hover {
  background: var(--primary-dark, #1e5fb8);
}

.dashboard {
  padding: 1.5rem;
  max-width: 1400px;
  margin: 0 auto;
}

.dashboard-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1.5rem;
  flex-wrap: wrap;
  gap: 1rem;
}

.dashboard-header h1 {
  margin: 0;
  color: var(--text, #222);
}

.dashboard-actions {
  display: flex;
  gap: 0.5rem;
}

.dash-btn {
  padding: 0.5rem 1rem;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s;
}

.edit-btn {
  background: var(--surface-2, #eee);
  color: var(--text, #222);
}

.edit-btn:hover {
  background: var(--surface-3, #ddd);
}

.save-btn {
  background: var(--primary, #2a78d6);
  color: white;
}

.save-btn:hover {
  background: var(--primary-dark, #1e5fb8);
}

.cancel-btn {
  background: transparent;
  color: var(--muted, #666);
}

.cancel-btn:hover {
  background: var(--surface-2, #eee);
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1.5rem;
}

.dashboard-grid.editing-mode {
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
}

.widget-palette {
  grid-column: 1 / -1;
  padding: 1rem;
  background: var(--surface-2, #f5f5f5);
  border-radius: 8px;
  margin-bottom: 1rem;
}

.widget-palette h3 {
  margin: 0 0 0.75rem 0;
  font-size: 0.9rem;
  color: var(--muted, #666);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.widget-options {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.widget-option {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 0.75rem;
  background: var(--surface, #fff);
  border: 1px solid var(--border, #ddd);
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
}

.widget-option:hover:not(:disabled) {
  border-color: var(--primary, #2a78d6);
  background: var(--surface-2, #f0f7ff);
}

.widget-option:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.widget-icon {
  font-size: 1.2rem;
}

.widget-name {
  font-size: 0.85rem;
  font-weight: 500;
}

.dashboard-widget {
  position: relative;
  background: var(--surface, #fff);
  border: 1px solid var(--border, #ddd);
  border-radius: 10px;
  padding: 1rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  transition: all 0.2s;
}

.dashboard-widget:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.dashboard-grid.editing-mode .dashboard-widget {
  cursor: move;
  border: 2px dashed var(--border, #ddd);
}

.dashboard-grid.editing-mode .dashboard-widget:hover {
  border-color: var(--primary, #2a78d6);
}

.widget-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border-soft, #eee);
}

.widget-drag-handle {
  cursor: move;
  color: var(--muted, #999);
  font-size: 1.2rem;
  user-select: none;
}

.widget-remove {
  background: transparent;
  border: none;
  color: var(--muted, #999);
  font-size: 1.5rem;
  line-height: 1;
  cursor: pointer;
  padding: 0 4px;
  transition: color 0.2s;
}

.widget-remove:hover {
  color: var(--danger, #dc3545);
}

.no-widgets {
  grid-column: 1 / -1;
  text-align: center;
  padding: 3rem;
  color: var(--muted, #999);
}
</style>
