<template>
  <Teleport to="body">
    <TransitionGroup tag="div" name="alert-fade" class="app-alerts" aria-live="assertive" aria-atomic="true">
      <div
        v-for="a in alerts"
        :key="a.id"
        class="app-alert"
        :class="`app-alert--${a.level}`"
        role="alert"
      >
        <span class="app-alert__msg">{{ a.message }}</span>
        <button class="app-alert__close" aria-label="Dismiss" @click="dismiss(a.id)">✕</button>
      </div>
    </TransitionGroup>
  </Teleport>
</template>

<script setup lang="ts">
const { alerts, dismiss } = useAppAlerts()
</script>

<style scoped>
.app-alerts {
  position: fixed; bottom: 20px; right: 20px; z-index: 9999;
  display: flex; flex-direction: column; gap: 8px; max-width: min(420px, calc(100vw - 32px));
  pointer-events: none;
}
.app-alert {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 10px 12px; border-radius: 8px; font-size: 0.86rem;
  box-shadow: 0 4px 12px var(--shadow); pointer-events: all;
  border: 1px solid transparent;
}
.app-alert--error { background: #3d1515; border-color: #7a2020; color: #fca5a5; }
.app-alert--warn  { background: #3d2f10; border-color: #7a5c1e; color: #fcd34d; }
.app-alert--info  { background: var(--surface-2); border-color: var(--border); color: var(--text); }
:root[data-theme="light"] .app-alert--error { background: #fff1f1; border-color: #f87171; color: #b00020; }
:root[data-theme="light"] .app-alert--warn  { background: #fffbeb; border-color: #f59e0b; color: #92400e; }
:root[data-theme="light"] .app-alert--info  { background: var(--surface-2); border-color: var(--border); color: var(--text); }
.app-alert__msg { flex: 1; line-height: 1.4; }
.app-alert__close {
  flex: 0 0 auto; background: none; border: none; cursor: pointer;
  color: inherit; opacity: 0.6; font-size: 0.8rem; padding: 0 2px; line-height: 1;
}
.app-alert__close:hover { opacity: 1; }
.alert-fade-enter-active, .alert-fade-leave-active { transition: opacity 0.25s, transform 0.25s; }
.alert-fade-enter-from { opacity: 0; transform: translateY(8px); }
.alert-fade-leave-to { opacity: 0; transform: translateX(10px); }
</style>
