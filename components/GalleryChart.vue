<template>
  <ChartCard v-if="visible" class="gallery-chart" :style="{ order: layout.orderOf(id) }">
    <div v-if="layout.editing.value" class="layout-tools">
      <button title="Move earlier" :disabled="isFirst" @click="layout.move(id, -1)">‹</button>
      <button title="Move later" :disabled="isLast" @click="layout.move(id, 1)">›</button>
      <button title="Hide this chart" class="hide" @click="layout.hide(id)">✕</button>
    </div>
    <slot />
  </ChartCard>
</template>

<script setup>
const props = defineProps({ id: { type: String, required: true } })

const layout = useChartLayout()

// Only cards actually on screen count as "shown": a preset whose data condition
// fails never mounts this component at all, and hiding one drops its card.
const visible = computed(() => layout.isVisible(props.id))
watch(visible, (v) => (v ? layout.register(props.id) : layout.unregister(props.id)), { immediate: true })
onUnmounted(() => layout.unregister(props.id))

const visibleOrder = computed(() => layout.order.value.filter((x) => layout.isVisible(x)))
const isFirst = computed(() => visibleOrder.value[0] === props.id)
const isLast = computed(() => visibleOrder.value[visibleOrder.value.length - 1] === props.id)
</script>

<style scoped>
/* The expand button sits top-right; these sit just left of it. */
/* Clears the card's own tools (save + full screen) sitting at the right edge. */
.layout-tools { position: absolute; top: 8px; right: 62px; display: flex; gap: 2px; z-index: 3; }
.layout-tools button {
  border: 1px solid var(--border); background: var(--surface); color: var(--muted); cursor: pointer;
  width: 22px; height: 22px; border-radius: 5px; font-size: 0.85rem; line-height: 1; padding: 0;
}
.layout-tools button:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.layout-tools button:disabled { opacity: 0.35; cursor: default; }
.layout-tools .hide:hover { background: #fdecec; color: #b00020; border-color: #f5c2c2; }
</style>
