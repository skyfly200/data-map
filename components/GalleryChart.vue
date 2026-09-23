<template>
  <ChartCard
    v-if="visible"
    class="gallery-chart"
    :class="{ arranging: layout.editing.value,
              dragging: drag.dragging.value === id,
              'drop-target': isTarget }"
    :style="{ order: drag.orderOf(id) }"
    :draggable="layout.editing.value"
    @dragstart="drag.start(id, $event)"
    @dragenter.prevent="drag.enter(id)"
    @dragover.prevent
    @drop.prevent="drag.end()"
    @dragend="drag.end()"
  >
    <div v-if="layout.editing.value" class="layout-tools">
      <!-- The grip is what makes the card look draggable; the arrows stay
           because HTML drag-and-drop does not fire on touch, and because a
           keyboard user has no drag gesture at all. -->
      <span class="grip" title="Drag to rearrange" aria-hidden="true">⠿</span>
      <button title="Move earlier" :disabled="isFirst" @click="layout.move(id, -1)">‹</button>
      <button title="Move later" :disabled="isLast" @click="layout.move(id, 1)">›</button>
      <button title="Hide this chart" class="hide" @click="layout.hide(id)">✕</button>
    </div>
    <LazyVisible min-height="260px">
      <slot />
    </LazyVisible>
  </ChartCard>
</template>

<script setup>
const props = defineProps({ id: { type: String, required: true } })

const layout = useChartLayout()
const { pauseRendering, resumeRendering } = useRenderPause()

// Only cards actually on screen count as "shown": a preset whose data condition
// fails never mounts this component at all, and hiding one drops its card.
const visible = computed(() => layout.isVisible(props.id))
watch(visible, (v) => (v ? layout.register(props.id) : layout.unregister(props.id)), { immediate: true })
onUnmounted(() => layout.unregister(props.id))

const visibleOrder = computed(() => layout.order.value.filter((x) => layout.isVisible(x)))
const isFirst = computed(() => visibleOrder.value[0] === props.id)
const isLast = computed(() => visibleOrder.value[visibleOrder.value.length - 1] === props.id)

// Dragging reorders the VISIBLE cards only: a hidden chart has no card to drop
// onto, and including it would leave gaps a drag could fall into.
//
// Redraws are suspended for the duration. Changing a card's CSS order reflows
// the grid, which resizes every chart's container, which fires every chart's
// ResizeObserver — so without this a single pointer move recomputes the whole
// gallery. See composables/useRenderPause.js.
const drag = useDragReorder({
  key: 'gallery',
  ids: () => visibleOrder.value,
  onCommit: (next) => layout.setOrder(next),
  onPause: pauseRendering,
  onResume: resumeRendering,
})

const isTarget = computed(() =>
  drag.dragging.value && drag.over.value === props.id && drag.over.value !== drag.dragging.value)

// A drag abandoned with the keyboard has to release the render pause, or the
// charts stay frozen at whatever size they were.
function onKey(e) { if (e.key === 'Escape' && drag.dragging.value) drag.cancel() }
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => window.removeEventListener('keydown', onKey))
</script>

<style scoped>
/* The expand button sits top-right; these sit just left of it. */
/* Clears the card's own tools (save + full screen) sitting at the right edge. */
.layout-tools { position: absolute; top: 8px; right: 62px; display: flex; align-items: center; gap: 2px; z-index: 3; }
.layout-tools button {
  border: 1px solid var(--border); background: var(--surface); color: var(--muted); cursor: pointer;
  width: 22px; height: 22px; border-radius: 5px; font-size: 0.85rem; line-height: 1; padding: 0;
}
.layout-tools button:hover:not(:disabled) { background: var(--surface-2); color: var(--text); }
.layout-tools button:disabled { opacity: 0.35; cursor: default; }
.layout-tools .hide:hover { background: #fdecec; color: #b00020; border-color: #f5c2c2; }

.grip {
  color: var(--muted); font-size: 0.9rem; line-height: 1; padding: 0 4px;
  cursor: grab; user-select: none;
}

/* The card's own tools and these both float over the header, and the title
   reserves no room for either — so a long one ran underneath them. Adding the
   grip made that worse, so the space is reserved now, but only while
   arranging: the rest of the time the title should have the full width. */
.gallery-chart.arranging :deep(.chart-title) { padding-right: 156px; }
.gallery-chart[draggable='true'] { cursor: grab; }
.gallery-chart.dragging { opacity: 0.4; cursor: grabbing; }
/* Where it would land. A line rather than a filled state, so the card under it
   is still readable while deciding. */
.gallery-chart.drop-target { outline: 2px solid var(--accent); outline-offset: 2px; }

/* The lift is cosmetic and cheap, but a transition on every card would animate
   twenty reflows at once during a drag. Only the dragged card gets one. */
@media (prefers-reduced-motion: reduce) {
  .gallery-chart.dragging { opacity: 0.4; }
}
</style>
