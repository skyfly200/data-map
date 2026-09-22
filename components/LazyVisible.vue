<template>
  <div ref="root" class="lazy-visible" :style="shown ? undefined : { minHeight }">
    <slot v-if="shown" />
    <div v-else class="lazy-placeholder" aria-hidden="true"></div>
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'

// Render the slot only once it is near the viewport, and keep it rendered after.
//
// A gallery draws every saved chart, and a chart is not free to build even when
// it is a screen below the fold. This defers the ones that are not visible yet
// (V12-PERF-3), so the page paints the charts a reader can see and builds the
// rest as they scroll. Once a chart is shown it stays shown — re-tearing it down
// on scroll-away would trade one cost for another and lose its state.
//
// SSR- and no-IntersectionObserver-safe: with no observer it shows immediately,
// so the content is never trapped unrendered.

const props = defineProps({
  // Reserved height while deferred, so the page does not jump as items resolve.
  minHeight: { type: String, default: '240px' },
  // How far outside the viewport to start building, so it is ready by the time
  // it scrolls in rather than blank at the edge.
  rootMargin: { type: String, default: '300px' },
})

const root = ref(null)
const shown = ref(false)
let observer = null

onMounted(() => {
  if (typeof IntersectionObserver === 'undefined') { shown.value = true; return }
  observer = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      shown.value = true
      observer?.disconnect()
      observer = null
    }
  }, { rootMargin: props.rootMargin })
  if (root.value) observer.observe(root.value)
})

onBeforeUnmount(() => { observer?.disconnect() })
</script>

<style scoped>
.lazy-visible { display: block; }
.lazy-placeholder { width: 100%; height: 100%; }
</style>
