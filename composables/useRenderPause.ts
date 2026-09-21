// A switch that stops charts redrawing themselves while something else needs
// the main thread.
//
// The case it exists for is dragging a card to rearrange the gallery. Changing
// a card's CSS `order` reflows the grid, every chart's container changes size
// as the columns settle, and each one's ResizeObserver fires — so a single
// pointer move can trigger twenty charts to recompute their geometry and redraw
// their SVG. The drag then stutters, which reads as the drag being broken
// rather than as the charts being busy.
//
// Nothing is unmounted or hidden while paused: the charts stay exactly as they
// were drawn, at the size they were drawn at. Only the recomputation stops.
// Anything that missed a measurement takes one when the pause lifts.

import { ref } from 'vue'

/** True while redraws are suspended. Module-level: one switch for the page. */
export const renderPaused = ref(false)

/** Nested pauses, so two callers cannot un-pause each other's work. */
let depth = 0

export function pauseRendering(): void {
  depth += 1
  renderPaused.value = true
}

export function resumeRendering(): void {
  depth = Math.max(0, depth - 1)
  if (depth === 0) renderPaused.value = false
}

/** Drop every outstanding pause. For a teardown that cannot trust its pairing. */
export function resetRenderPause(): void {
  depth = 0
  renderPaused.value = false
}

export function useRenderPause() {
  return { renderPaused, pauseRendering, resumeRendering, resetRenderPause }
}
