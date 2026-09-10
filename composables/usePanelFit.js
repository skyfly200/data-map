// Keep a dropdown panel inside the window.
//
// A panel anchored under its button runs off the screen when the button is near
// an edge — which on a phone is most of a toolbar. Whether it fits depends on
// the button's position, the panel's own width and the viewport, and CSS cannot
// compare those to each other, so it is measured.
//
// Shared because three components grew their own panel: PopoverMenu, ShareMenu
// and MapSettings. Fixing one of them left the other two off screen, which is
// exactly the kind of thing that comes back.

import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

/** Breathing room between the panel and the window edge. */
export const EDGE = 8

/**
 * How far to slide a panel so it sits inside the window.
 *
 * Pure, so the arithmetic can be checked without a browser: the rectangle in,
 * the horizontal offset out.
 */
export function fitOffset(rect, viewportWidth, edge = EDGE) {
  if (!rect) return 0
  let dx = 0
  if (rect.right > viewportWidth - edge) dx = viewportWidth - edge - rect.right
  // Never push it off the near edge in the process. A panel wider than the
  // window is clamped by its own max-width; this keeps its left edge visible.
  if (rect.left + dx < edge) dx = edge - rect.left
  return dx
}

/**
 * Track a panel and report the shift it needs.
 *
 * `panel` is a template ref to the panel element, `isOpen` a ref that says
 * whether it is showing. Returns a `shift` ref to bind as a translateX.
 */
export function usePanelFit(panel, isOpen) {
  const shift = ref(0)

  async function measure() {
    shift.value = 0
    if (!isOpen.value) return
    // After the panel has actually been laid out — before that it has no box.
    await nextTick()
    const el = panel.value
    if (!el) return
    shift.value = fitOffset(el.getBoundingClientRect(), window.innerWidth)
  }

  watch(isOpen, measure)

  onMounted(() => {
    // A rotation or a resize while a panel is open changes the answer.
    window.addEventListener('resize', measure)
  })
  onBeforeUnmount(() => window.removeEventListener('resize', measure))

  return { shift, measure }
}
