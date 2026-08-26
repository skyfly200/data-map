// Measures a chart's container so the SVG can be drawn at the real available
// size (viewBox === container px), letting it FILL the space instead of being
// letterboxed at a fixed aspect ratio. Attach `container` to the chart-area
// element; `width`/`height` update on resize.
//
// The chart-area must get its height from layout (flex-fill on the Explore
// page, or a min-height elsewhere) with the SVG absolutely positioned inside,
// so measuring the container never feeds back into its own size.
export function useChartSize(defaultW = 640, defaultH = 360, minW = 240, minH = 200) {
  const container = ref(null)
  const width = ref(defaultW)
  const height = ref(defaultH)
  let ro

  function measure() {
    const el = container.value
    if (!el) return
    const w = el.clientWidth
    const h = el.clientHeight
    if (w > 0) width.value = Math.max(minW, Math.round(w))
    if (h > 0) height.value = Math.max(minH, Math.round(h))
  }

  onMounted(() => {
    measure()
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(measure)
      if (container.value) ro.observe(container.value)
    }
  })
  onBeforeUnmount(() => ro && ro.disconnect())

  return { container, width, height }
}
