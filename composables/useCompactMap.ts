// Whether the map's controls should be consolidated for a narrow screen.
//
// A media query rather than a CSS breakpoint, because this decides WHERE a
// control renders and not just how it looks: on a phone the basemap and the
// heatmap move inside the layer window and clustering moves inside the points
// menu, which is a change of parent that CSS cannot make.
//
// The same 720px the stylesheet uses, so the two cannot disagree about what
// narrow means.

import { onBeforeUnmount, onMounted, ref } from 'vue'

export const COMPACT_QUERY = '(max-width: 720px)'

export function useCompactMap() {
  // False during server rendering, and false on the first client frame: the
  // server cannot know the viewport, so starting anywhere else guarantees a
  // hydration mismatch on every phone.
  const compact = ref(false)

  let mq: MediaQueryList | null = null
  const sync = () => { compact.value = !!mq?.matches }

  onMounted(() => {
    if (!window.matchMedia) return
    mq = window.matchMedia(COMPACT_QUERY)
    sync()
    mq.addEventListener('change', sync)
  })
  onBeforeUnmount(() => mq?.removeEventListener('change', sync))

  return { compact }
}
