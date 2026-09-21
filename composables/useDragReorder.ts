// Rearranging a list of cards by dragging one onto another.
//
// The arithmetic is separated from the pointer handling because the arithmetic
// is where the mistakes are. "Insert before or after the card I am over" flips
// depending on which direction the drag came from, and getting it wrong makes a
// card that refuses to reach the end of the list — a bug that looks like a
// broken drop target rather than like an off-by-one.
//
// Nothing here moves DOM nodes. The caller applies the result as CSS `order`,
// so a drag reflows the grid without re-mounting a single chart; the underlying
// array is written once, on drop. That is what keeps twenty charts draggable
// without re-rendering twenty charts per pointer move.

import { computed, ref } from 'vue'

/**
 * `list` with `fromId` lifted out and dropped where `toId` sits.
 *
 * Direction matters. Dragging forwards, the dragged card takes the target's
// place and the target shifts back; dragging backwards, it lands in front of
// the target. Both fall out of removing first and then inserting at the
// target's index in the SHORTENED list, which is why this is not just a splice
// at the original index.
 */
export function reorder(list: string[], fromId: string, toId: string): string[] {
  if (fromId === toId) return [...list]
  const from = list.indexOf(fromId)
  const to = list.indexOf(toId)
  // An id that is not in the list would otherwise insert at 0, silently moving
  // something the caller never asked to move.
  if (from < 0 || to < 0) return [...list]
  const next = [...list]
  next.splice(from, 1)
  next.splice(next.indexOf(toId) + (to > from ? 1 : 0), 0, fromId)
  return next
}

/**
 * The order to DRAW while a drag is in flight.
 *
 * The committed list is left alone until the drop, so a drag that is abandoned
// — escape, or released outside the grid — costs nothing to undo. Until then
// this is what the cards are positioned by.
 */
export function previewOrder(list: string[], draggingId: string | null | undefined, overId: string | null | undefined): string[] {
  if (!draggingId || !overId) return [...list]
  return reorder(list, draggingId, overId)
}

/**
 * Drag state, shared per list.
 *
 * Every card is its own component instance calling this, and they all have to
// agree on which one is being dragged — so the state is keyed and shared rather
// than created per caller. Module-level is safe here because a drag is a
// client-only, transient thing: there is no request during which it is anything
// but empty.
 */
interface DragState {
  dragging: Ref<string>
  over: Ref<string>
}

const shared = new Map<string, DragState>()
function stateFor(key: string): DragState {
  if (!shared.has(key)) shared.set(key, { dragging: ref(''), over: ref('') })
  return shared.get(key)!
}

interface DragReorderOptions {
  key?: string
  ids?: string[] | Ref<string[]> | (() => string[])
  onCommit?: (newOrder: string[]) => void
  onPause?: () => void
  onResume?: () => void
}

/**
 * Drag state for one list of cards.
 *
 * `key` names the list, so two lists on one page drag independently. `ids` is a
// getter for the current committed order; `onCommit` receives the new order
// once, on a successful drop. `onPause`/`onResume` bracket the drag so the
// caller can stop expensive redraws while the pointer is down.
 */
export function useDragReorder({ key = 'default', ids, onCommit, onPause, onResume }: DragReorderOptions = {}) {
  const { dragging, over } = stateFor(key)

  const current = () => (typeof ids === 'function' ? ids() : (ids?.value ?? ids ?? [])) as string[]

  /** The order to position by right now: the preview mid-drag, else the real one. */
  const shown = computed(() => previewOrder(current(), dragging.value, over.value))

  /** CSS `order` for one card. Cards not in the list sort to the end. */
  function orderOf(id: string): number {
    const i = shown.value.indexOf(id)
    return i === -1 ? shown.value.length : i
  }

  function start(id: string, event?: DragEvent) {
    dragging.value = id
    over.value = id
    onPause?.()
    if (event?.dataTransfer) {
      event.dataTransfer.effectAllowed = 'move'
      // Firefox ignores a drag that carries no payload at all.
      try { event.dataTransfer.setData('text/plain', id) } catch { /* not fatal */ }
    }
  }

  /**
   * The pointer has moved over a card.
   *
   * Entering the DRAGGED card is ignored, and that is not a nicety. The preview
// moves the dragged card to where it would land, which slides it under the
// pointer — so the browser immediately fires dragenter on it, `over` becomes
// the dragged card again, and the drop sees source === target and commits
// nothing. The drag looked right and did nothing.
   */
  function enter(id: string) {
    if (!dragging.value || id === dragging.value || id === over.value) return
    over.value = id
  }

  function end() {
    // The browser fires BOTH drop and dragend, and both handlers land here. The
    // second call has nothing to commit — and, more importantly, must not
// release a pause it never took, which would un-pause whatever else had
// asked for one.
    if (!dragging.value) return
    const next = shown.value
    const moved = dragging.value && over.value && dragging.value !== over.value
    dragging.value = ''
    over.value = ''
    // Resume before committing: the commit is the one redraw worth paying for,
    // and it should happen with measurement switched back on.
    onResume?.()
    if (moved) onCommit?.(next)
  }

  function cancel() {
    if (!dragging.value) return
    dragging.value = ''
    over.value = ''
    onResume?.()
  }

  return { dragging, over, shown, orderOf, start, enter, end, cancel }
}
