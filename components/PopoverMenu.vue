<template>
  <div ref="root" class="pop" :class="{ open }">
    <button
      type="button"
      class="pop-btn"
      :class="[{ on: open || active }, btnClass]"
      :aria-expanded="String(open)"
      :aria-haspopup="'true'"
      :title="title"
      @click="toggle"
    >
      <span class="pop-icon" aria-hidden="true">{{ icon }}</span>
      <!-- The label is for screens with room and for screen readers always. On a
           phone the icon carries it, which is the whole point of this component:
           six labelled dropdowns is three rows of chrome before any map. -->
      <span class="pop-label">{{ label }}</span>
      <span v-if="badge" class="pop-badge">{{ badge }}</span>
    </button>

    <div v-if="open" ref="panel" class="pop-panel" :class="align"
         :style="shift ? { transform: `translateX(${shift}px)` } : null"
         role="dialog" :aria-label="title || label">
      <div v-if="label" class="pop-head">{{ label }}</div>
      <slot />
    </div>
  </div>
</template>

<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'

// A control that costs one icon until someone wants it.
//
// The map had six labelled dropdowns laid side by side, which wrapped to three
// rows on a phone and took a quarter of the screen before drawing anything. The
// controls themselves were fine; showing all of them all of the time was not.

const props = defineProps({
  icon: { type: String, required: true },
  label: { type: String, default: '' },
  // Hover text. Falls back to the label, since a bare icon with no title is
  // unguessable for anyone who does not already know what it does.
  title: { type: String, default: '' },
  // Marks the button when the control behind it is doing something — a heatmap
  // is on, a filter is set. Otherwise closing the panel hides that fact.
  active: { type: Boolean, default: false },
  // A short current value, so the common case does not need opening at all.
  badge: { type: String, default: '' },
  align: { type: String, default: 'left' },
  btnClass: { type: String, default: '' },
})

const open = ref(false)
const root = ref(null)
const panel = ref(null)

// How far to slide the panel back into view.
//
// The panel is anchored to its button, so a button near the right edge opens a
// panel that runs off it — which on a phone is most of the bar. Measured rather
// than guessed, because it depends on the button's position, the panel's
// content and the viewport, none of which CSS can compare to each other.
const shift = ref(0)

const EDGE = 8

async function reposition() {
  shift.value = 0
  if (!open.value) return
  await nextTick()
  const el = panel.value
  if (!el) return
  const r = el.getBoundingClientRect()
  let dx = 0
  if (r.right > window.innerWidth - EDGE) dx = window.innerWidth - EDGE - r.right
  // Never push it off the other edge doing so: a panel wider than the viewport
  // is clamped by max-width, and this keeps its left edge on screen.
  if (r.left + dx < EDGE) dx = EDGE - r.left
  shift.value = dx
}

function toggle() { open.value = !open.value }
function close() { open.value = false }

function onDocPointer(e) {
  if (!open.value) return
  if (root.value && !root.value.contains(e.target)) close()
}
function onKey(e) {
  if (e.key === 'Escape' && open.value) {
    close()
    // Focus goes back to the button, or a keyboard user is stranded mid-page.
    root.value?.querySelector('.pop-btn')?.focus()
  }
}

onMounted(() => {
  // pointerdown rather than click: a click on another popover's button would
  // otherwise open that one and close this one in an order that depends on
  // which listener ran first.
  document.addEventListener('pointerdown', onDocPointer)
  document.addEventListener('keydown', onKey)
})
onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onDocPointer)
  document.removeEventListener('keydown', onKey)
})

// Only one open at a time on a phone, where two overlapping panels cover the
// whole screen. Broadcast on a shared event rather than a store: this component
// has no idea who its siblings are.
watch(open, (v) => {
  reposition()
  if (!v || !import.meta.client) return
  window.dispatchEvent(new CustomEvent('popover-open', { detail: root.value }))
})

// A rotation or a resize while a panel is open changes the answer.
onMounted(() => {
  window.addEventListener('resize', reposition)
  onBeforeUnmount(() => window.removeEventListener('resize', reposition))
})
onMounted(() => {
  const others = (e) => { if (e.detail !== root.value) close() }
  window.addEventListener('popover-open', others)
  onBeforeUnmount(() => window.removeEventListener('popover-open', others))
})

// Exposed so a keyboard shortcut can reach a control that now lives behind a
// button. Dropping the shortcut instead would quietly remove a documented one.
defineExpose({ close, toggle, show: () => { open.value = true } })
</script>

<style scoped>
.pop { position: relative; display: inline-flex; }

.pop-btn {
  display: inline-flex; align-items: center; gap: 6px;
  min-height: 34px; padding: 0 9px;
  background: var(--surface, #fff); color: var(--text, #222);
  border: 1px solid var(--border, #ddd); border-radius: 8px;
  font: inherit; font-size: 0.82rem; cursor: pointer; white-space: nowrap;
}
.pop-btn:hover { border-color: var(--muted, #999); }
.pop-btn.on { border-color: var(--accent, #2b7a3d); box-shadow: 0 0 0 2px rgba(43, 122, 61, 0.18); }
.pop-icon { font-size: 0.95rem; line-height: 1; }
.pop-badge {
  background: var(--surface-2, #eee); border-radius: 999px;
  padding: 1px 6px; font-size: 0.7rem; color: var(--muted, #666); max-width: 11ch;
  overflow: hidden; text-overflow: ellipsis;
}

.pop-panel {
  position: absolute; top: calc(100% + 6px); z-index: 1300;
  min-width: 210px; max-width: min(300px, calc(100vw - 24px));
  background: var(--surface, #fff); color: var(--text, #222);
  border: 1px solid var(--border, #ddd); border-radius: 10px;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.18);
  padding: 10px 12px;
  display: flex; flex-direction: column; gap: 9px;
  /* Long panels (the heatmap has a mode list, a cell size and a season window)
     scroll rather than running off the bottom of a phone. */
  max-height: min(70vh, 520px); overflow-y: auto; overscroll-behavior: contain;
}
.pop-panel.left { left: 0; }
.pop-panel.right { right: 0; }

.pop-head {
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted, #777); font-weight: 700;
}

/* Below this the label is dead weight: the icon and the panel's own heading say
   the same thing, and the row has to fit on a 390px screen. */
@media (max-width: 720px) {
  /* The badge goes with the label. It is worth a row of screen on a desktop and
     not on a phone, where the same answer is one tap away and the map is the
     thing you came for. The active outline still says a heatmap is on. */
  .pop-label, .pop-badge { display: none; }
  .pop-btn { padding: 0 8px; gap: 0; }
}
</style>
